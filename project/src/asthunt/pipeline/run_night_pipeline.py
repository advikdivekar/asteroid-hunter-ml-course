"""Run the whole search on one night: images in, ranked candidate list and report out."""

import pandas as pd

from asthunt.crossmatch.crossmatch_known_asteroids import crossmatch_known_asteroids
from asthunt.dataset.build_night_detections import build_night_detections
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.dataset.normalize_stamps import normalize_stamps
from asthunt.clean.load_night_images import load_night_images
from asthunt.detect.cut_tracklet_stamps import cut_tracklet_stamps
from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS, compute_tracklet_features
from asthunt.features.add_path_features import add_path_features
from asthunt.label.label_tracklets import label_tracklets
from asthunt.label.load_known_asteroids import load_known_asteroids
from asthunt.label.refine_missing_known_positions import refine_missing_known_positions
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets
from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.report.write_ades_report import write_ades_report
from asthunt.report.write_candidate_report import write_candidate_report


def run_night_pipeline(unit, metadata, configs, bundle, paths, rng, device="cpu",
                       with_injections=False, write_reports=True):
    """Return a dict of tables for one unit (one CCD quadrant on one night) and (optionally) write the report.

    unit    : {field, ccd, quadrant, night}
    metadata: the archive's exposure table for that field/CCD/quadrant (acquire/load_unit_metadata.py)

    configs : {"acquire": ..., "inject": ..., "detect": ..., "link": ...} (the YAML files)
    bundle  : a loaded model bundle (see models/load_model_bundle.py): the CNN, the
              tracklet classifier and the thresholds of ONE pipeline version

    with_injections=True plants fakes and labels everything: the evaluation run.
    with_injections=False is the real search. Both go through exactly the same
    steps below, so what we evaluate is what we deploy.

    Steps:
      1. detect (and label, if fakes are planted) every frame of the night
      2. score every detection with the real/bogus CNN
      3. link detections that pass the CNN threshold (on the common sky grid)
      4. describe tracklets (motion, brightness, CNN opinion, static neighbours,
         forced photometry along the predicted path) and score them
      5. cross-match with ALL known asteroids (positions refined with Horizons first)
      6. candidates = good tracklets that match nothing known (and no fake)
      7. write the HTML report and a draft ADES file to reports/<unit tag>/

    Returned keys: detections, tracklets, members, candidates, injections, known
    """
    acquire, link = configs["acquire"], configs["link"]
    thresholds = bundle["thresholds"]

    # 1. Detections and stamps
    detections, stamps, injections = build_night_detections(
        unit, metadata, acquire, configs["inject"], configs["detect"], paths, rng, with_injections=with_injections)
    stamps = normalize_stamps(stamps)

    # 2. Real/bogus scores
    detections["score"] = predict_realbogus(bundle["realbogus_model"], stamps, device, detections=detections)

    # Known objects in every frame (all brightnesses). SkyBoT alone can be 4-5 px off on a fresh night,
    # which would make known asteroids look new, so get Horizons positions first (nothing to do if done already)
    night_folder = paths["raw"] / "ztf" / unit["night"]
    refine_missing_known_positions([night_folder / f"{exposure}_sciimg.fits" for exposure in detections["exposure"].unique()],
                                   paths["interim"])
    known = pd.concat([load_known_asteroids(exposure, paths["interim"]).assign(exposure=exposure)
                       for exposure in detections["exposure"].unique()], ignore_index=True)

    # 3. Link only detections the CNN finds plausible, on the common (frame 0) grid:
    #    the telescope pointing shifts between exposures
    kept = detections[detections["score"] >= thresholds["realbogus"]]
    kept = kept.assign(x=kept["x_common"], y=kept["y_common"])
    tracklets, members = link_detections_into_tracklets(
        kept, pixel_scale=link["pixel_scale_arcsec"], min_rate=link["min_rate_arcsec_per_hour"],
        max_rate=link["max_rate_arcsec_per_hour"], match_radius=link["match_radius_pixels"],
        min_detections=thresholds.get("min_detections", link["min_detections"]),   # a bundle may relax it (v2: 3 of 5)
        max_residual=link["max_residual_pixels"])
    result = {"detections": detections, "members": members, "injections": injections, "known": known}
    if len(tracklets) == 0:
        return {**result, "tracklets": tracklets, "candidates": tracklets}

    # 4. Tracklet features and scores (labels too, when we know the truth)
    tracklets = compute_tracklet_features(tracklets, members, detections)

    #    Stamps along each predicted path, in every frame (detected or not), for forced photometry
    frames = list(detections.groupby("exposure")["obsjd"].first().sort_values().items())
    images = load_night_images(unit, [name for name, _ in frames], paths, acquire["bad_mask_bits"], with_injections)
    path_stamps = cut_tracklet_stamps(tracklets, frames, images[frames[0][0]]["wcs"], images)
    path_stamps = normalize_stamps(path_stamps.reshape(-1, *path_stamps.shape[2:])).reshape(path_stamps.shape)
    del images                                       # free ~1 GB
    tracklets = add_path_features(tracklets, members, detections, path_stamps)

    feature_columns = bundle["info"].get("feature_columns", FEATURE_COLUMNS)   # each version names its own features
    tracklets["tracklet_score"] = bundle["tracklet_model"].predict_proba(tracklets[feature_columns].to_numpy())[:, 1]
    tracklets = label_tracklets(tracklets, members, detections)

    # 5. Known objects
    known_by_exposure = {exposure: rows for exposure, rows in known.groupby("exposure")}
    tracklets = crossmatch_known_asteroids(tracklets, members, detections, known_by_exposure)

    # 6. Candidates: high score, not known, not one of our own fakes
    has_fake = members.merge(detections[["injection_id"]], left_on="detection_id", right_index=True) \
                      .groupby("tracklet_id")["injection_id"].max() >= 0
    tracklets["contains_fake"] = tracklets["tracklet_id"].map(has_fake).fillna(False)
    candidates = tracklets[(tracklets["tracklet_score"] >= thresholds["tracklet"])
                           & (tracklets["known_name"] == "") & (~tracklets["contains_fake"])]
    candidates = candidates.sort_values("tracklet_score", ascending=False)

    # 7. Reports
    if write_reports:
        tag = make_unit_tag(unit)
        report_dir = paths["reports"] / tag
        write_candidate_report(candidates, members, detections, stamps, report_dir, f"Asteroid candidates, {tag}")
        write_ades_report(candidates, members, detections, report_dir / "candidates_draft_ades.psv")

    return {**result, "tracklets": tracklets, "candidates": candidates}
