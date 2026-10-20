"""Run the point-source search on one night, one stage at a time, and report every stage to the live viewer.

It calls the same functions in the same order as dataset/build_night_detections.py and pipeline/run_night_pipeline.py
(tests/viewer checks the tracks come out the same); the difference is that each step stops to describe and draw what it did.
"""

import time

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.wcs import WCS

from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.clean.load_exposure import load_exposure
from asthunt.crossmatch.crossmatch_known_asteroids import crossmatch_known_asteroids
from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.dataset.normalize_stamps import normalize_stamps
from asthunt.detect.cut_tracklet_stamps import cut_tracklet_stamps
from asthunt.detect.cut_triplet_stamps import cut_triplet_stamps
from asthunt.detect.extract_sources import extract_sources
from asthunt.features.add_path_features import add_path_features
from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS, compute_tracklet_features
from asthunt.label.label_detections import label_detections
from asthunt.label.label_tracklets import label_tracklets
from asthunt.label.load_known_asteroids import load_known_asteroids
from asthunt.label.query_known_asteroids import query_known_asteroids
from asthunt.label.refine_missing_known_positions import refine_missing_known_positions
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets
from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.simulate.plan_injections import plan_injections
from asthunt.simulate.plant_fakes_in_exposure import plant_fakes_in_exposure
from asthunt.viewer import draw_stage_figures as draw

STAGE_PLAN = [
    ("pick", "1. Pick the night",
     "The telescope takes a photo of the same patch of sky every few minutes all night. This step picks one night and takes the first 5 photos of one patch, in time order. An asteroid creeps across the sky, so we need several photos of the same patch to see anything move.",
     [("exposure", "one photo; ZTF exposes each for 30 seconds"),
      ("field / CCD / quadrant", "the patch of sky, and which piece of the camera took it (the camera has 16 chips, each read out in 4 quarters)"),
      ("UT", "Universal Time, the clock astronomers use")]),
    ("open", "2. Open the files of each photo",
     "Each photo comes as 4 files. SCIENCE is tonight's photo. REFERENCE is an old, very deep photo of the same sky. DIFFERENCE is science minus reference: stars are the same in both, so they cancel and vanish. What is left is only what changed: moving things, new things, and mistakes of the subtraction. The MASK marks pixels we must not trust (broken or overexposed ones).",
     [("pixel", "one tiny square of a picture; each photo here is about 3,000 x 3,000 of them"),
      ("mask", "a map of broken or overexposed pixels that we ignore"),
      ("seeing", "how blurry the sky was, in arcseconds (smaller = sharper)")]),
    ("noise", "3. Measure how noisy the picture is",
     "Every pixel has a small random error called noise. To say 'this spot is too bright to be an accident' we must first know how big an accident can be. We count how many pixels have each brightness and draw the counts as a histogram. Pure noise makes a bell shape centred on zero. Pixels far out on the right are too bright to be noise.",
     [("histogram", "bars that show how many pixels have each value; a tall bar means many pixels"),
      ("bell curve", "the shape random noise makes: most values near the middle, very few far away. The yellow line is the bell that pure noise would make"),
      ("sigma", "the width of the bell, i.e. the typical size of noise. '5 sigma' means 5 times that size: very unlikely to be an accident"),
      ("log scale", "the vertical axis goes 1, 10, 100, 1000... in equal steps, so the rare bright pixels stay visible next to the huge pile near zero")]),
    ("plant", "4. Plant fake asteroids (test mode)",
     "A real night usually contains nothing new, so we could not tell whether the program works. So we secretly paint fake asteroids into the photos. We choose how bright each is and how fast it moves, and each creeps in a straight line across the 5 photos. Later we count how many the program finds. In the table, 'magnitude' is brightness and a BIGGER number means FAINTER.",
     [("magnitude", "a brightness scale where bigger = fainter: about 17 is easy to see, 21.5 is very faint"),
      ("arcsec/hour", "how fast it moves across the sky; 1 arcsecond is 1/3600 of a degree")]),
    ("detect", "5. Find blobs in every difference image",
     "The program scans each difference image and circles every bright spot (a blob). Thousands appear and nearly all are junk: noise, leftovers of bright stars, cosmic-ray hits. LEFT = the whole camera chip, with every planted fake that was found ringed in teal and numbered (a fake with no ring was too faint to make a blob). RIGHT = a zoom into the yellow square.",
     [("blob", "a spot brighter than the noise; a candidate object"),
      ("teal / blue / grey ring", "teal = a fake we planted, blue = a real asteroid already in the catalogue, grey = not labelled (mostly junk)"),
      ("2 sigma", "the brightness cut for calling something a blob: 2 times the typical noise. Low on purpose, so nothing real is missed")]),
    ("stamps", "6. Cut a small picture around every blob",
     "Each blob gets a small square picture (31 x 31 pixels) in three versions side by side: science, reference and difference. A real new object shows in science and difference but NOT in the reference. A badly subtracted star shows in science AND in the reference. These three pictures are what the neural network judges.",
     [("stamp", "a small cut-out picture around one blob"),
      ("SNR", "signal-to-noise ratio: how strong the blob is compared with the noise")]),
    ("cnn", "7. A neural network decides: real or bogus?",
     "A neural network is a program that learned from thousands of labelled examples what real objects and junk look like. For each stamp it gives a score from 0 (certainly junk) to 1 (certainly a real moving object). The histogram counts blobs by score. Nearly all pile up near 0, which is right because most blobs are junk, while our fakes sit near 1. The yellow line is the cut: blobs above it continue, the rest are dropped.",
     [("score / p(real)", "the network's probability that a blob is real, from 0 to 1"),
      ("bogus", "junk: not a real moving object"),
      ("histogram", "bars that count how many blobs have each score. The vertical axis is a log scale, so each step up means 10 times more blobs"),
      ("cut / threshold", "the score above which a blob continues; chosen so about 95% of real objects survive")]),
    ("link", "8. Join blobs into straight-line tracks",
     "An asteroid moves in a straight line at a steady speed. The program looks for blobs in the different photos that line up like that and joins them into a track. The big picture shows every track over the chip; each small picture is a close-up of one good track: one coloured dot per photo (the colour tells which photo) and a grey line fitted through them.",
     [("track / tracklet", "blobs from different photos that line up as one moving object"),
      ("common grid", "the telescope points slightly differently for each photo, so positions are first converted onto one shared sky map before joining")]),
    ("features", "9. Describe each track with numbers",
     "To score a track the computer needs numbers. Each track gets 17: how fast, how straight, how steady the brightness, how sure the neural network was, and what the sky shows at the predicted spot in photos where the object was missed. The picture shows the same patch of sky along the best track, photo after photo.",
     [("feature", "one number that describes something about a track"),
      ("forced photometry", "measuring brightness at the predicted spot even where no blob was found")]),
    ("classify", "10. A second model scores each track",
     "A second model (XGBoost, made of many small decision trees) reads each track's 17 numbers and gives the whole track a score from 0 to 1. The bar chart sorts all tracks by score. The yellow line is the cut. Teal bars are our planted fakes, which should score high.",
     [("XGBoost", "a model made of many small decision trees, each asking yes/no questions about the numbers")]),
    ("known", "11. Is it already known?",
     "Astronomers have catalogued hundreds of thousands of asteroids. Each track is compared with that list. Almost everything we find is already known, which is a good sign: it means the program finds real asteroids by itself. Blue bars = known asteroids, teal = our fakes, pink = nothing known.",
     [("catalogue", "the official list of known asteroids and where each should be"),
      ("candidate", "a track that scored high but matches nothing known; a person must examine it")]),
    ("result", "12. What is left for a human to look at",
     "The final count. 'Fakes recovered' shows how well the whole program works on a known answer. Faint fakes are missed because one 30-second photo simply does not hold enough light from them: that is the limit of this single-photo search. Candidates are high-scoring tracks that match nothing known and are not our fakes. Most nights there are none.",
     [("recovered", "a planted fake that came out as a high-scoring track")]),
]


def run_traced_night(unit, state, configs, bundle, paths, n_fakes, seed, device):
    """Run every stage; state receives the pictures, numbers and tables. Raises if a stage fails (state records which)."""
    began = time.time()
    acquire, inject, detect, link = configs["acquire"], configs["inject"], configs["detect"], configs["link"]
    thresholds = bundle["thresholds"]
    night, tag = unit["night"], make_unit_tag(unit)
    rng = make_unit_rng(seed, unit)

    with state.stage("pick") as s:
        metadata = load_unit_metadata(unit, paths["raw"])
        frames = select_night_exposures(metadata, night, acquire["filter"], acquire["frames_per_night"])
        rows = [[k + 1, Time(frame["obsjd"], format="jd").strftime("%H:%M:%S"), f"{frame['obsjd']:.5f}"] for k, (_, frame) in enumerate(frames.iterrows())]
        s.number("night", night)
        s.number("patch of sky", f"field {unit['field']}, CCD {unit['ccd']}, quadrant {unit['quadrant']}")
        s.number("exposures used", len(frames))
        s.table(["frame", "time (UT)", "Julian date"], rows)

    with state.stage("open") as s:
        exposures = []
        for k, (_, frame) in enumerate(frames.iterrows()):
            name = (f"ztf_{frame['filefracday']}_{int(unit['field']):06d}_{acquire['filter']}"
                    f"_c{int(unit['ccd']):02d}_o_q{int(unit['quadrant'])}_sciimg.fits")
            exposures.append(load_exposure(paths["raw"] / "ztf" / night / name, acquire["bad_mask_bits"]))
            s.note(f"frame {k + 1}/{len(frames)} opened: {exposures[-1]['name']}")
        first = exposures[0]
        s.number("image size", f"{first['difference'].shape[1]} x {first['difference'].shape[0]} pixels")
        s.number("pixels masked as bad", f"{100 * first['bad'].mean():.1f}%")
        s.number("seeing (blur) in frame 1", f"{first['header']['SEEING']:.2f} arcsec")
        s.number("brightness zero point", f"{first['magzp']:.2f}")
        s.image("Left to right: tonight's photo, the old reference, their difference (the stars are gone), and the mask (bright = pixels we ignore)", draw.draw_four_panels(first["science"], first["reference"], first["difference"], first["bad"],
                                                                  f"{first['name']}  (shown at reduced size)"))

    with state.stage("noise") as s:
        values = first["difference"][~first["bad"]]
        sigma_robust = 1.4826 * np.median(np.abs(values - np.median(values)))
        s.number("noise sigma (robust)", f"{sigma_robust:.1f} counts")
        s.number("noise sigma (plain std)", f"{values.std():.1f} counts")
        s.number("pixels above +5 sigma", f"{int((values > 5 * sigma_robust).sum()):,}")
        s.note("If the picture were pure noise only about 1 pixel in 3.5 million would exceed +5 sigma. The excess is real blobs and artifacts.")
        s.image("Horizontal: brightness of a pixel. Vertical: how many pixels (log scale). Teal = this photo, yellow curve = pure noise. Pink dashed lines = 3 and 5 sigma: bars to the right of them are too bright to be noise", draw.draw_noise_histogram(values, sigma_robust, values.std()))

    reference_wcs = WCS(first["header"])                   # frame 0's pixel grid is the common grid
    science_frames, difference_frames, injections, fakes_by_frame = [], [], None, []
    with state.stage("plant") as s:
        if n_fakes == 0:
            s.skip("real search: nothing is planted")
        injections = plan_injections(n_fakes, first["science"].shape, [e["obsjd"] for e in exposures], inject["magnitude_range"],
                                     inject["rate_range_arcsec_per_hour"], first["header"]["PIXSCALE"], inject["edge_margin_pixels"], rng)
        for k, exposure in enumerate(exposures):
            if n_fakes == 0:
                science_frames.append(exposure["science"])
                difference_frames.append(exposure["difference"])
                fakes_by_frame.append(injections[injections["frame_index"] == k])
                continue
            science, difference = exposure["science"].copy(), exposure["difference"].copy()
            fakes_by_frame.append(plant_fakes_in_exposure(exposure, science, difference, injections[injections["frame_index"] == k],
                                                          reference_wcs, WCS(exposure["header"]), rng))
            science_frames.append(science)
            difference_frames.append(difference)
        if n_fakes:
            per_fake = injections.groupby("injection_id").first()
            s.number("fake asteroids planted", n_fakes)
            s.table(["fake #", "brightness (mag)", "speed (arcsec/hour)"],
                    [[int(i), f"{r['mag']:.1f}", f"{r['rate_arcsec_per_hour']:.0f}"] for i, r in per_fake.sort_values("mag").iterrows()][:20])
            ring = pd.DataFrame({"x": fakes_by_frame[0]["x"], "y": fakes_by_frame[0]["y"], "injection_id": fakes_by_frame[0]["injection_id"],
                                 "known_asteroid": ""})
            s.image("Whole camera chip, photo 1: a numbered teal ring marks where each planted fake is (they are tiny on a chip this big)", draw.draw_detections(difference_frames[0], ring, "planted fakes (teal rings)"))

    detections_by_frame = []
    with state.stage("detect") as s:
        for k, exposure in enumerate(exposures):
            found = extract_sources(difference_frames[k], exposure["bad"], detect["threshold_sigma"], detect["min_area_pixels"])
            known = _known_asteroids(exposure, paths, s)
            known = known[known["v_mag"] <= inject["known_asteroid_max_v_mag"]]
            found = label_detections(found, fakes_by_frame[k], known, inject["match_radius_pixels"], inject["known_match_radius_pixels"])
            frame_wcs = WCS(exposure["header"])
            sky = frame_wcs.pixel_to_world(found["x"].to_numpy(), found["y"].to_numpy())
            found["ra"], found["dec"] = sky.ra.deg, sky.dec.deg
            found["x_common"], found["y_common"] = reference_wcs.world_to_pixel(sky)
            found["mag"] = exposure["magzp"] - 2.5 * np.log10(found["flux"].clip(lower=1e-3))
            found["night"], found["unit"], found["frame_index"] = night, tag, k
            found["exposure"], found["obsjd"] = exposure["name"], exposure["obsjd"]
            found["seeing"], found["maglim"] = exposure["header"]["SEEING"], exposure["header"]["MAGLIM"]
            detections_by_frame.append(found)
            s.note(f"frame {k + 1}/{len(exposures)}: {len(found):,} blobs found")
        s.table(["frame", "blobs found"], [[k + 1, f"{len(d):,}"] for k, d in enumerate(detections_by_frame)])
        s.number("blobs in all frames", f"{sum(len(d) for d in detections_by_frame):,}")
        if n_fakes:
            found_in_frame_1 = int(detections_by_frame[0].loc[detections_by_frame[0]["injection_id"] >= 0, "injection_id"].nunique())
            s.number("planted fakes found as a blob in photo 1", f"{found_in_frame_1} of {n_fakes}")
            s.note("A planted fake that is too faint to make a blob gets no ring: the rest of the pipeline never sees it.")
        s.image("Left: whole chip; teal numbered rings = planted fakes that were found, blue = known asteroids, yellow square = the zoom. Right: the zoom with every blob ringed (grey = unlabelled, mostly junk)",
                draw.draw_detections(difference_frames[0], detections_by_frame[0], f"{len(detections_by_frame[0]):,} blobs in frame 1"))

    detections = pd.concat(detections_by_frame, ignore_index=True)
    with state.stage("stamps") as s:
        all_stamps = []
        for k, exposure in enumerate(exposures):
            rows = detections[detections["frame_index"] == k]
            stamps = cut_triplet_stamps(science_frames[k], science_frames[k] - difference_frames[k], difference_frames[k],
                                        rows["x"].to_numpy(), rows["y"].to_numpy(), detect["stamp_size"])
            all_stamps.append(np.clip(stamps, -65000, 65000).astype(np.float16))
        stamps = np.concatenate(all_stamps)
        order = np.argsort(-detections["snr"].to_numpy())
        pick = list(np.where(detections["injection_id"].to_numpy() >= 0)[0][:4]) + list(np.where(detections["known_asteroid"].to_numpy() != "")[0][:4])
        pick += [i for i in order if i not in pick][:12 - len(pick)]
        captions = [f"frame {int(detections.loc[i, 'frame_index']) + 1}, SNR {detections.loc[i, 'snr']:.1f}"
                    + (" - fake" if detections.loc[i, "injection_id"] >= 0 else " - known" if detections.loc[i, "known_asteroid"] else "") for i in pick]
        s.number("stamps cut", f"{len(stamps):,}")
        s.number("size of each", f"3 x {detect['stamp_size']} x {detect['stamp_size']} pixels")
        s.image("Each group of three = science | reference | difference for one blob. Top: planted fakes. Middle: known asteroids. Bottom: the brightest blobs (mostly leftovers of stars)", draw.draw_stamp_mosaic(stamps[pick].astype(np.float32), captions, "A few of the stamps (planted fakes, known asteroids, then the brightest blobs)"))
        stamps = normalize_stamps(stamps)                  # each channel in units of its own noise, as the CNN was trained

    with state.stage("cnn") as s:
        detections["score"] = predict_realbogus(bundle["realbogus_model"], stamps, device)
        scores = detections["score"].to_numpy()
        cut = thresholds["realbogus"]
        kept_mask = scores >= cut
        fake_mask, known_mask = detections["injection_id"].to_numpy() >= 0, (detections["known_asteroid"].to_numpy() != "") & ~(detections["injection_id"].to_numpy() >= 0)
        s.number("detections scored", f"{len(scores):,}")
        s.number("cut chosen on validation nights", f"{cut:.3f} (keeps about 95% of real objects)")
        s.number("kept for linking", f"{int(kept_mask.sum()):,} ({100 * kept_mask.mean():.0f}%)")
        if fake_mask.any():
            s.number("planted-fake detections kept", f"{int((kept_mask & fake_mask).sum())} of {int(fake_mask.sum())}")
        if known_mask.any():
            s.number("known-asteroid detections kept", f"{int((kept_mask & known_mask).sum())} of {int(known_mask.sum())}")
        s.image("Horizontal: the network's score (0 = junk, 1 = real). Vertical: number of blobs, log scale. Grey = all blobs, teal = planted fakes, blue = known asteroids, yellow line = the cut", draw.draw_score_histogram(scores, cut, [("planted fakes", draw.TEAL, fake_mask), ("known asteroids", draw.BLUE, known_mask)]))
        best, worst = np.argsort(-scores)[:8], np.argsort(scores)[:8]
        s.image("The blobs the network is most sure are REAL (score near 1): round, sharp, new points", draw.draw_stamp_mosaic(stamps[best], [f"p(real) = {scores[i]:.3f}" for i in best], "Highest scores", columns=4))
        s.image("The blobs the network is most sure are JUNK (score near 0): look how plain they are", draw.draw_stamp_mosaic(stamps[worst], [f"p(real) = {scores[i]:.4f}" for i in worst], "Lowest scores", columns=4))

    kept = detections[detections["score"] >= thresholds["realbogus"]]
    kept = kept.assign(x=kept["x_common"], y=kept["y_common"])
    with state.stage("link") as s:
        tracklets, members = link_detections_into_tracklets(
            kept, pixel_scale=link["pixel_scale_arcsec"], min_rate=link["min_rate_arcsec_per_hour"], max_rate=link["max_rate_arcsec_per_hour"],
            match_radius=link["match_radius_pixels"], min_detections=thresholds.get("min_detections", link["min_detections"]),
            max_residual=link["max_residual_pixels"])
        s.number("detections searched", f"{len(kept):,}")
        s.number("tracks found", len(tracklets))
        s.number("rule", f"at least {thresholds.get('min_detections', link['min_detections'])} of {len(exposures)} frames, speed {link['min_rate_arcsec_per_hour']:.0f}-{link['max_rate_arcsec_per_hour']:.0f} arcsec/hour, within {link['max_residual_pixels']} px of a straight line")
        if len(tracklets):
            s.image("Left: all tracks over the chip (teal). Right: close-ups of the 8 best tracks, one dot per photo, joined by the fitted straight line", draw.draw_tracklets(kept, members, tracklets, len(exposures), "Linking on the common sky grid (colour = frame)"))

    result = {"detections": detections, "tracklets": tracklets, "members": members, "candidates": tracklets, "injections": injections}
    if len(tracklets) == 0:
        for stage_id in ["features", "classify", "known"]:
            with state.stage(stage_id) as s:
                s.skip("no tracks were found, so there is nothing to describe")
        _finish(state, result, injections, n_fakes, thresholds, began)
        return result

    with state.stage("features") as s:
        tracklets = compute_tracklet_features(tracklets, members, detections)
        frames_in_time = list(detections.groupby("exposure")["obsjd"].first().sort_values().items())
        images = {e["name"]: {"science": science_frames[k], "difference": difference_frames[k], "reference": science_frames[k] - difference_frames[k],
                              "wcs": WCS(e["header"])} for k, e in enumerate(exposures)}
        path_stamps = cut_tracklet_stamps(tracklets, frames_in_time, images[frames_in_time[0][0]]["wcs"], images)
        path_stamps = normalize_stamps(path_stamps.reshape(-1, *path_stamps.shape[2:])).reshape(path_stamps.shape)
        tracklets = add_path_features(tracklets, members, detections, path_stamps)
        show = [c for c in ["n_detections", "rate_arcsec_per_hour", "max_residual", "mag_mean", "mag_std", "snr_min", "score_min", "stack_snr"] if c in tracklets]
        s.number("numbers per track", len(bundle["info"].get("feature_columns", FEATURE_COLUMNS)))
        s.table(["track"] + show, [[int(r["tracklet_id"])] + [f"{r[c]:.2f}" for c in show] for _, r in tracklets.head(12).iterrows()])
        best_track = tracklets["n_detections"].idxmax()
        s.image("The best track, photo after photo: the same patch of sky at the predicted spot, whether or not a blob was found there",
                draw.draw_stamp_mosaic(path_stamps[best_track], [f"frame {k + 1}" for k in range(path_stamps.shape[1])],
                                       f"Track #{int(tracklets.loc[best_track, 'tracklet_id'])}: stamps along its predicted path", columns=path_stamps.shape[1]))

    with state.stage("classify") as s:
        feature_columns = bundle["info"].get("feature_columns", FEATURE_COLUMNS)
        tracklets["tracklet_score"] = bundle["tracklet_model"].predict_proba(tracklets[feature_columns].to_numpy())[:, 1]
        tracklets = label_tracklets(tracklets, members, detections)
        has_fake = members.merge(detections[["injection_id"]], left_on="detection_id", right_index=True).groupby("tracklet_id")["injection_id"].max() >= 0
        tracklets["contains_fake"] = tracklets["tracklet_id"].map(has_fake).fillna(False)
        tracklets["known_name"] = ""
        s.number("tracks scored", len(tracklets))
        s.number("cut chosen on validation nights", f"{thresholds['tracklet']:.3f}")
        s.number("tracks above the cut", int((tracklets["tracklet_score"] >= thresholds["tracklet"]).sum()))
        s.image("Each bar is one track, sorted by score (right = more likely real). Yellow line = the cut", draw.draw_tracklet_scores(tracklets, thresholds["tracklet"], "Score of every track (teal = planted fake; the rest not yet compared with the catalogue)"))

    with state.stage("known") as s:
        night_folder = paths["raw"] / "ztf" / night
        try:
            refine_missing_known_positions([night_folder / f"{name}_sciimg.fits" for name in detections["exposure"].unique()], paths["interim"])
        except Exception as error:                      # no network: use whatever positions are already on disk
            s.note(f"Horizons refinement skipped ({type(error).__name__}); using the positions already on disk")
        known = pd.concat([load_known_asteroids(name, paths["interim"]).assign(exposure=name) for name in detections["exposure"].unique()], ignore_index=True)
        known_by_exposure = {name: rows for name, rows in known.groupby("exposure")}
        tracklets = crossmatch_known_asteroids(tracklets, members, detections, known_by_exposure)
        names = tracklets.loc[tracklets["known_name"] != "", "known_name"].unique()
        s.number("known asteroids in the field (all brightnesses)", f"{known['name'].nunique():,}")
        s.number("tracks that are a known asteroid", f"{int((tracklets['known_name'] != '').sum())} of {len(tracklets)}")
        s.table(["track", "score", "matches"], [[int(r["tracklet_id"]), f"{r['tracklet_score']:.3f}",
                                                  "planted fake" if r["contains_fake"] else r["known_name"] or "nothing known"]
                                                 for _, r in tracklets.sort_values("tracklet_score", ascending=False).head(15).iterrows()])
        s.image("Same bars, now coloured: teal = our fake, blue = known asteroid, pink = nothing known", draw.draw_tracklet_scores(tracklets, thresholds["tracklet"], "Score of every track (teal = planted fake, blue = known asteroid, pink = nothing known)"))
        if len(names):
            s.note("known asteroids recovered: " + ", ".join(sorted(names)))

    result.update({"tracklets": tracklets, "known": known})
    _finish(state, result, injections, n_fakes, thresholds, began)
    return result


def _known_asteroids(exposure, paths, stage):
    """Known asteroids of one frame from the cache; ask the catalogue only if it was never fetched (and carry on if that fails)."""
    cache = paths["interim"] / "known_asteroids" / f"known_asteroids__{exposure['name']}.csv"
    try:
        if not cache.exists():
            query_known_asteroids(exposure["header"], exposure["difference"].shape, cache_file=cache)
        return load_known_asteroids(exposure["name"], paths["interim"])
    except Exception as error:
        stage.note(f"known-asteroid lookup failed for {exposure['name']} ({type(error).__name__}): continuing without labels")
        return pd.DataFrame({"name": [], "x": [], "y": [], "v_mag": []})


def _finish(state, result, injections, n_fakes, thresholds, began):
    """The last stage: candidates, and how many planted fakes the system found."""
    tracklets, members, detections = result["tracklets"], result["members"], result["detections"]
    with state.stage("result") as s:
        if "tracklet_score" in tracklets:
            candidates = tracklets[(tracklets["tracklet_score"] >= thresholds["tracklet"]) & (tracklets["known_name"] == "") & (~tracklets["contains_fake"])]
            result["candidates"] = candidates.sort_values("tracklet_score", ascending=False)
        else:
            result["candidates"] = tracklets
        s.number("candidates for a human to examine", len(result["candidates"]))
        if n_fakes:
            passing = tracklets[tracklets["tracklet_score"] >= thresholds["tracklet"]]["tracklet_id"] if "tracklet_score" in tracklets else []
            joined = members[members["tracklet_id"].isin(passing)].merge(detections[["injection_id"]], left_on="detection_id", right_index=True)
            counts = joined[joined["injection_id"] >= 0].groupby("injection_id").size()
            found = set(counts[counts >= 3].index)
            per_fake = injections.groupby("injection_id").first()
            s.number("planted fakes recovered", f"{len(found)} of {n_fakes}")
            s.table(["fake #", "brightness (mag)", "speed (arcsec/hour)", "recovered?"],
                    [[int(i), f"{r['mag']:.1f}", f"{r['rate_arcsec_per_hour']:.0f}", "yes" if i in found else "no"] for i, r in per_fake.sort_values("mag").iterrows()][:25])
            s.note("The faint fakes are missed because a single 30-second frame cannot show them: the depth limit of this single-frame search.")
        if len(result["candidates"]):
            s.table(["track", "score", "speed (arcsec/hour)", "frames"], [[int(r["tracklet_id"]), f"{r['tracklet_score']:.3f}", f"{r['rate_arcsec_per_hour']:.0f}", int(r["n_detections"])]
                                                                            for _, r in result["candidates"].iterrows()])
        else:
            s.note("No unexplained track: everything that was found is a known asteroid or one of our fakes. That is the usual outcome.")
        s.number("total time", f"{time.time() - began:.0f} s")
    state.finish(f"done in {time.time() - began:.0f} s")
