"""For one night: link (relaxed settings), label, describe, and cut stamps along every tracklet's path."""

import numpy as np
import pandas as pd

from asthunt.clean.load_night_images import load_night_images
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.dataset.normalize_stamps import normalize_stamps
from asthunt.detect.cut_tracklet_stamps import cut_tracklet_stamps
from asthunt.features.compute_tracklet_features import compute_tracklet_features
from asthunt.label.label_tracklets import label_tracklets
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets


def build_tracklet_stamp_dataset(unit, paths, bad_mask_bits, link, realbogus_threshold, min_detections, stamp_size=21,
                                 version_suffix=""):
    """Save and return (tracklets, members, stamps) for one unit that was built with fakes.

    Reads   detections__<tag>.parquet, realbogus_scores__<tag>.npy (stage-1 CNN scores)
            and the planted images (data/interim/night_images/<tag>/)
    Writes  tracklets_mf__<tag>.parquet         features, fitted line, label
            tracklet_members_mf__<tag>.parquet  which detections form each tracklet
            tracklet_stamps_mf__<tag>.npy       (n, frames, 3, size, size) normalized, float16

    version_suffix (e.g. "_v5") reads realbogus_scores_v5__<tag>.npy and writes
    tracklets_mf_v5__..., so a new model version never overwrites an older one's files.

    min_detections (e.g. 3 of 5) and realbogus_threshold (e.g. keep 99% of real
    detections) are looser than v1's: more faint asteroids get linked, and the
    many extra chance alignments become realistic hard negatives to learn from.
    """
    tag = make_unit_tag(unit)
    processed = paths["processed"]
    detections = pd.read_parquet(processed / f"detections__{tag}.parquet")
    detections["score"] = np.load(processed / f"realbogus_scores{version_suffix}__{tag}.npy")

    kept = detections[detections["score"] >= realbogus_threshold]
    kept = kept.assign(x=kept["x_common"], y=kept["y_common"])          # common grid (pointing shifts)
    tracklets, members = link_detections_into_tracklets(
        kept, pixel_scale=link["pixel_scale_arcsec"], min_rate=link["min_rate_arcsec_per_hour"],
        max_rate=link["max_rate_arcsec_per_hour"], match_radius=link["match_radius_pixels"],
        min_detections=min_detections, max_residual=link["max_residual_pixels"])
    tracklets = label_tracklets(tracklets, members, detections)
    tracklets = compute_tracklet_features(tracklets, members, detections)
    tracklets["night"] = unit["night"]
    tracklets["unit"] = tag

    frames = list(detections.groupby("exposure")["obsjd"].first().sort_values().items())
    images = load_night_images(unit, [name for name, _ in frames], paths, bad_mask_bits, with_injections=True)
    stamps = cut_tracklet_stamps(tracklets, frames, images[frames[0][0]]["wcs"], images, stamp_size)
    shape = stamps.shape
    stamps = normalize_stamps(stamps.reshape(-1, *shape[2:])).reshape(shape).astype(np.float16)

    tracklets.to_parquet(processed / f"tracklets_mf{version_suffix}__{tag}.parquet")
    members.to_parquet(processed / f"tracklet_members_mf{version_suffix}__{tag}.parquet")
    np.save(processed / f"tracklet_stamps_mf{version_suffix}__{tag}.npy", stamps)
    return tracklets, members, stamps
