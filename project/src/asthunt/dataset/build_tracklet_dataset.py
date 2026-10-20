"""Link one night's detections into tracklets, then label and describe them."""

import numpy as np
import pandas as pd

from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.features.compute_tracklet_features import compute_tracklet_features
from asthunt.label.label_tracklets import label_tracklets
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets


def build_tracklet_dataset(unit, processed_dir, realbogus_threshold, link):
    """Save and return (tracklets, members) for one unit (one CCD quadrant on one night).

    Reads   detections__<tag>.parquet
            realbogus_scores__<tag>.npy   (CNN score for every detection)
    Writes  tracklets__<tag>.parquet          one row per tracklet: features + label
            tracklet_members__<tag>.parquet   which detections form each tracklet

    Only detections with score >= realbogus_threshold are linked. A lower
    threshold keeps more faint real detections but gives the linker more
    noise to sift through (slower, more false tracklets).
    """
    tag = make_unit_tag(unit)
    detections = pd.read_parquet(processed_dir / f"detections__{tag}.parquet")
    detections["score"] = np.load(processed_dir / f"realbogus_scores__{tag}.npy")

    kept = detections[detections["score"] >= realbogus_threshold]
    # Link on the common (frame 0) grid: the telescope pointing shifts between exposures
    kept = kept.assign(x=kept["x_common"], y=kept["y_common"])
    tracklets, members = link_detections_into_tracklets(
        kept, pixel_scale=link["pixel_scale_arcsec"],
        min_rate=link["min_rate_arcsec_per_hour"], max_rate=link["max_rate_arcsec_per_hour"],
        match_radius=link["match_radius_pixels"], min_detections=link["min_detections"],
        max_residual=link["max_residual_pixels"])

    if len(tracklets):
        tracklets = label_tracklets(tracklets, members, detections)
        tracklets = compute_tracklet_features(tracklets, members, detections)
    tracklets["night"] = unit["night"]
    tracklets["unit"] = tag

    tracklets.to_parquet(processed_dir / f"tracklets__{tag}.parquet")
    members.to_parquet(processed_dir / f"tracklet_members__{tag}.parquet")
    return tracklets, members
