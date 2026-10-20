"""Add the round-2 tracklet features: static neighbours + forced photometry along the path."""

import pandas as pd

from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS
from asthunt.features.count_static_neighbours import STATIC_COLUMNS, count_static_neighbours
from asthunt.features.measure_forced_photometry import FORCED_COLUMNS, measure_forced_photometry

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + STATIC_COLUMNS + FORCED_COLUMNS


def add_path_features(tracklets, members, detections, path_stamps):
    """Return tracklets with STATIC_COLUMNS and FORCED_COLUMNS added (same row order).

    path_stamps: normalized stamps along each tracklet's predicted path, one row
                 per tracklet in the same order (cut_tracklet_stamps + normalize_stamps).
    Used by BOTH training and the deployed pipeline, so the features can't drift apart.
    """
    tracklets = tracklets.reset_index(drop=True)
    static = count_static_neighbours(tracklets, members, detections).drop(columns="tracklet_id")
    forced = measure_forced_photometry(path_stamps)
    return pd.concat([tracklets, static.reset_index(drop=True), forced], axis=1)
