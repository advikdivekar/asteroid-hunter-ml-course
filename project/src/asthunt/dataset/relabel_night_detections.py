"""Re-apply labelling to a night that was already built (when label sources improve)."""

import pandas as pd

from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.label.label_detections import label_detections
from asthunt.label.load_known_asteroids import load_known_asteroids


def relabel_night_detections(unit, paths, inject):
    """Recompute injection_id, known_asteroid and label for a saved unit; save and return the table.

    Detections, stamps and planted fakes are unchanged; only the labels are
    recomputed with the current known-asteroid positions and match radii.
    Row order is kept exactly, so stamps__ files still line up with the rows.
    """
    tag = make_unit_tag(unit)
    detections_file = paths["processed"] / f"detections__{tag}.parquet"
    detections = pd.read_parquet(detections_file)
    injections = pd.read_parquet(paths["processed"] / f"injections__{tag}.parquet")

    parts = []
    for frame_index, rows in detections.groupby("frame_index", sort=False):
        known = load_known_asteroids(rows["exposure"].iloc[0], paths["interim"])
        known = known[known["v_mag"] <= inject["known_asteroid_max_v_mag"]]
        fakes = injections[injections["frame_index"] == frame_index]
        unlabelled = rows.drop(columns=["injection_id", "known_asteroid", "label"])
        parts.append(label_detections(unlabelled, fakes, known, inject["match_radius_pixels"],
                                      inject["known_match_radius_pixels"])[detections.columns])

    relabelled = pd.concat(parts).loc[detections.index]          # original row order
    relabelled.to_parquet(detections_file)
    return relabelled
