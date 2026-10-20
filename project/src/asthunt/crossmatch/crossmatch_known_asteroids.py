"""Check whether each tracklet is a known solar-system object."""

import numpy as np
import pandas as pd

from asthunt.label.match_detections_to_positions import match_detections_to_positions


def crossmatch_known_asteroids(tracklets, members, detections, known_by_exposure, match_radius_pixels=4.0):
    """Return a copy of `tracklets` with `known_name` ("" if unknown) and `known_matches`.

    known_by_exposure : {exposure name: SkyBoT table with name, x, y} for every frame,
                        with ALL known objects (no brightness cut: a faint known
                        asteroid that we detected is still known).

    For each detection we find the nearest predicted known position in its own
    frame. A tracklet is "known" if at least half of its detections match the
    SAME known object. The radius (4 px = 4") is looser than for fakes,
    because catalogue predictions carry orbit uncertainties.
    """
    detection_names = pd.Series("", index=detections.index)
    for exposure, rows in detections.groupby("exposure"):
        known = known_by_exposure.get(exposure)
        if known is None or len(known) == 0:
            continue
        index = match_detections_to_positions(rows["x"], rows["y"], known["x"], known["y"], match_radius_pixels)
        names = np.where(index >= 0, known["name"].to_numpy()[np.maximum(index, 0)], "")
        detection_names.loc[rows.index] = names

    joined = members.assign(known_name=detection_names.loc[members["detection_id"]].to_numpy())
    result = tracklets.copy()
    best_name, best_count = {}, {}
    for tracklet_id, group in joined.groupby("tracklet_id"):
        counts = group.loc[group["known_name"] != "", "known_name"].value_counts()
        if len(counts) and counts.iloc[0] >= len(group) / 2:
            best_name[tracklet_id], best_count[tracklet_id] = counts.index[0], int(counts.iloc[0])

    result["known_name"] = result["tracklet_id"].map(best_name).fillna("")
    result["known_matches"] = result["tracklet_id"].map(best_count).fillna(0).astype(int)
    return result
