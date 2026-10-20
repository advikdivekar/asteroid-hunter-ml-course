"""How many detections sit at the SAME sky position in other frames? (Movers: none. Artifacts: many.)"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

STATIC_COLUMNS = ["static_neighbours_mean", "static_neighbours_max"]


def count_static_neighbours(tracklets, members, detections, radius=2.0):
    """Return a table (tracklet_id + STATIC_COLUMNS).

    For every detection in a tracklet, count detections in OTHER frames within
    `radius` pixels of the same position on the common grid. A real asteroid
    moves away, so nothing should be left behind at its old spots. A "tracklet"
    that walks along a persistent artifact (a bad column, a star's residual, a
    faint line present in every frame) finds detections at its own positions in
    the other frames too.

    The tracklet's OWN detections are never counted: a slow asteroid moves only
    1-3 px per frame, so its own other detections fall inside the circle.
    (A first version forgot this and flagged 22% of real tracklets.)
    """
    positions = detections[["x_common", "y_common"]].to_numpy()
    frame_of = detections["frame_index"].to_numpy()
    ids = detections.index.to_numpy()
    tree = cKDTree(positions)
    row_of = {detection_id: row for row, detection_id in enumerate(ids)}
    own = members.groupby("tracklet_id")["detection_id"].apply(set).to_dict()

    counts = []
    for tracklet_id, detection_id in zip(members["tracklet_id"], members["detection_id"]):
        row = row_of[detection_id]
        nearby = tree.query_ball_point(positions[row], r=radius)
        others = [n for n in nearby if frame_of[n] != frame_of[row] and ids[n] not in own[tracklet_id]]
        counts.append(len(others))

    per_member = members.assign(static_neighbours=counts)
    grouped = per_member.groupby("tracklet_id")["static_neighbours"]
    table = pd.DataFrame({"static_neighbours_mean": grouped.mean(), "static_neighbours_max": grouped.max()})
    return tracklets[["tracklet_id"]].merge(table, left_on="tracklet_id", right_index=True, how="left").fillna(0)
