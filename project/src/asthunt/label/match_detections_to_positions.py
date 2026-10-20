"""Pair each detection with the nearest known position (if close enough)."""

import numpy as np
from scipy.spatial import cKDTree


def match_detections_to_positions(detection_x, detection_y, position_x, position_y, max_distance):
    """Return, for every detection, the index of the matched position or -1.

    A KD-tree finds the nearest position for every detection quickly
    (instead of comparing every detection with every position).
    If the nearest one is further than `max_distance` pixels: no match (-1).

    If two detections both sit near the same position, both are matched to
    it; the caller decides what to do with duplicates.
    """
    matches = np.full(len(detection_x), -1, dtype=int)
    if len(position_x) == 0 or len(detection_x) == 0:
        return matches

    tree = cKDTree(np.column_stack([position_x, position_y]))
    distance, nearest = tree.query(np.column_stack([detection_x, detection_y]))

    close_enough = distance <= max_distance
    matches[close_enough] = nearest[close_enough]
    return matches
