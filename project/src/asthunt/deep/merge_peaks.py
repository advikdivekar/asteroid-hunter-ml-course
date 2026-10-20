"""One object shows up at many neighbouring trial motions: keep only its strongest peak."""

import numpy as np
from scipy.spatial import cKDTree


def merge_peaks(peaks, radius_pixels=3.0):
    """Return the peaks that are the highest SNR within `radius_pixels` of their position.

    Greedy "non-maximum suppression": take the strongest peak, drop every
    other peak within the radius, repeat. Positions are compared at the same
    time (the columns x, y must refer to one common time and grid).
    """
    if len(peaks) == 0:
        return peaks
    peaks = peaks.sort_values("snr", ascending=False).reset_index(drop=True)
    tree = cKDTree(peaks[["x", "y"]].to_numpy(dtype=float))
    removed = np.zeros(len(peaks), dtype=bool)
    keep = []
    for i in range(len(peaks)):
        if removed[i]:
            continue
        keep.append(i)
        removed[tree.query_ball_point(peaks.loc[i, ["x", "y"]].to_numpy(dtype=float), r=radius_pixels)] = True
    return peaks.loc[keep].reset_index(drop=True)
