"""Join detections from different frames of one night into straight-line tracklets."""

import itertools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from asthunt.link.fit_linear_motion import fit_linear_motion


def link_detections_into_tracklets(detections, pixel_scale, min_rate, max_rate, match_radius,
                                   min_detections, max_residual):
    """Return (tracklets, members).

    detections : one night; needs columns x, y, frame_index, obsjd.
                 x, y must be on ONE common grid for all frames (the pointing
                 shifts between exposures, so use x_common, y_common).
                 Its index is used as the detection id.
    tracklets  : one row per tracklet: tracklet_id, n_detections,
                 rate_arcsec_per_hour, vx, vy (pixels/hour), max_residual,
                 and the fitted line: x0, y0 (position at time t0_jd)
    members    : one row per (tracklet_id, detection_id)

    Method ("seed and predict"):
      1. Seeds: every pair of detections in two different frames whose
         implied speed lies between min_rate and max_rate ("/hr).
      2. Predict: a seed fixes a straight line, so we know where the
         object should be in every other frame. A KD-tree finds the
         nearest detection to each predicted position (within match_radius).
      3. Keep: candidates with >= min_detections detections whose straight-line
         fit has max_residual <= the limit AND whose fitted speed is within
         [min_rate, max_rate]. (The seed check alone isn't enough: a stationary
         source with a wobbling centroid can make a "fast" seed pair but has
         almost zero fitted speed.)
      4. Clean up: many seeds find the same tracklet. We take tracklets
         best-first (most detections, then smallest residual) and drop any
         that re-use a detection already taken.
    """
    frames = sorted(detections["frame_index"].unique())
    times = {f: detections.loc[detections["frame_index"] == f, "obsjd"].iloc[0] for f in frames}
    per_frame = {f: detections[detections["frame_index"] == f] for f in frames}
    trees = {f: cKDTree(per_frame[f][["x", "y"]].to_numpy()) for f in frames if len(per_frame[f])}

    candidates = []
    for first, second in itertools.combinations(frames, 2):
        if first not in trees or second not in trees:
            continue
        hours = (times[second] - times[first]) * 24
        seeds_first, seeds_second = _find_seed_pairs(per_frame[first], trees[second], hours, pixel_scale, min_rate, max_rate)
        if len(seeds_first) == 0:
            continue

        # Velocity of every seed, in pixels per hour
        start = per_frame[first][["x", "y"]].to_numpy()[seeds_first]
        end = per_frame[second][["x", "y"]].to_numpy()[seeds_second]
        velocity = (end - start) / hours

        # For each seed: which detection (index in the night table) is used in each frame, or -1
        chosen = np.full((len(seeds_first), len(frames)), -1, dtype=np.int64)
        chosen[:, frames.index(first)] = per_frame[first].index.to_numpy()[seeds_first]
        chosen[:, frames.index(second)] = per_frame[second].index.to_numpy()[seeds_second]
        for other in frames:
            if other in (first, second) or other not in trees:
                continue
            predicted = start + velocity * (times[other] - times[first]) * 24
            distance, nearest = trees[other].query(predicted, distance_upper_bound=match_radius)
            found = np.isfinite(distance)
            chosen[found, frames.index(other)] = per_frame[other].index.to_numpy()[nearest[found]]

        enough = (chosen >= 0).sum(axis=1) >= min_detections
        candidates.append(chosen[enough])

    if not candidates or sum(len(c) for c in candidates) == 0:
        return _empty_tracklets(), pd.DataFrame(columns=["tracklet_id", "detection_id"])

    # The same tracklet is found by many seeds: keep each distinct set of detections once
    unique_candidates = np.unique(np.concatenate(candidates), axis=0)

    fitted = []
    for row in unique_candidates:
        ids = row[row >= 0]
        members = detections.loc[ids]
        t0_jd = members["obsjd"].min()
        fit = fit_linear_motion((members["obsjd"] - t0_jd) * 24, members["x"], members["y"])
        rate = np.hypot(fit["vx"], fit["vy"]) * pixel_scale
        if fit["max_residual"] <= max_residual and min_rate <= rate <= max_rate:
            fitted.append((ids, {**fit, "t0_jd": t0_jd}))

    return _pick_best_without_sharing(fitted, pixel_scale)


def _find_seed_pairs(first_frame, second_tree, hours, pixel_scale, min_rate, max_rate):
    """All (detection in first frame, detection in second frame) pairs at a plausible speed.

    sparse_distance_matrix returns every pair closer than max_distance in one
    array, computed inside the KD-tree code: no Python loop over pairs, which
    matters when there are millions of them (e.g. stacks hours apart).
    """
    min_distance = min_rate * hours / pixel_scale
    max_distance = max_rate * hours / pixel_scale
    first_tree = cKDTree(first_frame[["x", "y"]].to_numpy())
    pairs = first_tree.sparse_distance_matrix(second_tree, max_distance, output_type="ndarray")
    fast_enough = pairs["v"] >= min_distance
    return pairs["i"][fast_enough].astype(int), pairs["j"][fast_enough].astype(int)


def _pick_best_without_sharing(fitted, pixel_scale):
    """Best-first selection so that no detection belongs to two tracklets."""
    fitted.sort(key=lambda item: (-len(item[0]), item[1]["max_residual"]))
    used = set()
    tracklet_rows, member_rows = [], []
    for ids, fit in fitted:
        if used.intersection(ids):
            continue
        used.update(ids)
        tracklet_id = len(tracklet_rows)
        tracklet_rows.append({
            "tracklet_id": tracklet_id,
            "n_detections": len(ids),
            "rate_arcsec_per_hour": np.hypot(fit["vx"], fit["vy"]) * pixel_scale,
            "vx": fit["vx"],
            "vy": fit["vy"],
            "max_residual": fit["max_residual"],
            "x0": fit["x0"],
            "y0": fit["y0"],
            "t0_jd": fit["t0_jd"],
        })
        member_rows += [{"tracklet_id": tracklet_id, "detection_id": int(i)} for i in ids]

    if not tracklet_rows:
        return _empty_tracklets(), pd.DataFrame(columns=["tracklet_id", "detection_id"])
    return pd.DataFrame(tracklet_rows), pd.DataFrame(member_rows)


def _empty_tracklets():
    return pd.DataFrame(columns=["tracklet_id", "n_detections", "rate_arcsec_per_hour", "vx", "vy", "max_residual",
                                 "x0", "y0", "t0_jd"])
