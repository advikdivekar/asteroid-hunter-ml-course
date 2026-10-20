"""Does a fast tracklet's motion agree with the shape of its streaks?"""

import numpy as np


def check_streak_consistency(tracklets, members, detections, exposure_seconds, pixel_scale,
                             max_angle_difference, length_ratio_range, max_length_spread=None):
    """Return tracklets with angle_difference, length_ratio and `consistent` (True/False) columns.

    A real fast object leaves a trail ALONG its direction of motion, and the
    trail's length equals the distance moved during one exposure:
        expected length (px) = rate ("/hr) x exposure (s) / 3600 / pixel scale
    Random streaks (satellites, cosmic rays, artifacts) that happen to line up
    across frames almost never satisfy both. Angles are compared modulo 180,
    because a single image can't tell which end of a trail is the front.
    One object also keeps one trail length: length_spread = std / mean of the
    lengths across frames must stay below max_length_spread (if given).
    """
    result = tracklets.copy()
    joined = members.merge(detections[["angle", "length"]], left_on="detection_id", right_index=True)
    motion_angle = np.degrees(np.arctan2(result["vy"], result["vx"])) % 180
    expected_length = result["rate_arcsec_per_hour"] * exposure_seconds / 3600 / pixel_scale

    angle_differences, length_ratios, length_spreads = [], [], []
    for (tracklet_id, group), angle, expected in zip(joined.groupby("tracklet_id"), motion_angle, expected_length):
        difference = np.abs(((group["angle"] - angle) + 90) % 180 - 90)
        angle_differences.append(float(difference.mean()))
        length_ratios.append(float(group["length"].median() / expected) if expected > 0 else np.inf)
        length_spreads.append(float(group["length"].std(ddof=0) / group["length"].mean()))

    result["angle_difference"] = angle_differences
    result["length_ratio"] = length_ratios
    result["length_spread"] = length_spreads
    low, high = length_ratio_range
    result["consistent"] = (result["angle_difference"] <= max_angle_difference) & result["length_ratio"].between(low, high)
    if max_length_spread is not None:
        result["consistent"] &= result["length_spread"] <= max_length_spread
    return result
