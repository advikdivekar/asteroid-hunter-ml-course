"""Does a fast tracklet run along a CCD column, the way bleed-trail residuals do?"""

import numpy as np

COLUMN_ANGLE = 90.0                  # the CCD column direction in the image's x-y axes (streak angles are measured the same way)


def flag_column_aligned(tracklets, members, detections, max_angle_from_column, max_column_spread):
    """Return tracklets with column_angle_difference, column_spread and `column_aligned` (True/False) columns.

    Bleed trails of saturated stars run along CCD columns, and their residuals can line up across frames into a
    "consistent" fast track (six of the seven fast tracks looked at on field 567 did). Such a track
      - moves within `max_angle_from_column` degrees of the column direction, and
      - stays on one column: the spread (std, pixels) of its detections' x is at most `max_column_spread`.
    A real object that happens to move along the columns looks the same, so flagging costs those (about
    2 x max_angle / 180 of random directions): the flag is a column, and removing flagged tracks is a separate decision.
    detections needs x_common (the column on the common grid), indexed by detection_id.
    """
    result = tracklets.copy()
    motion_angle = np.degrees(np.arctan2(result["vy"], result["vx"])) % 180
    result["column_angle_difference"] = np.abs(motion_angle - COLUMN_ANGLE)
    joined = members.merge(detections[["x_common"]], left_on="detection_id", right_index=True)
    spread = joined.groupby("tracklet_id")["x_common"].std(ddof=0)
    result["column_spread"] = result["tracklet_id"].map(spread)
    result["column_aligned"] = (result["column_angle_difference"] <= max_angle_from_column) & (result["column_spread"] <= max_column_spread)
    return result
