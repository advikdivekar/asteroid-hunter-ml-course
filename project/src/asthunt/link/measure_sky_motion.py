"""Describe each tracklet by where it is on the sky and how fast it moves there."""

import numpy as np
import pandas as pd


def measure_sky_motion(members, detections):
    """Return one row per tracklet: tracklet_id, jd, ra, dec (at the middle time), v_ra, v_dec ("/hr).

    v_ra already includes cos(dec): it's the true angular speed along RA, so
    v_ra and v_dec can be combined with Pythagoras.
    A straight-line fit in time is used per coordinate: over one night that's
    all the curvature there is room to measure.
    """
    joined = members.merge(detections[["ra", "dec", "obsjd"]], left_on="detection_id", right_index=True)
    rows = []
    for tracklet_id, group in joined.groupby("tracklet_id"):
        middle = group["obsjd"].mean()
        hours = (group["obsjd"] - middle).to_numpy() * 24
        cos_dec = np.cos(np.radians(group["dec"].mean()))
        v_ra, ra = np.polyfit(hours, group["ra"].to_numpy(), 1)
        v_dec, dec = np.polyfit(hours, group["dec"].to_numpy(), 1)
        rows.append({"tracklet_id": tracklet_id, "jd": middle, "ra": ra, "dec": dec,
                     "v_ra": v_ra * 3600 * cos_dec, "v_dec": v_dec * 3600})
    return pd.DataFrame(rows)
