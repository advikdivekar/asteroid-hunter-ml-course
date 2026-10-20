"""Pair tracklets from two different nights that could be the same object."""

import numpy as np
import pandas as pd


def link_tracklets_across_nights(first_night, second_night, velocity_tolerance=0.10,
                                 position_tolerance_fraction=0.15, min_position_tolerance_arcsec=20.0):
    """Return pairs (first_id, second_id, separation, tolerance, velocity_difference) that agree.

    Both tables come from measure_sky_motion (jd, ra, dec, v_ra, v_dec).

    Over a few days an asteroid's sky path CURVES (its speed and direction
    slowly change), so a straight line can't predict night 2 exactly. We:
      1. carry each night-1 tracklet forward to night 2's time with its velocity;
      2. accept a night-2 tracklet within a tolerance that grows with the distance
         travelled: max(min_tolerance, fraction x speed x hours) arcsec;
      3. require similar velocities: |v2 - v1| <= velocity_tolerance x |v1|.
      4. keep each tracklet in at most ONE pair: best first (smallest
         separation/tolerance + velocity difference/velocity tolerance).
    Checked on 5 known asteroids seen 3 days apart (field 618, 2019-01-20/23):
    correct pairs had separations 10-131" and velocity differences 0.5-1.7 "/hr;
    the one wrong pair at 25% velocity tolerance differed by 7.7 "/hr.
    """
    rows = []
    for _, a in first_night.iterrows():
        cos_dec = np.cos(np.radians(a["dec"]))
        speed = np.hypot(a["v_ra"], a["v_dec"])
        for _, b in second_night.iterrows():
            hours = (b["jd"] - a["jd"]) * 24
            predicted_ra = a["ra"] + a["v_ra"] * hours / 3600 / cos_dec
            predicted_dec = a["dec"] + a["v_dec"] * hours / 3600
            separation = np.hypot((b["ra"] - predicted_ra) * cos_dec, b["dec"] - predicted_dec) * 3600
            tolerance = max(min_position_tolerance_arcsec, position_tolerance_fraction * speed * abs(hours))
            velocity_difference = np.hypot(b["v_ra"] - a["v_ra"], b["v_dec"] - a["v_dec"])
            if separation <= tolerance and velocity_difference <= velocity_tolerance * speed:
                rows.append({"first_id": a["tracklet_id"], "second_id": b["tracklet_id"], "separation_arcsec": separation,
                             "tolerance_arcsec": tolerance, "velocity_difference": velocity_difference,
                             "badness": separation / tolerance + velocity_difference / (velocity_tolerance * speed)})
    columns = ["first_id", "second_id", "separation_arcsec", "tolerance_arcsec", "velocity_difference"]
    if not rows:
        return pd.DataFrame(columns=columns)

    # Best pairs first; a tracklet already used can't be paired again
    used_first, used_second, kept = set(), set(), []
    for row in sorted(rows, key=lambda r: r["badness"]):
        if row["first_id"] not in used_first and row["second_id"] not in used_second:
            kept.append(row)
            used_first.add(row["first_id"]); used_second.add(row["second_id"])
    return pd.DataFrame(kept)[columns]
