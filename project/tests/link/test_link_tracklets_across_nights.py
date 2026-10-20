import numpy as np
import pandas as pd

from asthunt.link.link_tracklets_across_nights import link_tracklets_across_nights
from asthunt.link.measure_sky_motion import measure_sky_motion


def curved_path(t_hours):
    """A main-belt-like path: -35 "/hr in RA, slowly changing (curving) over days."""
    ra = 135.0 + (-35 * t_hours - 0.02 * t_hours ** 2) / 3600 / np.cos(np.radians(22.5))
    dec = 22.5 + (10 * t_hours + 0.01 * t_hours ** 2) / 3600
    return ra, dec


def night(start_hour, other_ra, other_dec):
    rows = []
    for k in range(5):
        t = start_hour + k * 0.3
        ra, dec = curved_path(t)
        rows.append({"ra": ra, "dec": dec, "obsjd": 2459000 + t / 24})
        rows.append({"ra": other_ra - 30 * t / 3600, "dec": other_dec, "obsjd": 2459000 + t / 24})   # an unrelated object
    detections = pd.DataFrame(rows)
    members = pd.DataFrame({"tracklet_id": [0, 1] * 5, "detection_id": range(10)})
    return measure_sky_motion(members, detections)


def test_same_object_three_days_later_is_linked_and_the_other_is_not():
    first = night(0.0, 135.3, 22.8)
    second = night(72.0, 135.9, 22.1)

    pairs = link_tracklets_across_nights(first, second)

    assert list(zip(pairs["first_id"], pairs["second_id"])) == [(0, 0)]
    assert abs(first.loc[0, "v_ra"] + 35) < 1                       # the speed is measured in true arcsec/hour


def test_each_tracklet_gets_at_most_one_partner():
    first = pd.DataFrame({"tracklet_id": [0], "jd": [2459000.0], "ra": [135.0], "dec": [22.5], "v_ra": [-35.0], "v_dec": [10.0]})
    hours = 72
    cos_dec = np.cos(np.radians(22.5))
    exact_ra, exact_dec = 135.0 - 35 * hours / 3600 / cos_dec, 22.5 + 10 * hours / 3600
    second = pd.DataFrame({"tracklet_id": [5, 6], "jd": [2459003.0] * 2, "ra": [exact_ra, exact_ra + 0.01], "dec": [exact_dec] * 2,
                           "v_ra": [-35.5, -34.0], "v_dec": [10.0, 10.5]})
    pairs = link_tracklets_across_nights(first, second)
    assert pairs["second_id"].tolist() == [5]


def test_tracklets_from_different_units_with_the_same_local_id_stay_separate():
    first = pd.DataFrame({"tracklet_id": ["a/point-1"], "jd": [2459000.0], "ra": [135.0], "dec": [22.5], "v_ra": [-35.0], "v_dec": [10.0]})
    cos_dec = np.cos(np.radians(22.5))
    second = pd.DataFrame({"tracklet_id": ["b/point-1", "c/point-1"], "jd": [2459003.0] * 2,
                           "ra": [135.0 - 35 * 72 / 3600 / cos_dec, 140.0], "dec": [22.5 + 10 * 72 / 3600, 25.0],
                           "v_ra": [-35.0, -35.0], "v_dec": [10.0, 10.0]})
    pairs = link_tracklets_across_nights(first, second)
    assert pairs["second_id"].tolist() == ["b/point-1"]
