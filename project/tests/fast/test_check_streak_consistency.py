import pandas as pd

from asthunt.fast.check_streak_consistency import check_streak_consistency


def test_motion_along_the_streaks_is_consistent_across_them_is_not():
    # 1,200 "/hr at 1 "/px over 30 s -> 10 px trails
    tracklets = pd.DataFrame({"tracklet_id": [0, 1], "vx": [1200.0, 1200.0], "vy": [0.0, 0.0],
                              "rate_arcsec_per_hour": [1200.0, 1200.0]})
    detections = pd.DataFrame({"angle": [0, 180, 0, 90, 90, 90], "length": [10, 9, 11, 10, 10, 10]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 1, 1, 1], "detection_id": [0, 1, 2, 3, 4, 5]})

    result = check_streak_consistency(tracklets, members, detections, 30, 1.0, 20, (0.5, 2.0))

    assert result["consistent"].tolist() == [True, False]          # tracklet 1's trails are perpendicular to its motion
    assert result.loc[0, "angle_difference"] == 0                  # 0 and 180 degrees are the same trail direction


def test_changing_trail_length_is_not_one_object():
    tracklets = pd.DataFrame({"tracklet_id": [0], "vx": [1200.0], "vy": [0.0], "rate_arcsec_per_hour": [1200.0]})
    detections = pd.DataFrame({"angle": [0, 0, 0], "length": [4, 32, 16]})          # spread 0.66
    members = pd.DataFrame({"tracklet_id": [0, 0, 0], "detection_id": [0, 1, 2]})
    result = check_streak_consistency(tracklets, members, detections, 30, 1.0, 15, (0.1, 10.0), max_length_spread=0.5)
    assert not result.loc[0, "consistent"]
