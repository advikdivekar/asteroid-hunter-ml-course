import pandas as pd

from asthunt.fast.flag_column_aligned import flag_column_aligned


def test_only_a_track_that_moves_and_stays_along_a_column_is_flagged():
    # tracklet 0: moves along the columns (vy only) and stays on x = 100; 1: moves along the rows; 2: moves along the columns but across many x
    tracklets = pd.DataFrame({"tracklet_id": [0, 1, 2], "vx": [0.0, 1000.0, 50.0], "vy": [1000.0, 0.0, 1000.0]})
    detections = pd.DataFrame({"x_common": [100, 100.5, 99.5, 10, 60, 110, 100, 140, 180]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 1, 1, 1, 2, 2, 2], "detection_id": range(9)})

    result = flag_column_aligned(tracklets, members, detections, max_angle_from_column=10, max_column_spread=3)

    assert result["column_aligned"].tolist() == [True, False, False]
    assert result.loc[0, "column_angle_difference"] == 0
    assert 2.8 < result.loc[2, "column_angle_difference"] < 2.9          # 50 against 1000: 2.86 degrees off the column
