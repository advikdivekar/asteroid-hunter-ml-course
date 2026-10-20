import numpy as np
import pandas as pd

from asthunt.features.add_path_features import ALL_FEATURE_COLUMNS, add_path_features


def test_adds_columns_and_keeps_row_order():
    detections = pd.DataFrame({"x_common": [0.0, 50.0, 10.0, 60.0], "y_common": 0.0, "frame_index": [0, 0, 1, 1]})
    tracklets = pd.DataFrame({"tracklet_id": [5, 2]})
    members = pd.DataFrame({"tracklet_id": [5, 5, 2, 2], "detection_id": [0, 2, 1, 3]})
    stamps = np.zeros((2, 2, 3, 21, 21), dtype=np.float32)
    stamps[1, :, 2, 10, 10] = 9.0                       # only tracklet 2 (second row) has signal

    result = add_path_features(tracklets, members, detections, stamps)

    assert result["tracklet_id"].tolist() == [5, 2]
    assert result.loc[1, "stack_snr"] > result.loc[0, "stack_snr"]
    assert {"static_neighbours_mean", "stack_snr"} <= set(result.columns) and len(ALL_FEATURE_COLUMNS) == 17
