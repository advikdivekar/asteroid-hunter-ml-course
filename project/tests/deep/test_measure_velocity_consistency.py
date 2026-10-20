import pandas as pd

from asthunt.deep.measure_velocity_consistency import measure_velocity_consistency


def test_consistent_object_scores_low_chance_alignment_high():
    tracklets = pd.DataFrame({"tracklet_id": [0, 1], "vx": [30.0, 30.0], "vy": [-10.0, -10.0]})
    stack_detections = pd.DataFrame({"vx": [31.0, 29.0, 30.0, -50.0, 80.0, 0.0], "vy": [-10.0, -9.0, -11.0, 40.0, 5.0, -60.0]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 1, 1, 1], "detection_id": [0, 1, 2, 3, 4, 5]})

    result = measure_velocity_consistency(tracklets, members, stack_detections).set_index("tracklet_id")

    assert result.loc[0, "velocity_mismatch"] < 2
    assert result.loc[1, "velocity_mismatch"] > 30
