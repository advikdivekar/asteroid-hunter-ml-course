from asthunt.label.match_detections_to_positions import match_detections_to_positions


def test_matches_within_radius_only():
    matches = match_detections_to_positions(
        detection_x=[10.0, 50.0, 90.0], detection_y=[10.0, 50.0, 90.0],
        position_x=[11.0, 95.0], position_y=[10.0, 95.0], max_distance=2.0)
    assert matches.tolist() == [0, -1, -1]


def test_no_positions_means_no_matches():
    assert match_detections_to_positions([1.0], [1.0], [], [], 2.0).tolist() == [-1]
