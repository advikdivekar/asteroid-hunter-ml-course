import pandas as pd

from asthunt.crossmatch.crossmatch_known_asteroids import crossmatch_known_asteroids


def test_tracklet_following_a_known_object_is_named():
    detections = pd.DataFrame({
        "exposure": ["e0", "e1", "e2", "e0", "e1", "e2"],
        "x": [100.0, 110.0, 120.0, 500.0, 510.0, 520.0],
        "y": [100.0, 100.0, 100.0, 500.0, 500.0, 500.0],
    })
    known = {e: pd.DataFrame({"name": ["(1) Ceres"], "x": [100.0 + 10 * i + 1], "y": [101.0]})
             for i, e in enumerate(["e0", "e1", "e2"])}
    tracklets = pd.DataFrame({"tracklet_id": [0, 1]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 1, 1, 1], "detection_id": [0, 1, 2, 3, 4, 5]})

    result = crossmatch_known_asteroids(tracklets, members, detections, known)

    assert result["known_name"].tolist() == ["(1) Ceres", ""]
    assert result["known_matches"].tolist() == [3, 0]
