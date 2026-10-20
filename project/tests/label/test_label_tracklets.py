import pandas as pd

from asthunt.label.label_tracklets import label_tracklets


def test_only_single_object_tracklets_are_real():
    detections = pd.DataFrame({
        "injection_id":   [5, 5, 5, 5, 9, -1, -1, -1],
        "known_asteroid": ["", "", "", "", "", "", "", ""],
    })
    tracklets = pd.DataFrame({"tracklet_id": [0, 1, 2]})
    members = pd.DataFrame({
        "tracklet_id":  [0, 0, 0, 0, 1, 1, 1, 2, 2],
        "detection_id": [0, 1, 2, 3, 4, 5, 6, 6, 7],
    })
    labelled = label_tracklets(tracklets, members, detections)

    assert labelled["label"].tolist() == [1, 0, 0]          # pure fake / fake + noise / only noise
    assert labelled["injection_id"].tolist() == [5, -1, -1]
