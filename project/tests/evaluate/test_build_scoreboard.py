import numpy as np
import pandas as pd

from asthunt.evaluate.build_scoreboard import build_scoreboard


def fake_night_result():
    """A tiny hand-made night: 2 frames, 2 planted fakes (one found), 1 known asteroid (found), 1 unexplained alarm."""
    detections = pd.DataFrame({
        "night": "n1", "exposure": ["e0", "e0", "e0", "e1", "e1", "e1"],
        "label": [1, 1, 0, 1, 1, 0],
        "score": [0.9, 0.2, 0.8, 0.95, 0.3, 0.1],
    })
    injections = pd.DataFrame({"injection_id": [0, 0, 1, 1], "mag": [18.0, 18.0, 21.5, 21.5]})
    tracklets = pd.DataFrame({
        "tracklet_id": [0, 1, 2], "tracklet_score": [0.99, 0.97, 0.98], "label": [1, 0, 0],
        "injection_id": [0, -1, -1], "known_name": ["", "Ceres", ""], "contains_fake": [True, False, False],
    })
    known = pd.DataFrame({"name": ["Ceres", "Ceres", "Faint"], "v_mag": [18.0, 18.0, 23.0], "exposure": ["e0", "e1", "e0"]})
    return {"detections": detections, "injections": injections, "tracklets": tracklets, "known": known}


def test_headline_numbers():
    row = build_scoreboard([fake_night_result()], {"realbogus": 0.5, "tracklet": 0.95},
                           min_detections=2, known_max_v_mag=20.5)

    assert row["frames"] == 2
    assert row["detection_recall"] == 0.5                  # 2 of 4 real detections kept
    assert row["bogus_kept_per_frame"] == 0.5              # 1 bogus kept over 2 frames
    assert row["known_recovered"] == "1/1"                 # Ceres yes; "Faint" is too faint to count
    assert row["unexplained_per_night"] == 1               # tracklet 2
    assert 18.0 < row["mag_at_50_completeness"] < 21.5     # bright fake found, faint one not
