import numpy as np
import pandas as pd

from asthunt.label.label_detections import label_detections
from asthunt.simulate.plan_injections import plan_injections


def test_labels_fakes_known_and_bogus():
    detections = pd.DataFrame({"x": [10.0, 50.0, 90.0], "y": [10.0, 50.0, 90.0]})
    injections = pd.DataFrame({"injection_id": [7], "x": [10.5], "y": [10.0]})
    known = pd.DataFrame({"name": ["(433) Eros"], "x": [90.0], "y": [91.0]})

    labelled = label_detections(detections, injections, known, match_radius=2.0)

    assert labelled["injection_id"].tolist() == [7, -1, -1]
    assert labelled["known_asteroid"].tolist() == ["", "", "(433) Eros"]
    assert labelled["label"].tolist() == [1, 0, 1]


def test_known_asteroids_can_use_a_wider_radius():
    detections = pd.DataFrame({"x": [100.0], "y": [100.0]})
    no_fakes = pd.DataFrame({"injection_id": [], "x": [], "y": []})
    known = pd.DataFrame({"name": ["Kirillavrov"], "x": [97.6], "y": [101.5]})     # 2.8 px away

    assert label_detections(detections, no_fakes, known, match_radius=2.0)["label"].tolist() == [0]
    assert label_detections(detections, no_fakes, known, 2.0, known_match_radius=4.0)["label"].tolist() == [1]


def test_works_with_no_fakes_and_no_known_asteroids():
    detections = pd.DataFrame({"x": [10.0, 50.0], "y": [10.0, 50.0]})
    # Exactly what the search run produces: plan_injections with zero fakes
    no_fakes = plan_injections(0, (100, 100), [2459000.0], (18, 20), (10, 20), 1.0, 10, np.random.default_rng(0))
    no_known = pd.DataFrame(columns=["name", "x", "y"])

    labelled = label_detections(detections, no_fakes, no_known, match_radius=2.0)

    assert labelled["label"].tolist() == [0, 0]
    assert labelled["injection_id"].tolist() == [-1, -1]
