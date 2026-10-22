import subprocess

import numpy as np
import pandas as pd

from asthunt.models.load_model_bundle import load_model_bundle
from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.models.save_model_bundle import save_model_bundle
from asthunt.models.train_realbogus_trees import train_realbogus_trees
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier


def make_data(n, rng):
    """Real = a star-shaped blob in the centre of science and difference; bogus = noise only."""
    yy, xx = np.mgrid[:31, :31] - 15
    star = 8 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.3 ** 2))
    stamps = rng.normal(0, 1, size=(n, 3, 31, 31)).astype(np.float32)
    labels = (np.arange(n) % 2).astype(int)
    stamps[labels == 1, 0] += star
    stamps[labels == 1, 2] += star
    detections = pd.DataFrame({c: rng.uniform(1, 2, n) for c in ["snr", "peak", "npix", "a", "b", "flux"]})
    return stamps, detections, labels


def test_learns_an_easy_pattern():
    rng = np.random.default_rng(0)
    stamps, detections, labels = make_data(300, rng)
    model = train_realbogus_trees(stamps, detections, labels, seed=0)
    test_stamps, test_detections, test_labels = make_data(100, rng)
    scores = predict_realbogus(model, test_stamps, detections=test_detections)
    assert ((scores > 0.5) == test_labels).mean() > 0.95


def test_bundle_round_trip_keeps_the_same_scores(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    rng = np.random.default_rng(1)
    stamps, detections, labels = make_data(200, rng)
    model = train_realbogus_trees(stamps, detections, labels, seed=0)
    tracklet_model = train_tracklet_classifier(rng.uniform(size=(50, 3)), np.arange(50) % 2)

    save_model_bundle(models_dir, "v9", model, tracklet_model, {"realbogus": 0.01, "tracklet": 0.5}, {})
    bundle = load_model_bundle(models_dir, "v9")

    assert isinstance(bundle["realbogus_model"], dict)
    assert np.allclose(predict_realbogus(bundle["realbogus_model"], stamps, detections=detections),
                       predict_realbogus(model, stamps, detections=detections), atol=1e-6)
