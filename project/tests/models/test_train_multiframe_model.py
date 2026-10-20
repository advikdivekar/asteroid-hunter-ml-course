import numpy as np

from asthunt.models.predict_multiframe import predict_multiframe
from asthunt.models.train_multiframe_model import train_multiframe_model


def make_tracklets(n, rng):
    """Real = a faint blob centred in every frame's difference stamp (visible mostly in the stack); bogus = noise."""
    stamps = rng.normal(0, 1, size=(n, 5, 3, 21, 21)).astype(np.float32)
    labels = (np.arange(n) % 3 == 0).astype(int)
    stamps[labels == 1, :, 2, 9:12, 9:12] += 1.0          # only 1 sigma per pixel per frame
    features = rng.normal(0, 1, size=(n, 4)).astype(np.float32)
    return {"stamps": stamps, "features": features, "labels": labels}


def test_learns_a_faint_signal_shared_across_frames():
    rng = np.random.default_rng(0)
    train, validation = make_tracklets(600, rng), make_tracklets(300, rng)
    settings = {"learning_rate": 0.002, "batch_size": 64, "max_epochs": 6, "patience": 3}

    model, history = train_multiframe_model(train, validation, settings, device="cpu", seed=0)

    scores = predict_multiframe(model, validation["stamps"], validation["features"])
    assert max(h["validation_ap"] for h in history) > 0.9
    assert scores[validation["labels"] == 1].mean() > scores[validation["labels"] == 0].mean()
