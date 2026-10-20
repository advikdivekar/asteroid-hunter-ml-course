import numpy as np

from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.models.train_realbogus_cnn import train_realbogus_cnn


def make_stamps(n, rng):
    """Real = a bright blob in the centre of the difference channel; bogus = noise only."""
    stamps = rng.normal(0, 1, size=(n, 3, 31, 31)).astype(np.float32)
    labels = (np.arange(n) % 2).astype(int)
    stamps[labels == 1, 2, 13:18, 13:18] += 5.0
    return stamps, labels


def test_learns_an_easy_pattern():
    rng = np.random.default_rng(0)
    train_stamps, train_labels = make_stamps(400, rng)
    validation_stamps, validation_labels = make_stamps(200, rng)
    settings = {"learning_rate": 0.001, "batch_size": 64, "max_epochs": 5, "patience": 2}

    model, history = train_realbogus_cnn(train_stamps, train_labels, validation_stamps, validation_labels,
                                         settings, device="cpu", seed=0)

    scores = predict_realbogus(model, validation_stamps)
    assert history[-1]["validation_ap"] > 0.95
    assert scores[validation_labels == 1].mean() > scores[validation_labels == 0].mean()
