import numpy as np

from asthunt.models.train_tracklet_classifier import train_tracklet_classifier


def test_learns_a_two_feature_rule():
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 1, size=(500, 2))
    labels = ((features[:, 0] > 0.5) & (features[:, 1] < 0.5)).astype(int)

    model = train_tracklet_classifier(features, labels)

    assert model.predict_proba([[0.9, 0.1]])[0, 1] > 0.9
    assert model.predict_proba([[0.1, 0.9]])[0, 1] < 0.1
