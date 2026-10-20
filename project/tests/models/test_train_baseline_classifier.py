import numpy as np

from asthunt.models.train_baseline_classifier import train_baseline_classifier


def test_learns_a_simple_threshold():
    # One feature: values below 0 are class 0, values above 0 are class 1.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y = (x[:, 0] > 0).astype(int)

    model = train_baseline_classifier(x, y)

    assert model.predict([[-2.0]])[0] == 0
    assert model.predict([[2.0]])[0] == 1
    assert model.score(x, y) > 0.95
