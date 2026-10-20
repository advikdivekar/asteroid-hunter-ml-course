import numpy as np

from asthunt.evaluate.calibrate_probabilities import calibrate_probabilities


def test_overconfident_scores_are_pulled_down():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 20000)
    labels = rng.uniform(0, 1, 20000) < scores / 2      # truly real only half as often as the score claims

    calibrator = calibrate_probabilities(scores, labels)

    assert abs(calibrator.predict([0.8])[0] - 0.4) < 0.05
