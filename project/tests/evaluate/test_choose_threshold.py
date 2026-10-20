import numpy as np

from asthunt.evaluate.choose_threshold import choose_threshold


def test_threshold_keeps_requested_fraction_of_real():
    labels = np.array([1] * 10 + [0] * 10)
    scores = np.concatenate([np.linspace(0.1, 1.0, 10), np.zeros(10)])

    threshold = choose_threshold(labels, scores, target_recall=0.8)

    kept = (scores[labels == 1] >= threshold).mean()
    assert kept >= 0.8
    assert threshold > 0.1
