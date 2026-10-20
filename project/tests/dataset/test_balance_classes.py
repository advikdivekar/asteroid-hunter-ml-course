import numpy as np

from asthunt.dataset.balance_classes import balance_classes


def test_keeps_all_real_and_limits_bogus():
    labels = np.array([1, 1] + [0] * 100)
    chosen = balance_classes(labels, bogus_per_real=3, rng=np.random.default_rng(0))

    assert len(chosen) == 2 + 6
    assert set([0, 1]) <= set(chosen)
    assert (labels[chosen] == 0).sum() == 6
