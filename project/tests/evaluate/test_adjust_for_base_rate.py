import numpy as np

from asthunt.evaluate.adjust_for_base_rate import adjust_for_base_rate


def test_worked_example_from_docstring():
    adjusted = adjust_for_base_rate([0.9], training_real_fraction=1 / 6, true_real_fraction=1 / 1000)
    assert np.isclose(adjusted[0], 0.043, atol=0.001)


def test_same_base_rate_changes_nothing():
    assert np.allclose(adjust_for_base_rate([0.2, 0.7], 0.3, 0.3), [0.2, 0.7])
