import numpy as np

from asthunt.dataset.make_unit_rng import make_unit_rng


def test_same_unit_same_numbers_different_unit_different_numbers():
    a = {"field": 618, "ccd": 3, "quadrant": 1, "night": "2020-09-19"}
    b = {"field": 618, "ccd": 12, "quadrant": 2, "night": "2020-09-19"}
    assert make_unit_rng(42, a).random() == make_unit_rng(42, a).random()
    assert make_unit_rng(42, a).random() != make_unit_rng(42, b).random()


def test_original_quadrant_keeps_round_one_seed():
    unit = {"field": 618, "ccd": 3, "quadrant": 4, "night": "2020-09-19"}
    assert make_unit_rng(42, unit).random() == np.random.default_rng([42, 20200919]).random()
