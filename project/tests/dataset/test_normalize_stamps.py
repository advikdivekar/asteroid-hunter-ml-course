import numpy as np

from asthunt.dataset.normalize_stamps import normalize_stamps


def test_sky_becomes_zero_and_noise_becomes_one():
    rng = np.random.default_rng(0)
    stamps = 500 + rng.normal(0, 10, size=(4, 3, 31, 31))     # sky 500, noise 10

    normalized = normalize_stamps(stamps)

    assert normalized.dtype == np.float32
    assert abs(np.median(normalized)) < 0.05
    assert abs(normalized.std() - 1) < 0.05


def test_empty_channel_stays_zero():
    stamps = np.zeros((1, 3, 5, 5))
    assert (normalize_stamps(stamps) == 0).all()
