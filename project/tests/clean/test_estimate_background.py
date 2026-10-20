import numpy as np

from asthunt.clean.estimate_background import estimate_background


def test_recovers_flat_sky_and_noise():
    rng = np.random.default_rng(0)
    image = 500 + rng.normal(0, 10, size=(256, 256))

    background, noise = estimate_background(image)

    assert abs(np.median(background) - 500) < 1
    assert abs(np.median(noise) - 10) < 1
