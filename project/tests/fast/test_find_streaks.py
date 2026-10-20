import numpy as np

from asthunt.fast.build_streak_kernels import build_streak_kernels
from asthunt.fast.find_streaks import find_streaks
from asthunt.simulate.render_point_source import render_point_source
from asthunt.simulate.render_trailed_source import render_trailed_source


def gaussian_psf(size=11, sigma=1.0):
    yy, xx = np.indices((size, size)) - size // 2
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def test_finds_the_trail_with_its_length_and_angle_and_ignores_a_star():
    rng = np.random.default_rng(0)
    psf = gaussian_psf()
    image = rng.normal(0, 5, size=(200, 200))
    trail = render_trailed_source(psf, 3000.0, 16, 60)
    half = trail.shape[0] // 2
    image[100 - half:100 + half + 1, 70 - half:70 + half + 1] += trail
    star = render_point_source(psf, 3000.0, 0.0, 0.0)
    image[40 - 5:40 + 6, 150 - 5:150 + 6] += star
    bad = np.zeros(image.shape, dtype=bool)

    streaks = find_streaks(image, bad, psf, build_streak_kernels(psf, [4, 8, 16, 32], 15), threshold=8)

    best = streaks.loc[streaks["snr"].idxmax()]
    assert abs(best["x"] - 70) <= 2 and abs(best["y"] - 100) <= 2
    assert best["length"] == 16 and abs(best["angle"] - 60) <= 15
    assert (np.hypot(streaks["x"] - 150, streaks["y"] - 40) > 5).all()      # the round star is not a streak


def test_a_trail_lying_on_the_image_border_is_not_reported():
    rng = np.random.default_rng(1)
    psf = gaussian_psf()
    image = rng.normal(0, 5, size=(200, 200))
    image[0:3, :] += 40.0                                    # a bright band along the bottom edge
    bad = np.zeros(image.shape, dtype=bool)

    streaks = find_streaks(image, bad, psf, build_streak_kernels(psf, [4, 8, 16, 32], 15), threshold=8)

    assert (streaks["y"] > 10).all()
