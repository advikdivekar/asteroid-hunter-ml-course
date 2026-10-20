import numpy as np

from asthunt.simulate.inject_point_source import inject_point_source


def test_adds_the_right_flux_to_both_images():
    psf = np.zeros((5, 5)); psf[2, 2] = 1.0                           # all flux in one pixel
    science = np.zeros((50, 50)); difference = np.zeros((50, 50))

    inject_point_source(science, difference, psf, magnitude=20.0, magzp=26.0, gain=1000.0,
                        x=20, y=30, rng=np.random.default_rng(0))

    expected = 10 ** ((26.0 - 20.0) / 2.5)                            # 251.2 counts
    assert np.isclose(science.sum(), expected, rtol=0.01)
    assert np.allclose(science, difference)                           # same stamp in both
    assert science[30, 20] == science.max()                           # row = y, column = x


def test_source_at_edge_is_skipped():
    psf = np.ones((5, 5)) / 25
    science = np.zeros((10, 10)); difference = np.zeros((10, 10))
    inject_point_source(science, difference, psf, 20.0, 26.0, 6.2, x=0.5, y=0.5, rng=np.random.default_rng(0))
    assert science.sum() == 0
