import numpy as np

from asthunt.simulate.render_trailed_source import render_trailed_source


def gaussian_psf(size=11, sigma=1.0):
    yy, xx = np.indices((size, size)) - size // 2
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def test_flux_is_exact_and_the_trail_points_the_right_way():
    stamp = render_trailed_source(gaussian_psf(), total_flux=1000.0, length_pixels=12, angle_degrees=0)
    assert np.isclose(stamp.sum(), 1000.0)
    rows, columns = np.nonzero(stamp > stamp.max() * 0.5)
    assert np.ptp(columns) > 10 and np.ptp(rows) <= 2          # long in x (angle 0), thin in y


def test_a_too_small_requested_size_is_enlarged():
    stamp = render_trailed_source(gaussian_psf(), 100.0, length_pixels=32, angle_degrees=30, size=21)
    assert stamp.shape[0] >= 32 + 11 and np.isclose(stamp.sum(), 100.0)
