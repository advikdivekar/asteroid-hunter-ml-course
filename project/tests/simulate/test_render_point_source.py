import numpy as np

from asthunt.simulate.render_point_source import render_point_source


def gaussian_psf(size=25, sigma=1.0):
    yy, xx = np.indices((size, size)) - size // 2
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def test_total_flux_is_exact():
    stamp = render_point_source(gaussian_psf(), total_flux=1234.0, x_offset=0.3, y_offset=-0.2)
    assert np.isclose(stamp.sum(), 1234.0)
    assert (stamp >= 0).all()


def test_offset_moves_the_centre():
    stamp = render_point_source(gaussian_psf(), total_flux=1.0, x_offset=0.4, y_offset=0.0)
    yy, xx = np.indices(stamp.shape)
    centre_x = (stamp * xx).sum()
    assert np.isclose(centre_x, 12.4, atol=0.05)    # 12 is the middle pixel of 25
