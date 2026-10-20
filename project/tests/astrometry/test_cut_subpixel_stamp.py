import numpy as np

from asthunt.astrometry.cut_subpixel_stamp import cut_subpixel_stamp
from asthunt.astrometry.measure_stamp_width import measure_stamp_width


def blob(x0, y0, sigma, size=80, amplitude=50.0):
    y, x = np.indices((size, size))
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def test_the_source_lands_on_the_centre_even_when_it_sits_between_pixels():
    image = blob(40.4, 33.7, 1.5)
    stamp = cut_subpixel_stamp(image, 40.4, 33.7, half_size=8)
    assert stamp.shape == (17, 17)
    assert np.unravel_index(np.nanargmax(stamp), stamp.shape) == (8, 8)
    assert abs(stamp[8, 8] - 50.0) < 0.5


def test_pixels_beyond_the_image_are_nan_and_bad_pixels_do_not_spread():
    image = blob(5, 5, 1.5)
    image[10, 10] = np.nan
    stamp = cut_subpixel_stamp(image, 5, 5, half_size=8)
    assert np.isnan(stamp[:3, :]).all() and np.isnan(stamp[:, :3]).all()      # stamp rows/columns 0-2 are image coordinates -3..-1
    assert np.isfinite(stamp[3:, 3:]).all()                                   # everything on the image, including the bad pixel, is finite


def test_a_wider_source_measures_wider():
    narrow = cut_subpixel_stamp(blob(40, 40, 1.2), 40, 40, 10)
    wide = cut_subpixel_stamp(blob(40, 40, 2.4), 40, 40, 10)
    narrow_fwhm, _, _ = measure_stamp_width(narrow, pixel_scale=1.0)
    wide_fwhm, _, _ = measure_stamp_width(wide, pixel_scale=1.0)
    assert abs(narrow_fwhm - 1.2 * 2.355) < 0.1
    assert abs(wide_fwhm - 2.4 * 2.355) < 0.2
