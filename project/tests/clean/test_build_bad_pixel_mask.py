import numpy as np

from asthunt.clean.build_bad_pixel_mask import build_bad_pixel_mask


def test_nan_and_saturated_pixels_are_bad():
    image = np.full((10, 10), 100.0)
    image[0, 0] = np.nan
    image[5, 5] = 60000.0

    bad = build_bad_pixel_mask(image, saturation_level=50000, grow_saturated_by=1)

    assert bad[0, 0]
    assert bad[5, 5]
    assert bad[4, 5] and bad[6, 5] and bad[5, 4] and bad[5, 6]   # grown border
    assert not bad[8, 8]


def test_ztf_mask_bits():
    image = np.zeros((3, 3))
    ztf_mask = np.array([[0, 1, 2], [4, 16, 17], [0, 0, 0]])

    bad = build_bad_pixel_mask(image, saturation_level=1e9, ztf_mask=ztf_mask, bad_bits=(4,))

    # only values with bit 4 (= 16) set are bad
    assert bad.tolist() == [[False, False, False], [False, True, True], [False, False, False]]
