import numpy as np
from astropy.io import fits

from asthunt.clean.load_exposure import load_exposure


def write_fake_exposure(folder, with_science=True):
    """Write a tiny set of the ZTF products for one exposure (the science image only if with_science)."""
    prefix = folder / "ztf_test_exposure_"
    header = fits.Header({"SATURATE": 1000.0, "OBSJD": 2459000.5, "MAGZP": 26.0, "GAIN": 6.2})

    science = np.full((20, 20), 100.0)
    science[5, 5] = 5000.0                                         # saturated pixel
    difference = np.full((20, 20), 1.0)
    difference[0, 0] = np.nan                                      # unsubtracted pixel
    mask = np.zeros((20, 20), dtype=np.int16)
    mask[10, 10] = 1                                               # bit 0: satellite track
    mask[5, 5] = 1 << 8                                            # bit 8: saturated (the science image is also saturated there)

    if with_science:
        fits.PrimaryHDU(science, header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(difference.astype(np.float32), header=header)]).writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(mask).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(np.ones((5, 5)) * 2.0).writeto(f"{prefix}diffimgpsf.fits")
    return f"{prefix}sciimg.fits"


def test_loads_all_products_and_combines_bad_pixels(tmp_path):
    exposure = load_exposure(write_fake_exposure(tmp_path), bad_mask_bits=[0])

    assert exposure["obsjd"] == 2459000.5
    assert exposure["bad"][5, 5]                 # saturated
    assert exposure["bad"][0, 0]                 # NaN in difference
    assert exposure["bad"][10, 10]               # mask bit 0
    assert not exposure["bad"][15, 15]
    assert np.isclose(exposure["psf"].sum(), 1.0)
    assert np.isclose(exposure["reference"][15, 15], 99.0, atol=0.1)   # science - difference


def test_without_the_science_image_the_exposure_loads_in_light_mode_with_the_same_bad_pixels(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "light").mkdir()
    full = load_exposure(write_fake_exposure(tmp_path / "full"), bad_mask_bits=[0])
    light = load_exposure(write_fake_exposure(tmp_path / "light", with_science=False), bad_mask_bits=[0])

    assert light["science"] is None and light["reference"] is None
    assert light["obsjd"] == 2459000.5 and light["magzp"] == 26.0 and light["gain"] == 6.2
    assert np.array_equal(light["bad"], full["bad"])             # saturated pixel and its 2-pixel halo, NaN, mask bit 0: all identical
    assert light["bad"][5, 7] and not light["bad"][5, 8]         # the halo of the saturated pixel is grown by 2
    assert np.array_equal(light["difference"], full["difference"], equal_nan=True)
