import numpy as np
from astropy.io import fits

from asthunt.clean.read_exposure_header import read_exposure_header


def write_exposure(folder, with_science):
    prefix = folder / "ztf_test_"
    header = fits.Header({"SATURATE": 1000.0, "OBSJD": 2459000.5, "MAGZP": 26.0, "GAIN": 6.2})
    if with_science:
        fits.PrimaryHDU(np.full((20, 30), 100.0), header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(np.ones((20, 30), dtype=np.float32), header=header)]).writeto(f"{prefix}scimrefdiffimg.fits.fz")
    return f"{prefix}sciimg.fits"


def test_the_header_and_size_come_from_the_science_image_or_from_the_difference_image_without_it(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "light").mkdir()
    header_full, shape_full = read_exposure_header(write_exposure(tmp_path / "full", with_science=True))
    header_light, shape_light = read_exposure_header(write_exposure(tmp_path / "light", with_science=False))

    assert shape_full == shape_light == (20, 30)                 # (height, width)
    for key in ["OBSJD", "MAGZP", "GAIN", "SATURATE", "NAXIS1", "NAXIS2"]:     # refine_known_positions reads NAXIS1/NAXIS2 from this header
        assert header_full[key] == header_light[key]
