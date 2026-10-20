import numpy as np
from astropy.io import fits

from asthunt.clean.read_fits_image import read_fits_image


def test_reads_plain_fits(tmp_path):
    path = tmp_path / "plain.fits"
    fits.PrimaryHDU(data=np.ones((4, 5), dtype=np.int16), header=fits.Header({"MAGZP": 26.0})).writeto(path)

    image, header = read_fits_image(path)

    assert image.shape == (4, 5)
    assert image.dtype == np.float64
    assert header["MAGZP"] == 26.0


def test_reads_compressed_fits(tmp_path):
    path = tmp_path / "compressed.fits.fz"
    hdus = fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=np.full((8, 8), 3.0, dtype=np.float32))])
    hdus.writeto(path)

    image, _ = read_fits_image(path)

    assert image.shape == (8, 8)
    assert np.allclose(image, 3.0, atol=0.01)
