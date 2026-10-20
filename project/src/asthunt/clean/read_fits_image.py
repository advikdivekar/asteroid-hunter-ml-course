"""Read the pixel array and header from a FITS file (plain or .fz compressed)."""

import numpy as np
from astropy.io import fits


def read_fits_image(path):
    """Return (image, header) for the first extension that holds pixel data.

    - Plain ZTF images keep their pixels in extension 0.
    - Compressed files (.fits.fz) keep an empty extension 0 and the
      pixels in extension 1, so we search for the first one with data.
    - Pixels are converted to float64 so later maths (subtracting the
      background, adding fake asteroids) never overflows or rounds.
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                image = np.asarray(hdu.data, dtype=np.float64)
                header = hdu.header.copy()
                return image, header

    raise ValueError(f"No image data found in {path}")
