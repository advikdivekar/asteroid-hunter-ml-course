"""Load every product of one ZTF exposure into a single dict."""

from pathlib import Path

import numpy as np

from asthunt.clean.build_bad_pixel_mask import build_bad_pixel_mask
from asthunt.clean.read_fits_image import read_fits_image

SATURATED_BIT = 8          # BIT08 of ZTF's mask files: "SATURATED" (listed in the mask header)


def load_exposure(science_path, bad_mask_bits):
    """Return a dict with the images and numbers we need for one exposure.

    Keys:
      science, difference, reference : 2-D images on the SAME pixel grid (science and reference are None in light mode)
      bad                            : True where a pixel must not be used
      psf                            : PSF model of the difference image (sums to 1)
      header                         : the science image header (the difference image's own header in light mode)
      obsjd, magzp, gain             : time and calibration numbers

    The reference is computed as science - difference. ZTF already
    resampled and blurred the reference to match the science image before
    subtracting, so this gives us the reference on the science pixel grid
    without having to re-project it ourselves.

    LIGHT MODE: if the science image is not on disk, the exposure is loaded from the other three products. The science image
    is 59% of what an exposure costs to download (38 of 64 MB), and the deep, fast and counterpart searches never use it: the
    difference image's own header holds the time, zero point, gain, saturation level and a WCS identical to the science image's,
    and the mask's SATURATED bit stands in for "science >= saturation level". science and reference are then None, so the
    point search (it cuts science/reference/difference stamps) and planting fakes need the full exposure.
    """
    science_path = Path(science_path)
    prefix = str(science_path).replace("sciimg.fits", "")

    difference, difference_header = read_fits_image(prefix + "scimrefdiffimg.fits.fz")
    mask_bits, _ = read_fits_image(prefix + "mskimg.fits")
    psf, _ = read_fits_image(prefix + "diffimgpsf.fits")

    if science_path.exists():
        science, header = read_fits_image(science_path)
        bad = build_bad_pixel_mask(science, header["SATURATE"], ztf_mask=mask_bits, bad_bits=bad_mask_bits)
        reference = science - difference
    else:
        science, reference, header = None, None, difference_header
        saturated = (mask_bits.astype(np.int64) & (1 << SATURATED_BIT)) != 0
        stand_in = np.where(saturated, header["SATURATE"], 0.0)               # lets build_bad_pixel_mask grow the saturated areas as usual
        bad = build_bad_pixel_mask(stand_in, header["SATURATE"], ztf_mask=mask_bits, bad_bits=bad_mask_bits)
    bad = bad | np.isnan(difference)

    return {
        "name": science_path.name.replace("_sciimg.fits", ""),
        "science": science,
        "difference": difference,
        "reference": reference,
        "bad": bad,
        "psf": psf / psf.sum(),
        "header": header,
        "obsjd": header["OBSJD"],
        "magzp": header["MAGZP"],
        "gain": header["GAIN"],
    }
