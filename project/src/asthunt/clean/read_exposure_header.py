"""The header and image size of one exposure, without reading its pixels."""

from pathlib import Path

from astropy.io import fits


def read_exposure_header(science_path):
    """Return (header, (height, width)) of an exposure.

    Read from the science image when it is on disk. In light mode (the science image was not downloaded, see
    load_exposure) the difference image's own header is used: it has the same time, zero point, gain, saturation
    level and the same WCS (identical to 0.0 arcsec, measured). Only the header is read, never the pixels.
    """
    science_path = Path(science_path)
    if science_path.exists():
        header = fits.getheader(science_path)
        return header, (header["NAXIS2"], header["NAXIS1"])
    with fits.open(str(science_path).replace("sciimg.fits", "scimrefdiffimg.fits.fz")) as hdul:
        hdu = next(h for h in hdul if h.header.get("NAXIS", 0) == 2 or h.header.get("ZNAXIS", 0) == 2)
        return hdu.header.copy(), (hdu.header.get("NAXIS2", hdu.header.get("ZNAXIS2")), hdu.header.get("NAXIS1", hdu.header.get("ZNAXIS1")))
