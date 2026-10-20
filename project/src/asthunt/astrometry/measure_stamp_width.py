"""How wide is the source in a stamp? A point source has the width of the PSF; a comet or a blend is wider."""

import numpy as np
from scipy.optimize import curve_fit


def measure_stamp_width(stamp, pixel_scale=1.012):
    """Return (fwhm_x, fwhm_y, error) in arcsec from a 2-D Gaussian fit centred near the middle of the stamp.

    error is the larger of the two fitted widths' standard errors. Raises RuntimeError when the fit does not converge
    (no source there).
    """
    size = stamp.shape[0]
    y, x = np.indices(stamp.shape)
    centre = (size - 1) / 2

    def gaussian(coordinates, amplitude, x0, y0, sigma_x, sigma_y, offset):
        xx, yy = coordinates
        return (offset + amplitude * np.exp(-((xx - x0) ** 2 / (2 * sigma_x ** 2) + (yy - y0) ** 2 / (2 * sigma_y ** 2)))).ravel()

    good = np.isfinite(stamp)
    start = [np.nanmax(stamp) - np.nanmedian(stamp), centre, centre, 1.2, 1.2, np.nanmedian(stamp)]
    bounds = ([0, centre - 3, centre - 3, 0.4, 0.4, -np.inf], [np.inf, centre + 3, centre + 3, 6, 6, np.inf])
    popt, pcov = curve_fit(gaussian, (x[good], y[good]), stamp[good], p0=start, bounds=bounds)
    to_fwhm = 2 * np.sqrt(2 * np.log(2)) * pixel_scale
    errors = np.sqrt(np.diag(pcov))
    return popt[3] * to_fwhm, popt[4] * to_fwhm, max(errors[3], errors[4]) * to_fwhm
