"""Draw a point source (star or asteroid) using the image's own PSF model."""

import numpy as np
from scipy.ndimage import shift


def render_point_source(psf_model, total_flux, x_offset, y_offset):
    """Return a small image of a point source with the given total flux.

    psf_model          : 2-D PSF stamp from ZTF (its pixels sum to 1)
    total_flux         : total counts of the source
    x_offset, y_offset : sub-pixel shift (between -0.5 and 0.5) of the
                         source centre from the stamp's central pixel

    Real sources rarely sit exactly on a pixel centre, so we slide the PSF
    by the fractional offset. Interpolation can create tiny negative values
    at the edges; we clip them and renormalise so the total flux is exact.
    """
    shifted = shift(psf_model, (y_offset, x_offset), order=3, mode="constant")
    shifted = np.clip(shifted, 0, None)
    return total_flux * shifted / shifted.sum()
