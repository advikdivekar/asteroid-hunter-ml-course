"""Cut a small image centred on a position that is not a whole pixel."""

import numpy as np
from scipy.ndimage import map_coordinates


def cut_subpixel_stamp(image, x, y, half_size=10):
    """Return a (2*half_size+1)^2 stamp whose centre pixel is exactly (x, y) in `image` pixels.

    Cubic-spline interpolation, so a source 0.4 pixels off-centre is shifted onto the centre instead of
    being left blurred across two pixels. Pixels outside the image are NaN. Used to stack a moving
    object's frames on the object itself.
    """
    offsets = np.arange(-half_size, half_size + 1)
    xs, ys = np.meshgrid(x + offsets, y + offsets)
    inside = (xs >= 0) & (xs <= image.shape[1] - 1) & (ys >= 0) & (ys <= image.shape[0] - 1)
    filled = np.where(np.isfinite(image), image, 0.0)          # NaNs (bad pixels) must not spread through the spline
    stamp = map_coordinates(filled, [ys, xs], order=3, mode="nearest")
    stamp[~inside] = np.nan
    return stamp
