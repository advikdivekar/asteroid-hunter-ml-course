"""How far each frame's pointing is shifted relative to a reference frame (whole pixels)."""

import numpy as np


def compute_frame_offsets(wcs_list, image_shape):
    """Return an (n_frames, 2) integer array of (dx, dy): where the reference frame's centre
    lands in each frame's pixels, minus the centre.

    wcs_list[0] is the reference. Adding (dx, dy) to a position on the
    reference grid gives the same sky position on frame k's grid.
    Over one CCD quadrant the shift between exposures is a translation to a
    good approximation, so one number per frame is enough; rounding to whole
    pixels leaves at most 0.5 px, small next to the ~2 px PSF.
    """
    height, width = image_shape
    centre = wcs_list[0].pixel_to_world(width / 2, height / 2)
    offsets = []
    for wcs in wcs_list:
        x, y = wcs.world_to_pixel(centre)
        offsets.append((float(x) - width / 2, float(y) - height / 2))
    return np.round(np.array(offsets)).astype(int)
