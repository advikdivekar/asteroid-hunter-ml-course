"""The trail shapes to search for: the PSF smeared along lines of several lengths and angles."""

import numpy as np

from asthunt.simulate.render_trailed_source import render_trailed_source


def build_streak_kernels(psf_model, lengths_pixels, angle_step_degrees):
    """Return a list of dicts {length, angle, kernel}; each kernel sums to 1.

    Angles cover 0..180 degrees only: a trail at 30 degrees looks the same as
    one at 210 (we can't tell the direction of motion from a single image).
    """
    kernels = []
    for length in lengths_pixels:
        for angle in np.arange(0, 180, angle_step_degrees):
            kernel = render_trailed_source(psf_model, 1.0, length, angle)
            kernels.append({"length": float(length), "angle": float(angle), "kernel": kernel / kernel.sum()})
    return kernels
