"""The trial motions to stack along: every direction, speeds up to a limit."""

import numpy as np


def make_velocity_grid(max_rate_arcsec_per_hour, step_pixels_per_hour, pixel_scale):
    """Return an (n, 2) array of (vx, vy) in pixels/hour, including (0, 0).

    The step must be small enough that a velocity error of half a step moves
    the object less than ~1 pixel over the stack's time span:
        step / 2 * span_hours <= ~1 px   ->   for a 0.5 h stack, step <= ~4 px/h
    """
    max_pixels = max_rate_arcsec_per_hour / pixel_scale
    values = np.arange(-max_pixels, max_pixels + 1e-9, step_pixels_per_hour)
    vx, vy = np.meshgrid(values, values)
    keep = np.hypot(vx, vy) <= max_pixels
    return np.column_stack([vx[keep], vy[keep]])
