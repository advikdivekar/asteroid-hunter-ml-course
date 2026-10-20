"""Add one fake trailed (fast-moving) source into the science and difference images."""

import numpy as np

from asthunt.simulate.render_trailed_source import render_trailed_source


def inject_trailed_source(science, difference, psf_model, magnitude, magzp, gain, x, y, length_pixels, angle_degrees, rng):
    """Add a trail centred at (x, y). Changes the arrays IN PLACE (pass copies to keep originals).

    Same rules as inject_point_source: magnitude -> counts with the zeropoint,
    Poisson noise on the source's own photons, the same stamp added to both
    images, and stamps that would fall off the edge are skipped.
    """
    total_flux = 10 ** ((magzp - magnitude) / 2.5)
    stamp = render_trailed_source(psf_model, total_flux, length_pixels, angle_degrees)
    stamp = rng.poisson(np.clip(stamp, 0, None) * gain) / gain

    half = stamp.shape[0] // 2
    column, row = int(round(x)), int(round(y))
    rows = slice(row - half, row + half + 1)
    columns = slice(column - half, column + half + 1)
    if rows.start < 0 or columns.start < 0 or rows.stop > science.shape[0] or columns.stop > science.shape[1]:
        return
    science[rows, columns] += stamp
    difference[rows, columns] += stamp
