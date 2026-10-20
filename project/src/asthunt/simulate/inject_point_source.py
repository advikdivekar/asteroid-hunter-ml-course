"""Add one fake point source into the science and difference images."""

import numpy as np

from asthunt.simulate.render_point_source import render_point_source


def inject_point_source(science, difference, psf_model, magnitude, magzp, gain, x, y, rng):
    """Add a fake source of the given magnitude at pixel (x, y). Changes the arrays IN PLACE.

    Pass copies of the images if you need to keep the originals.

    The same noisy stamp is added to BOTH images. That keeps the reference
    (science - difference) unchanged, exactly as for a real new object:
    it appears tonight but is absent from the reference.

    Photon noise: a source giving N electrons fluctuates by sqrt(N), so we
    convert counts to electrons (x gain), draw Poisson random numbers, and
    convert back (/ gain).
    """
    total_flux = 10 ** ((magzp - magnitude) / 2.5)          # magnitude -> counts

    column, row = int(round(x)), int(round(y))
    stamp = render_point_source(psf_model, total_flux, x - column, y - row)
    stamp = rng.poisson(stamp * gain) / gain

    half_height, half_width = stamp.shape[0] // 2, stamp.shape[1] // 2
    rows = slice(row - half_height, row + half_height + 1)
    columns = slice(column - half_width, column + half_width + 1)

    # Skip sources whose stamp would fall off the image edge
    if rows.start < 0 or columns.start < 0 or rows.stop > science.shape[0] or columns.stop > science.shape[1]:
        return

    science[rows, columns] += stamp
    difference[rows, columns] += stamp
