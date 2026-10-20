"""Draw a moving source that smears into a short line (a trail) during the exposure."""

import numpy as np

from asthunt.simulate.render_point_source import render_point_source


def render_trailed_source(psf_model, total_flux, length_pixels, angle_degrees, size=None):
    """Return a stamp (at least size x size) of a trail: the PSF smeared along a line of the given length and angle.

    During an exposure a fast object moves, so its light is spread over every
    position along its path. We approximate that by adding many copies of the
    PSF, each with an equal share of the flux, at evenly spaced points along the
    line (one copy per half pixel is plenty). The total flux is exact.
    """
    needed = int(2 * np.ceil(length_pixels / 2) + psf_model.shape[0])
    size = max(size or 0, needed)                              # never smaller than the trail itself needs
    size += (size + 1) % 2                                     # odd, so there is a centre pixel
    n_points = max(2, int(np.ceil(length_pixels * 2)) + 1)
    positions = np.linspace(-length_pixels / 2, length_pixels / 2, n_points)
    direction = np.array([np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))])

    stamp = np.zeros((size, size))
    centre = size // 2
    half = psf_model.shape[0] // 2
    for distance in positions:
        x, y = centre + distance * direction
        column, row = int(round(x)), int(round(y))
        piece = render_point_source(psf_model, total_flux / n_points, x - column, y - row)
        stamp[row - half:row + half + 1, column - half:column + half + 1] += piece
    return stamp
