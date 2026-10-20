"""Fit straight-line motion to a set of detections of one object."""

import numpy as np


def fit_linear_motion(times_hours, x_pixels, y_pixels):
    """Fit x(t) and y(t) with straight lines and report how well they fit.

    Asteroids move in (almost) straight lines over one night, so:
        x(t) = x0 + vx * t
        y(t) = y0 + vy * t

    Returns a dict with:
        vx, vy        : velocity in pixels per hour
        x0, y0        : position at t = 0
        max_residual  : the largest distance (pixels) between a detection
                        and the fitted line. Small = straight track.
    """
    t = np.asarray(times_hours, dtype=float)
    x = np.asarray(x_pixels, dtype=float)
    y = np.asarray(y_pixels, dtype=float)

    # np.polyfit with degree 1 returns [slope, intercept]
    vx, x0 = np.polyfit(t, x, 1)
    vy, y0 = np.polyfit(t, y, 1)

    # Distance from each detection to where the line says it should be
    x_predicted = x0 + vx * t
    y_predicted = y0 + vy * t
    residuals = np.sqrt((x - x_predicted) ** 2 + (y - y_predicted) ** 2)

    return {
        "vx": vx,
        "vy": vy,
        "x0": x0,
        "y0": y0,
        "max_residual": residuals.max(),
    }
