"""Move a body along its two-body orbit around the Sun."""

import numpy as np

GM_SUN = 0.01720209895 ** 2        # AU^3 / day^2: the Gaussian gravitational constant, squared


def kepler_propagate(position, velocity, days):
    """Return (position, velocity) after `days`, from a heliocentric state in AU and AU/day.

    Works for ellipses, parabolas and hyperbolas (universal variables: one equation, solved for
    the "universal anomaly" chi with Newton's method). Only the Sun pulls: over a few days that is
    all a main-belt asteroid or NEO needs at the arcsecond level we care about here (planet
    perturbations over 3 days are far below a pixel; the orbit analysis states this limit).
    """
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    r0, v0 = np.linalg.norm(position), np.linalg.norm(velocity)
    radial = position @ velocity / r0
    root_mu = np.sqrt(GM_SUN)
    alpha = 2.0 / r0 - v0 ** 2 / GM_SUN                      # 1 / semi-major axis (negative: hyperbola)

    chi = root_mu * abs(alpha) * days if alpha != 0 else root_mu * days / r0
    for _ in range(100):
        z = alpha * chi ** 2
        c, s = _stumpff(z)
        f = r0 * radial / root_mu * chi ** 2 * c + (1 - alpha * r0) * chi ** 3 * s + r0 * chi - root_mu * days
        slope = r0 * radial / root_mu * chi * (1 - z * s) + (1 - alpha * r0) * chi ** 2 * c + r0
        step = f / slope
        chi -= step
        if abs(step) < 1e-13 * max(1.0, abs(chi)):
            break

    z = alpha * chi ** 2
    c, s = _stumpff(z)
    f = 1 - chi ** 2 / r0 * c
    g = days - chi ** 3 / root_mu * s
    new_position = f * position + g * velocity
    r = np.linalg.norm(new_position)
    f_dot = root_mu / (r * r0) * (z * s - 1) * chi
    g_dot = 1 - chi ** 2 / r * c
    return new_position, f_dot * position + g_dot * velocity


def _stumpff(z):
    """The two Stumpff functions C(z) and S(z) that make one formula cover every conic section."""
    if abs(z) < 1e-6:                                         # series: the closed forms lose precision near 0
        return 0.5 - z / 24 + z ** 2 / 720, 1 / 6 - z / 120 + z ** 2 / 5040
    if z > 0:
        root = np.sqrt(z)
        return (1 - np.cos(root)) / z, (root - np.sin(root)) / root ** 3
    root = np.sqrt(-z)
    return (np.cosh(root) - 1) / -z, (np.sinh(root) - root) / root ** 3
