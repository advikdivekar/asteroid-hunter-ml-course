"""Orbital elements from a heliocentric position and velocity."""

import numpy as np

from asthunt.orbit.kepler_propagate import GM_SUN

OBLIQUITY = np.radians(23.4392911)         # the tilt between the equator (our axes) and the ecliptic (where orbits are quoted)


def orbital_elements(position, velocity):
    """Return a dict: a (AU, negative for a hyperbola), e, i (degrees, to the ecliptic), q and Q (AU, perihelion and aphelion; Q is inf if unbound), r (AU now).

    position, velocity: heliocentric ICRS axes, AU and AU/day.
    """
    to_ecliptic = np.array([[1, 0, 0], [0, np.cos(OBLIQUITY), np.sin(OBLIQUITY)], [0, -np.sin(OBLIQUITY), np.cos(OBLIQUITY)]])
    r_vec, v_vec = to_ecliptic @ np.asarray(position, dtype=float), to_ecliptic @ np.asarray(velocity, dtype=float)
    r, speed_squared = np.linalg.norm(r_vec), v_vec @ v_vec
    angular_momentum = np.cross(r_vec, v_vec)
    energy = speed_squared / 2 - GM_SUN / r
    eccentricity = np.linalg.norm(np.cross(v_vec, angular_momentum) / GM_SUN - r_vec / r)
    semi_major = -GM_SUN / (2 * energy) if energy != 0 else np.inf
    perihelion = (angular_momentum @ angular_momentum) / (GM_SUN * (1 + eccentricity))
    return {"a": float(semi_major), "e": float(eccentricity),
            "i": float(np.degrees(np.arccos(angular_momentum[2] / np.linalg.norm(angular_momentum)))),
            "q": float(perihelion), "Q": float(semi_major * (1 + eccentricity)) if energy < 0 else float("inf"), "r": float(r)}
