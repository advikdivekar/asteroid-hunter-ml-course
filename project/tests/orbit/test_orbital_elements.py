import numpy as np

from asthunt.orbit.kepler_propagate import GM_SUN
from asthunt.orbit.orbital_elements import OBLIQUITY, orbital_elements


def equatorial(vector):
    """Ecliptic axes -> the equatorial axes our state vectors use."""
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ vector


def test_a_circular_orbit_in_the_ecliptic():
    speed = np.sqrt(GM_SUN / 2.5)
    elements = orbital_elements(equatorial(np.array([2.5, 0, 0])), equatorial(np.array([0, speed, 0])))
    assert abs(elements["a"] - 2.5) < 1e-9 and elements["e"] < 1e-9 and elements["i"] < 1e-6
    assert abs(elements["q"] - 2.5) < 1e-9 and abs(elements["Q"] - 2.5) < 1e-9


def test_an_eccentric_inclined_orbit_matches_hand_values():
    # perihelion at 1.0 AU with a speed that makes a = 2.0 AU, e = 0.5, inclined 10 degrees
    q, a = 1.0, 2.0
    speed = np.sqrt(GM_SUN * (2 / q - 1 / a))
    tilt = np.radians(10)
    position = np.array([q, 0, 0])
    velocity = speed * np.array([0, np.cos(tilt), np.sin(tilt)])
    elements = orbital_elements(equatorial(position), equatorial(velocity))
    assert abs(elements["a"] - 2.0) < 1e-9 and abs(elements["e"] - 0.5) < 1e-9
    assert abs(elements["i"] - 10.0) < 1e-6
    assert abs(elements["q"] - 1.0) < 1e-9 and abs(elements["Q"] - 3.0) < 1e-9


def test_an_unbound_orbit_has_no_aphelion():
    elements = orbital_elements(np.array([1.0, 0, 0]), np.array([0, 0.03, 0]))
    assert elements["a"] < 0 and elements["e"] > 1 and elements["Q"] == float("inf")
