import numpy as np

from asthunt.orbit.kepler_propagate import GM_SUN, kepler_propagate


def circular_state(radius):
    return np.array([radius, 0.0, 0.0]), np.array([0.0, np.sqrt(GM_SUN / radius), 0.0])


def test_a_circular_orbit_returns_to_its_start_after_one_period():
    position, velocity = circular_state(2.5)
    period = 2 * np.pi * np.sqrt(2.5 ** 3 / GM_SUN)
    new_position, new_velocity = kepler_propagate(position, velocity, period)
    assert np.allclose(new_position, position, atol=1e-9)
    assert np.allclose(new_velocity, velocity, atol=1e-11)


def test_a_quarter_period_rotates_by_ninety_degrees():
    position, velocity = circular_state(1.0)
    period = 2 * np.pi * np.sqrt(1.0 / GM_SUN)
    new_position, _ = kepler_propagate(position, velocity, period / 4)
    assert np.allclose(new_position, [0, 1, 0], atol=1e-9)


def test_an_eccentric_orbit_conserves_energy_and_angular_momentum():
    position, velocity = np.array([1.0, 0.2, 0.1]), np.array([-0.004, 0.015, 0.002])
    energy = lambda p, v: v @ v / 2 - GM_SUN / np.linalg.norm(p)
    for days in [-30.0, 0.5, 3.0, 400.0]:
        new_position, new_velocity = kepler_propagate(position, velocity, days)
        assert np.isclose(energy(new_position, new_velocity), energy(position, velocity), rtol=1e-10)
        assert np.allclose(np.cross(new_position, new_velocity), np.cross(position, velocity), atol=1e-12)


def test_going_forward_then_back_gives_the_start():
    position, velocity = np.array([2.0, -1.0, 0.3]), np.array([0.005, 0.009, -0.001])
    there = kepler_propagate(position, velocity, 12.0)
    back = kepler_propagate(*there, -12.0)
    assert np.allclose(back[0], position, atol=1e-11) and np.allclose(back[1], velocity, atol=1e-13)


def test_a_hyperbolic_orbit_works():
    position, velocity = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.03, 0.0])      # faster than escape speed
    new_position, new_velocity = kepler_propagate(position, velocity, 50.0)
    energy = lambda p, v: v @ v / 2 - GM_SUN / np.linalg.norm(p)
    assert energy(position, velocity) > 0
    assert np.isclose(energy(new_position, new_velocity), energy(position, velocity), rtol=1e-10)
