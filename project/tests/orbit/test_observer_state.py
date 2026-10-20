import numpy as np

from asthunt.orbit.observer_state import observer_state


def test_the_telescope_is_about_one_au_from_the_sun_and_moves_at_earth_speed():
    position, velocity = observer_state([2458511.8, 2458514.8])
    assert position.shape == (2, 3)
    assert np.all(np.abs(np.linalg.norm(position, axis=1) - 0.985) < 0.01)       # late January: near perihelion
    assert np.all(np.abs(np.linalg.norm(velocity, axis=1) - 0.0175) < 0.0005)    # ~30.3 km/s = 0.0175 AU/day


def test_the_telescope_moves_with_the_earth_between_nights():
    position, _ = observer_state([2458511.8, 2458514.8])
    assert 0.04 < np.linalg.norm(position[1] - position[0]) < 0.06                # ~3 days x 0.0175 AU/day
