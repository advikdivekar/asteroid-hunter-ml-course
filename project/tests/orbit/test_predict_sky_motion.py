import numpy as np

from asthunt.orbit.observer_state import observer_state
from asthunt.orbit.predict_sky_motion import predict_attributable, predict_sky_position, state_from_attributable

JD = 2458511.8


def test_a_state_built_from_an_attributable_reproduces_that_attributable():
    state, epoch = state_from_attributable(130.0, 18.0, -31.0, 5.0, distance=1.4, distance_rate=0.002, jd_utc=JD)
    ra, dec, v_ra, v_dec = predict_attributable(state, epoch, JD)
    assert abs(ra - 130.0) < 2e-5 and abs(dec - 18.0) < 2e-5                     # < 0.1 arcsec
    assert abs(v_ra - (-31.0)) < 0.05 and abs(v_dec - 5.0) < 0.05


def test_distance_changes_where_the_body_is_seen_later():
    # same sky position and motion tonight, different distance: they must part ways over three days
    near, near_epoch = state_from_attributable(130.0, 18.0, -31.0, 5.0, 0.8, 0.0, JD)
    far, far_epoch = state_from_attributable(130.0, 18.0, -31.0, 5.0, 2.5, 0.0, JD)
    a = predict_sky_position(near, near_epoch, [JD + 3.0])
    b = predict_sky_position(far, far_epoch, [JD + 3.0])
    separation = np.hypot((a[0][0] - b[0][0]) * np.cos(np.radians(20.5)), a[1][0] - b[1][0]) * 3600
    assert separation > 100


def test_light_time_shifts_the_position_by_the_distance_the_body_moved_while_the_light_travelled():
    state, epoch = state_from_attributable(130.0, 18.0, -31.0, 5.0, 1.5, 0.0, JD)
    observer = observer_state([JD])
    with_light_time = predict_sky_position(state, epoch, [JD])
    # the same state treated as if light were instantaneous: propagate nothing, look straight at the epoch position
    straight = state[0] - observer[0][0]
    ra = np.degrees(np.arctan2(straight[1], straight[0]))
    assert abs(with_light_time[0][0] - 130.0) < 2e-5                              # the state came from this very geometry
    assert abs(ra - 130.0) < 0.01
