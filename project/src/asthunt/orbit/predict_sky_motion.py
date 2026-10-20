"""Where, and how fast, a body on a given orbit appears to move as seen from ZTF."""

import numpy as np

from asthunt.orbit.kepler_propagate import kepler_propagate
from asthunt.orbit.observer_state import observer_state

LIGHT_SPEED = 173.1446326846693        # AU / day
ARCSEC_PER_RADIAN = 206264.80624709636


def predict_sky_position(state, epoch_jd, jd_utc, observer=None):
    """Return (ra_deg, dec_deg) of the body at each time in jd_utc, as the telescope sees it.

    state    : (position, velocity), heliocentric ICRS, AU and AU/day, valid at `epoch_jd`
    observer : optional precomputed observer_state(jd_utc), to avoid recomputing it in a fit

    The light we see left the body earlier (1.5 AU is 12.5 minutes, during which an asteroid moves
    ~15,000 km, ~14 arcsec): propagate to the time of emission, found by iteration.
    """
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    observer_position, _ = observer if observer is not None else observer_state(jd_utc)
    ra, dec = [], []
    for jd, observer_at in zip(jd_utc, observer_position):
        light_time = 0.0
        for _ in range(4):
            position, _ = kepler_propagate(state[0], state[1], jd - light_time - epoch_jd)
            line_of_sight = position - observer_at
            light_time = np.linalg.norm(line_of_sight) / LIGHT_SPEED
        x, y, z = line_of_sight / np.linalg.norm(line_of_sight)
        ra.append(np.degrees(np.arctan2(y, x)) % 360)
        dec.append(np.degrees(np.arcsin(z)))
    return np.array(ra), np.array(dec)


def predict_attributable(state, epoch_jd, jd_utc, half_span_hours=0.5, observer=None):
    """Return (ra, dec, v_ra, v_dec) at jd_utc: degrees, and arcsec/hour (v_ra includes cos(dec)).

    The rate is the slope between jd - half_span and jd + half_span, the same kind of
    straight-line slope the night's frames give us.
    observer : optional precomputed observer_state([jd - half_span, jd, jd + half_span]), for fits that call this often
    """
    jd = float(jd_utc)
    times = attributable_times(jd, half_span_hours)
    ra, dec = predict_sky_position(state, epoch_jd, times, observer=observer)
    d_ra = (ra[2] - ra[0] + 180) % 360 - 180                 # RA wraps at 360
    cos_dec = np.cos(np.radians(dec[1]))
    hours = 2 * half_span_hours
    return ra[1], dec[1], d_ra * cos_dec * 3600 / hours, (dec[2] - dec[0]) * 3600 / hours


def attributable_times(jd_utc, half_span_hours=0.5):
    """The three times predict_attributable looks at (so a caller can compute observer_state for them once)."""
    return np.array([jd_utc - half_span_hours / 24, jd_utc, jd_utc + half_span_hours / 24])


def state_from_attributable(ra, dec, v_ra, v_dec, distance, distance_rate, jd_utc, observer=None):
    """Return ((position, velocity), epoch_jd): the body's heliocentric state from what we measured on one night.

    ra, dec           : degrees. v_ra (with cos dec) and v_dec: arcsec/hour
    distance          : AU from the telescope (unknown from the sky motion alone: a free parameter)
    distance_rate     : AU/day along the line of sight (the other unknown)
    observer      : optional precomputed observer_state([jd_utc]), for fits that call this often
    The state is for the moment the light left the body, jd_utc - distance / c.
    """
    observer_position, observer_velocity = (a[0] for a in (observer if observer is not None else observer_state([jd_utc])))
    a, d = np.radians(ra), np.radians(dec)
    line_of_sight = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    east = np.array([-np.sin(a), np.cos(a), 0.0])
    north = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    per_day = 24 / ARCSEC_PER_RADIAN                         # arcsec/hour -> radians/day
    angular_velocity = v_ra * per_day * east + v_dec * per_day * north     # d(line of sight)/dt
    position = observer_position + distance * line_of_sight
    velocity = observer_velocity + distance_rate * line_of_sight + distance * angular_velocity
    return (position, velocity), jd_utc - distance / LIGHT_SPEED
