"""Where the ZTF telescope is, relative to the Sun."""

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel
from astropy.time import Time

# Palomar Observatory, the Samuel Oschin telescope (MPC code I41)
PALOMAR = EarthLocation.from_geodetic(lon=-116.8599 * u.deg, lat=33.3564 * u.deg, height=1672 * u.m)


def observer_state(jd_utc):
    """Return (position, velocity), each shaped (n, 3): the telescope relative to the Sun, in AU and AU/day.

    Axes are the equatorial ICRS axes, the same ones RA and Dec are measured in. The Earth comes from
    astropy's built-in ephemeris (accurate to ~1e-5 AU: 0.001 arcsec for an object 1 AU away) and
    the telescope's offset from the Earth's centre includes the planet's rotation, which matters
    for objects within a few AU (a 6,371 km baseline is ~6 arcsec at 1.5 AU).
    """
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    time = Time(jd_utc, format="jd", scale="utc")
    earth_position, earth_velocity = get_body_barycentric_posvel("earth", time, ephemeris="builtin")
    sun_position, sun_velocity = get_body_barycentric_posvel("sun", time, ephemeris="builtin")
    site_position, site_velocity = PALOMAR.get_gcrs_posvel(time)

    position = (earth_position - sun_position).xyz.to(u.AU) + site_position.xyz.to(u.AU)
    velocity = (earth_velocity - sun_velocity).xyz.to(u.AU / u.day) + site_velocity.xyz.to(u.AU / u.day)
    return position.value.T, velocity.value.T
