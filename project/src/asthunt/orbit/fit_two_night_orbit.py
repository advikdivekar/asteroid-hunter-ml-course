"""Does one two-body orbit explain both nights, and how far away can the body be?"""

import numpy as np
from scipy.optimize import least_squares

from asthunt.orbit.observer_state import observer_state
from asthunt.orbit.orbital_elements import orbital_elements
from asthunt.orbit.predict_sky_motion import attributable_times, predict_attributable, state_from_attributable

KM_PER_SECOND = 86400 / 149597870.7            # AU/day per km/s


def fit_two_night_orbit(night_1, night_2, distance, radial_velocity_starts=(-15.0, 0.0, 15.0)):
    """Best two-body orbit for both nights when the body is `distance` AU from the telescope on night 1.

    night_1, night_2 : {"jd", "ra", "dec", "v_ra", "v_dec", "cov"}: what we measured (degrees; arcsec/hour with
                       cos(dec) in v_ra; cov is 4x4 in arcsec, arcsec, arcsec/hour, arcsec/hour)
    Four numbers per night is 8 measurements. The orbit has 6 unknowns, and the sky motion cannot say how far the body
    is, so we fix the distance and fit the rest: night 1's own attributable (within its error), and the radial velocity.
    Fixing the distance leaves 8 - 5 = 3 degrees of freedom.

    Returns {"chi2", "dof", "radial_velocity" (km/s), "elements" (see orbital_elements), "residuals" (whitened),
             "state" ([position, velocity] heliocentric ICRS, AU and AU/day) and "epoch_jd" (when that state holds)}:
    the last two let a caller predict where the body was or will be on other dates.
    """
    observers = {"one": observer_state([night_1["jd"]]),
                 "two": observer_state(attributable_times(night_2["jd"]))}
    whiten_1, whiten_2 = (np.linalg.inv(np.linalg.cholesky(night["cov"])) for night in (night_1, night_2))
    cos_dec_1 = np.cos(np.radians(night_1["dec"]))
    cos_dec_2 = np.cos(np.radians(night_2["dec"]))

    def state_for(parameters):
        offsets, radial = parameters[:4], parameters[4]
        state, epoch = state_from_attributable(
            night_1["ra"] + offsets[0] / 3600 / cos_dec_1, night_1["dec"] + offsets[1] / 3600,
            night_1["v_ra"] + offsets[2], night_1["v_dec"] + offsets[3], distance, radial * KM_PER_SECOND,
            night_1["jd"], observer=observers["one"])
        return state, epoch

    def residuals(parameters):
        with np.errstate(all="ignore"):
            state, epoch = state_for(parameters)
            ra, dec, v_ra, v_dec = predict_attributable(state, epoch, night_2["jd"], observer=observers["two"])
            difference = np.array([(ra - night_2["ra"]) * cos_dec_2 * 3600, (dec - night_2["dec"]) * 3600,
                                   v_ra - night_2["v_ra"], v_dec - night_2["v_dec"]])
            values = np.concatenate([whiten_1 @ parameters[:4], whiten_2 @ difference])
        # A wildly wrong guess can give an orbit the solver cannot follow (overflow, no root): call it a terrible fit, not a crash
        return np.where(np.isfinite(values), values, 1e3)

    best = None
    for radial in radial_velocity_starts:
        solution = least_squares(residuals, np.array([0, 0, 0, 0, radial], dtype=float), x_scale=[1, 1, 1, 1, 10], xtol=1e-12, ftol=1e-12)
        if best is None or solution.cost < best.cost:
            best = solution
    state, epoch = state_for(best.x)
    return {"chi2": float(2 * best.cost), "dof": 3, "radial_velocity": float(best.x[4]),
            "elements": orbital_elements(*state), "residuals": best.fun.tolist(),
            "state": [state[0].tolist(), state[1].tolist()], "epoch_jd": float(epoch)}


def profile_distance(night_1, night_2, distances, radial_velocity_starts=(-15.0, 0.0, 15.0)):
    """fit_two_night_orbit for each distance (AU): the chi-square profile that says which distances the data allow."""
    return [{"distance": float(d), **fit_two_night_orbit(night_1, night_2, d, radial_velocity_starts)} for d in distances]
