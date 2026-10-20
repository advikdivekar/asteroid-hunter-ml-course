import numpy as np

from asthunt.orbit.fit_two_night_orbit import KM_PER_SECOND, fit_two_night_orbit, profile_distance
from asthunt.orbit.predict_sky_motion import predict_attributable, state_from_attributable

JD_1, JD_2 = 2458511.8, 2458514.75


def synthetic_nights(distance, radial_km_s, sigma=(0.15, 0.15, 0.3, 0.3)):
    """A body 'seen' on two nights: night 1 is what state_from_attributable says, night 2 is what its orbit predicts."""
    state, epoch = state_from_attributable(130.0, 18.0, -31.0, 5.0, distance, radial_km_s * KM_PER_SECOND, JD_1)
    one = {"jd": JD_1, "ra": 130.0, "dec": 18.0, "v_ra": -31.0, "v_dec": 5.0, "cov": np.diag(np.square(sigma))}
    ra, dec, v_ra, v_dec = predict_attributable(state, epoch, JD_2)
    two = {"jd": JD_2, "ra": ra, "dec": dec, "v_ra": v_ra, "v_dec": v_dec, "cov": np.diag(np.square(sigma))}
    return one, two


def test_the_true_distance_fits_perfectly_and_the_orbit_is_recovered():
    one, two = synthetic_nights(distance=1.45, radial_km_s=5.0)
    fit = fit_two_night_orbit(one, two, distance=1.45)
    assert fit["chi2"] < 1e-3
    assert abs(fit["radial_velocity"] - 5.0) < 0.1


def test_a_wrong_distance_fits_worse_than_the_true_one():
    one, two = synthetic_nights(distance=1.45, radial_km_s=5.0)
    profile = {round(row["distance"], 2): row["chi2"] for row in profile_distance(one, two, [0.5, 1.45, 4.0])}
    assert profile[1.45] < 0.01
    assert profile[0.5] > profile[1.45] + 1 and profile[4.0] > profile[1.45] + 1
    # and a night 2 that belongs to a different motion fits badly at EVERY distance
    one, two = synthetic_nights(distance=1.45, radial_km_s=5.0)
    two["v_ra"] += 6.0                                          # 6 arcsec/hour off: 20 sigma
    assert min(row["chi2"] for row in profile_distance(one, two, np.geomspace(0.3, 6, 12))) > 20


def test_a_hopeless_pairing_is_a_bad_fit_not_a_crash():
    one, two = synthetic_nights(distance=1.45, radial_km_s=5.0)
    two["ra"] += 3.0                                           # three degrees away: no sensible orbit connects them
    rows = profile_distance(one, two, [0.2, 1.45, 8.0, 30.0])
    assert all(np.isfinite(row["chi2"]) and row["chi2"] > 50 for row in rows)
