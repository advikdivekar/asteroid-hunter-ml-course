import numpy as np
import pandas as pd

from asthunt.astrometry.calibrate_fits import calibrate_fit, summarize_calibration


def known_table(rng, bias=(0.0, 0.10, 0.0, 0.0), floor=(0.05, 0.05, 0.08, 0.08), scale=1.3, n=60):
    """Fake calibration objects: bright ones scatter by the floor, faint ones by scale x their statistical sigma."""
    rows = []
    for i in range(n):
        snr = rng.uniform(10, 200)
        stat = np.array([0.6, 0.6, 0.9, 0.9]) / snr * 3                  # statistical sigma shrinks as SNR grows
        sigma = np.sqrt((scale * stat) ** 2 + np.array(floor) ** 2 * (snr >= 30))
        d = np.array(bias) + rng.normal(size=4) * sigma
        rows.append({"name": f"obj{i}", "snr": snr, **dict(zip(["d_ra", "d_dec", "d_v_ra", "d_v_dec"], d)),
                     **dict(zip(["sigma_ra", "sigma_dec", "sigma_v_ra", "sigma_v_dec"], stat))})
    return pd.DataFrame(rows)


def test_the_bias_floor_and_scale_are_recovered():
    summary = summarize_calibration(known_table(np.random.default_rng(0)))
    assert abs(summary["bias"][1] - 0.10) < 0.03 and abs(summary["bias"][0]) < 0.03
    assert 0.02 < summary["floor"][0] < 0.09 and 0.04 < summary["floor"][2] < 0.13
    assert 1.0 <= summary["scale"] < 2.0


def test_a_fit_that_found_another_object_or_has_an_implausible_error_is_left_out():
    table = known_table(np.random.default_rng(1))
    table.loc[0, ["d_ra", "d_dec"]] = [16.0, -15.0]                        # locked onto a neighbour
    table.loc[1, ["sigma_ra", "sigma_dec"]] = 1e-4                         # a spike, not a source
    summary = summarize_calibration(table)
    assert {"obj0", "obj1"} <= set(summary["left_out"])
    assert summary["scale"] < 3                                            # one wild z-score must not decide the scale


def test_calibrating_a_fit_removes_the_bias_and_widens_the_errors():
    fit = {"jd": 1.0, "ra": 125.0, "dec": 20.0, "v_ra": -38.0, "v_dec": 6.0, "cov": np.diag([0.01, 0.01, 0.01, 0.01])}
    summary = {"bias": [0.0, 0.10, 0.02, 0.0], "bias_error": [0.01] * 4, "floor": [0.05] * 4, "scale": 1.5}
    out = calibrate_fit(fit, summary)
    assert abs((out["dec"] - 20.0) * 3600 + 0.10) < 1e-9 and abs(out["v_ra"] - (-38.02)) < 1e-12
    assert np.isclose(out["cov"][0, 0], 1.5 ** 2 * 0.01 + 0.05 ** 2 + 0.01 ** 2)
