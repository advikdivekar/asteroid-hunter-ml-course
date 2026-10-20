import pytest

from asthunt.link.fit_linear_motion import fit_linear_motion


def test_perfect_straight_line():
    times = [0.0, 0.5, 1.0, 1.5]
    x = [100 + 10 * t for t in times]   # 10 px/hr in x
    y = [200 - 4 * t for t in times]    # -4 px/hr in y

    fit = fit_linear_motion(times, x, y)

    assert fit["vx"] == pytest.approx(10)
    assert fit["vy"] == pytest.approx(-4)
    assert fit["x0"] == pytest.approx(100)
    assert fit["max_residual"] == pytest.approx(0, abs=1e-9)


def test_one_bad_point_gives_large_residual():
    times = [0.0, 0.5, 1.0, 1.5]
    x = [100, 105, 110 + 8, 115]        # third detection is 8 px off
    y = [200, 200, 200, 200]

    fit = fit_linear_motion(times, x, y)

    assert fit["max_residual"] > 3
