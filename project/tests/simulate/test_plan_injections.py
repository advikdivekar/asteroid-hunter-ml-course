import numpy as np

from asthunt.simulate.plan_injections import plan_injections


def test_fakes_move_in_straight_lines_at_requested_speed():
    times = [2459000.0, 2459000.0 + 1 / 24, 2459000.0 + 2 / 24]      # 0, 1 and 2 hours
    plan = plan_injections(10, (1000, 1000), times, (18, 20), (36, 36),
                           pixel_scale=1.0, edge_margin=50, rng=np.random.default_rng(0))

    assert len(plan) == 30
    one = plan[plan["injection_id"] == 3].sort_values("frame_index")
    step = np.hypot(np.diff(one["x"]), np.diff(one["y"]))
    assert np.allclose(step, 36.0)                                    # 36 "/hr at 1 "/px = 36 px per hour
    assert plan["mag"].between(18, 20).all()


def test_zero_fakes_still_has_columns():
    plan = plan_injections(0, (100, 100), [2459000.0], (18, 20), (10, 20), 1.0, 10, np.random.default_rng(0))
    assert len(plan) == 0
    assert {"injection_id", "frame_index", "x", "y"} <= set(plan.columns)
