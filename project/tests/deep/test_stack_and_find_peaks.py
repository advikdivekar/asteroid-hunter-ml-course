import numpy as np

from asthunt.deep.make_velocity_grid import make_velocity_grid
from asthunt.deep.merge_peaks import merge_peaks
from asthunt.deep.stack_and_find_peaks import stack_and_find_peaks


def test_a_source_invisible_in_single_frames_appears_in_the_stack():
    rng = np.random.default_rng(0)
    frames, size = 8, 120
    hours = np.arange(frames) * 0.07                          # 8 frames over ~0.5 hour
    offsets = np.array([[0, 0], [2, -1], [3, 1], [-1, 2], [0, 0], [1, 1], [-2, 0], [2, 2]])
    vx, vy = 40.0, -20.0                                       # pixels per hour
    snr = rng.normal(0, 1, size=(frames, size, size)).astype(np.float32)
    for k in range(frames):                                    # 2.5 sigma per frame: hard to see in any one frame
        x = 50 + int(round(vx * hours[k])) + offsets[k][0]
        y = 60 + int(round(vy * hours[k])) + offsets[k][1]
        snr[k, y, x] += 2.5
    good = np.ones(snr.shape, dtype=bool)

    peaks = stack_and_find_peaks(snr, good, hours, offsets, make_velocity_grid(60, 4, 1.0), threshold=5.5)
    best = merge_peaks(peaks).iloc[0]

    assert (best["x"], best["y"]) == (50, 60)
    assert abs(best["vx"] - vx) <= 4 and abs(best["vy"] - vy) <= 4
    assert best["snr"] > 6                                      # 8 x 2.5 / sqrt(8) = 7.1 sigma expected


def test_velocity_grid_includes_zero_and_respects_the_limit():
    grid = make_velocity_grid(10, 2, 1.0)
    assert [0.0, 0.0] in grid.tolist()
    assert np.hypot(grid[:, 0], grid[:, 1]).max() <= 10
