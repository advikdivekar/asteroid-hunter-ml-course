import numpy as np
import pandas as pd

from asthunt.deep.cut_stacked_stamps import cut_stacked_stamps


def test_object_lands_in_the_centre_of_every_stack():
    frames, size = 8, 100
    times = 2459000 + np.arange(frames) * 0.1 / 24                  # 6 minutes apart
    offsets = np.array([[0, 0], [1, -1], [2, 0], [0, 2], [-1, 1], [1, 1], [0, -2], [2, 2]])
    snr = np.zeros((frames, size, size), dtype=np.float32)
    for k in range(frames):
        hours = k * 0.1
        snr[k, 40 + int(round(-10 * hours)) + offsets[k][1], 30 + int(round(20 * hours)) + offsets[k][0]] = 2.0
    tracklet = pd.Series({"x0": 30.0, "y0": 40.0, "vx": 20.0, "vy": -10.0, "t0_jd": times[0]})

    stamps = cut_stacked_stamps(snr, np.ones(snr.shape, bool), times, offsets, [[0, 1, 2, 3], [4, 5, 6, 7]], tracklet, size=11)

    assert np.isclose(stamps[0, 5, 5], 4 * 2.0 / 2) and np.isclose(stamps[1, 5, 5], 4.0)     # 4 frames x 2 / sqrt(4)
