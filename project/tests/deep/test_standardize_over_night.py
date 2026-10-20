import numpy as np

from asthunt.deep.standardize_over_night import standardize_over_night


def test_static_and_wobbling_structure_fade_moving_source_survives():
    rng = np.random.default_rng(0)
    frames, size = 15, 80
    offsets = rng.integers(-3, 4, size=(frames, 2))
    snr = rng.normal(0, 1, size=(frames, size, size)).astype(np.float32)
    for k in range(frames):
        dx, dy = offsets[k]
        snr[k, 30 + dy:40 + dy, 30 + dx:40 + dx] += 8.0                          # static residual
        snr[k, 50 + dy:56 + dy, 50 + dx:56 + dx] += rng.normal(0, 6)              # residual that wobbles each frame
        snr[k, 70 + dy, 4 + 5 * k + dx] += 6.0                                    # a source moving 5 px per frame
    good = np.ones(snr.shape, dtype=bool)

    cleaned = standardize_over_night(snr, good, offsets)

    dx, dy = offsets[0]
    assert np.abs(cleaned[0, 32 + dy:38 + dy, 32 + dx:38 + dx]).mean() < 1.5      # static blob gone
    wobble = np.array([cleaned[k, 52 + offsets[k][1], 52 + offsets[k][0]] for k in range(frames)])
    assert np.abs(wobble).max() < 4                                               # wobbles turned down
    k = 6
    assert cleaned[k, 70 + offsets[k][1], 4 + 5 * k + offsets[k][0]] > 4          # mover still there
