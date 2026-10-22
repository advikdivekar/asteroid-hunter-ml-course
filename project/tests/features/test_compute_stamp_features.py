import numpy as np
import pandas as pd

from asthunt.features.compute_stamp_features import FEATURE_COLUMNS, compute_stamp_features


def make_stamps():
    """Four kinds of stamp, on the normalized scale (noise = 1)."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:31, :31] - 15
    star = 20 * np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.3 ** 2))
    stamps = rng.normal(0, 1, size=(4, 3, 31, 31)).astype(np.float32)
    stamps[0, 0] += star; stamps[0, 2] += star                       # new object: science + difference, not reference
    stamps[1, 0] += star; stamps[1, 1] += star                       # a star that subtracted cleanly
    stamps[1, 2] += np.roll(star, 1, axis=1) - star                  # ... but slightly misaligned: a dipole
    stamps[2, 0, 15, 15] += 40; stamps[2, 2, 15, 15] += 40           # cosmic ray: one hot pixel
    # stamps[3]: noise only
    detections = pd.DataFrame({c: np.ones(4) for c in ["snr", "peak", "npix", "a", "b", "flux"]})
    return stamps, detections


def test_one_row_per_stamp_with_every_column():
    stamps, detections = make_stamps()
    table = compute_stamp_features(stamps, detections)
    assert list(table.columns) == FEATURE_COLUMNS and len(table) == 4
    assert np.isfinite(table.to_numpy()).all()


def test_features_tell_the_kinds_apart():
    stamps, detections = make_stamps()
    t = compute_stamp_features(stamps, detections)
    new, dipole, cosmic, noise = 0, 1, 2, 3
    assert t.loc[new, "diff_psf_match"] > 0.8 > t.loc[noise, "diff_psf_match"]     # star-shaped in the difference
    assert t.loc[dipole, "ref_to_sci"] > 0.5 > t.loc[new, "ref_to_sci"]            # already there in the old picture
    assert t.loc[dipole, "diff_dipole"] > t.loc[new, "diff_dipole"]                # strong negative next to positive
    assert t.loc[cosmic, "diff_sharpness"] > t.loc[new, "diff_sharpness"]          # one hot pixel, sharper than a star
