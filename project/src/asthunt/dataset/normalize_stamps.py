"""Put every stamp on the same scale so a model compares shapes, not raw brightness."""

import numpy as np


def normalize_stamps(stamps):
    """Return float32 stamps where each channel of each stamp is (value - median) / noise.

    - Subtracting the median removes the sky level (it differs per night).
    - Dividing by the noise turns pixel values into "sigmas": 5 means
      5 x the noise, on any night. We estimate the noise robustly with the
      median absolute deviation (MAD), so a bright star in the stamp
      doesn't inflate it:  noise = 1.4826 x median(|value - median|).
    - Finally we clip to [-10, 50] sigma so a saturated star can't dominate.
    """
    stamps = stamps.astype(np.float32)
    median = np.median(stamps, axis=(2, 3), keepdims=True)
    mad = np.median(np.abs(stamps - median), axis=(2, 3), keepdims=True)
    noise = 1.4826 * mad
    noise[noise == 0] = 1.0                         # empty (all-zero) channels stay zero
    normalized = (stamps - median) / noise
    return np.clip(normalized, -10, 50)
