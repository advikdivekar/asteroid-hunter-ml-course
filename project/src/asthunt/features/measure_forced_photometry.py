"""Measure brightness at a tracklet's PREDICTED position in every frame (detected or not)."""

import numpy as np
import pandas as pd

FORCED_COLUMNS = ["forced_snr_min", "forced_snr_median", "forced_snr_std", "stack_snr"]


def measure_forced_photometry(tracklet_stamps, aperture_radius=2.0):
    """Return a table with one row per tracklet and the columns in FORCED_COLUMNS.

    tracklet_stamps: normalized stamps (n, frames, 3, h, w) cut along each predicted
                     path (cut_tracklet_stamps + normalize_stamps). Normalized means
                     every pixel is in units of the noise (sigma), so:

        aperture SNR in one frame = (sum of difference pixels inside the circle) / sqrt(number of pixels)

    Features:
      forced_snr_min     the weakest frame. A real asteroid is present in every
                         frame (maybe faintly); a chance alignment of noise
                         usually has a frame with nothing at the predicted spot.
      forced_snr_median  typical strength
      forced_snr_std     brightness steadiness across frames
      stack_snr          all frames combined: sum of per-frame SNRs / sqrt(frames).
                         Signal adds up while noise grows only as the square root:
                         the core idea of shift-and-stack.
    """
    size = tracklet_stamps.shape[-1]
    centre = size // 2
    yy, xx = np.indices((size, size))
    aperture = (xx - centre) ** 2 + (yy - centre) ** 2 <= aperture_radius ** 2

    difference = tracklet_stamps[:, :, 2].astype(np.float32)           # channel 2 = difference
    per_frame_snr = difference[:, :, aperture].sum(axis=2) / np.sqrt(aperture.sum())
    frames = per_frame_snr.shape[1]

    return pd.DataFrame({
        "forced_snr_min": per_frame_snr.min(axis=1),
        "forced_snr_median": np.median(per_frame_snr, axis=1),
        "forced_snr_std": per_frame_snr.std(axis=1),
        "stack_snr": per_frame_snr.sum(axis=1) / np.sqrt(frames),
    })
