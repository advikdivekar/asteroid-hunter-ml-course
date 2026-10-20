"""Describe each tracklet with a few numbers a classifier can use."""

import pandas as pd

FEATURE_COLUMNS = [
    "n_detections", "rate_arcsec_per_hour", "max_residual",
    "mag_mean", "mag_std", "snr_mean", "snr_min",
    "score_mean", "score_min", "npix_std", "elongation_mean",
]


def compute_tracklet_features(tracklets, members, detections):
    """Return a copy of `tracklets` with the columns in FEATURE_COLUMNS added.

    Ideas behind the features:
      - motion:      a real asteroid moves steadily (rate, max_residual).
      - brightness:  it keeps nearly the same brightness for 30 minutes,
                     while noise blips vary randomly (mag_std).
      - strength:    weak links make weak tracklets (snr_min).
      - appearance:  what the real/bogus CNN thought of each detection
                     (score_mean, score_min; needs a `score` column).
      - shape:       a point source keeps the same size and roundness
                     (npix_std, elongation_mean = a / b).
    """
    columns = detections[["mag", "snr", "score", "npix", "a", "b"]].copy()
    columns["elongation"] = columns["a"] / columns["b"]
    joined = members.merge(columns, left_on="detection_id", right_index=True)

    grouped = joined.groupby("tracklet_id")
    per_tracklet = pd.DataFrame({
        "mag_mean": grouped["mag"].mean(),
        "mag_std": grouped["mag"].std(),
        "snr_mean": grouped["snr"].mean(),
        "snr_min": grouped["snr"].min(),
        "score_mean": grouped["score"].mean(),
        "score_min": grouped["score"].min(),
        "npix_std": grouped["npix"].std(),
        "elongation_mean": grouped["elongation"].mean(),
    })
    return tracklets.merge(per_tracklet, left_on="tracklet_id", right_index=True)
