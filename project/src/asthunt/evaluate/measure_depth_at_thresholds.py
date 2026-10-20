"""Depth and false alarms of one model at several tracklet thresholds."""

import numpy as np
import pandas as pd

from asthunt.evaluate.find_magnitude_at_completeness import find_magnitude_at_completeness
from asthunt.evaluate.measure_completeness import measure_completeness


def measure_depth_at_thresholds(night_results, thresholds, magnitude_bins=np.arange(17, 22.25, 0.25)):
    """Return one row per threshold: threshold, mag_at_50_completeness, unexplained_per_night.

    night_results: run_night_pipeline outputs made WITH injections.
    Two models must be compared at the SAME false-alarm level: a stricter
    threshold always trades depth for fewer alarms, so comparing each model
    at its own chosen threshold can make a tie look like a loss.
    """
    rows = []
    for threshold in thresholds:
        planted, unexplained = [], 0
        for result in night_results:
            tracklets = result["tracklets"]
            accepted = tracklets[tracklets["tracklet_score"] >= threshold]
            per_fake = result["injections"].groupby("injection_id").agg(mag=("mag", "first")).reset_index()
            per_fake["found"] = per_fake["injection_id"].isin(set(accepted.loc[accepted["label"] == 1, "injection_id"]))
            planted.append(per_fake)
            unexplained += int(((accepted["known_name"] == "") & ~accepted["contains_fake"].astype(bool)).sum())
        completeness = measure_completeness(pd.concat(planted, ignore_index=True), "found", "mag", magnitude_bins)
        rows.append({"threshold": threshold,
                     "mag_at_50_completeness": find_magnitude_at_completeness(completeness, 0.5),
                     "unexplained_per_night": unexplained / len(night_results)})
    return pd.DataFrame(rows)
