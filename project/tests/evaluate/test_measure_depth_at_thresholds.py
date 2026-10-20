import numpy as np
import pandas as pd

from asthunt.evaluate.measure_depth_at_thresholds import measure_depth_at_thresholds


def make_night():
    """Fakes at mag 18-22; brighter fakes get higher scores. Plus one false tracklet scored 0.4."""
    mags = np.repeat(np.arange(18.0, 22.0, 0.5), 10)
    injections = pd.DataFrame({"injection_id": range(len(mags)), "mag": mags})
    tracklets = pd.DataFrame({"tracklet_score": np.clip((22 - mags) / 4, 0, 1), "label": 1,
                              "injection_id": range(len(mags)), "known_name": "", "contains_fake": True})
    false = pd.DataFrame({"tracklet_score": [0.4], "label": [0], "injection_id": [-1], "known_name": [""], "contains_fake": [False]})
    return {"injections": injections, "tracklets": pd.concat([tracklets, false], ignore_index=True)}


def test_stricter_threshold_means_shallower_and_fewer_alarms():
    table = measure_depth_at_thresholds([make_night()], [0.3, 0.6])
    loose, strict = table.iloc[0], table.iloc[1]
    assert loose["unexplained_per_night"] == 1 and strict["unexplained_per_night"] == 0
    assert strict["mag_at_50_completeness"] < loose["mag_at_50_completeness"]
