import numpy as np
import pandas as pd

from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS, compute_tracklet_features


def test_steady_tracklet_has_small_brightness_scatter():
    detections = pd.DataFrame({
        "mag":   [19.0, 19.0, 19.1, 18.9, 20.0, 18.0, 21.0, 19.0],
        "snr":   [10, 10, 9, 11, 3, 8, 2, 6],
        "score": [0.9, 0.8, 0.9, 0.95, 0.2, 0.5, 0.1, 0.3],
        "npix":  [12, 12, 13, 12, 5, 20, 6, 9],
        "a":     [1.2, 1.2, 1.2, 1.2, 2.0, 1.0, 3.0, 1.0],
        "b":     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    tracklets = pd.DataFrame({"tracklet_id": [0, 1], "n_detections": [4, 4],
                              "rate_arcsec_per_hour": [30.0, 50.0], "max_residual": [0.3, 1.2]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 0, 1, 1, 1, 1], "detection_id": list(range(8))})

    features = compute_tracklet_features(tracklets, members, detections)

    assert set(FEATURE_COLUMNS) <= set(features.columns)
    steady, noisy = features.iloc[0], features.iloc[1]
    assert steady["mag_std"] < noisy["mag_std"]
    assert np.isclose(steady["score_min"], 0.8)
