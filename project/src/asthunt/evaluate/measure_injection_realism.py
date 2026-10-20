"""Can a classifier tell our planted fakes from real asteroids? (It shouldn't.)"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REALISM_FEATURES = ["snr", "npix", "a", "b", "elongation", "concentration"]


def measure_injection_realism(detections, mag_bins, seed=0):
    """Return (auc, importances): how well fakes can be told apart from real known asteroids.

    detections : labelled detections from several nights (needs injection_id,
                 known_asteroid, mag, night and the shape columns)

    This is "adversarial validation":
      - class 1 = detections of real known asteroids,
      - class 0 = detections of planted fakes, sampled to have the SAME
        magnitude distribution (otherwise the model just learns "brighter").
    A classifier is cross-validated by night. AUC ~0.5 means it can't tell them
    apart: our fakes look like real asteroids. AUC well above 0.5 means
    something gives the fakes away, and `importances` says what.
    """
    rng = np.random.default_rng(seed)
    table = detections.assign(elongation=detections["a"] / detections["b"],
                              concentration=detections["peak"] / detections["flux"].clip(lower=1e-6))
    real = table[table["known_asteroid"] != ""]
    fakes = table[table["injection_id"] >= 0]

    # Same number of fakes as real ones in every magnitude bin
    sampled = []
    real_bins = pd.cut(real["mag"], mag_bins)
    fake_bins = pd.cut(fakes["mag"], mag_bins)
    for interval, count in real_bins.value_counts().items():
        pool = fakes[fake_bins == interval]
        if count and len(pool):
            sampled.append(pool.iloc[rng.choice(len(pool), size=min(count, len(pool)), replace=False)])
    fakes = pd.concat(sampled)

    both = pd.concat([real.assign(is_real=1), fakes.assign(is_real=0)], ignore_index=True)
    features, labels, groups = both[REALISM_FEATURES].to_numpy(), both["is_real"].to_numpy(), both["night"].to_numpy()

    scores = np.zeros(len(both))
    folds = GroupKFold(n_splits=min(5, len(set(groups))))
    for fit_rows, test_rows in folds.split(features, labels, groups):
        model = GradientBoostingClassifier(random_state=seed).fit(features[fit_rows], labels[fit_rows])
        scores[test_rows] = model.predict_proba(features[test_rows])[:, 1]

    final = GradientBoostingClassifier(random_state=seed).fit(features, labels)
    importances = pd.Series(final.feature_importances_, index=REALISM_FEATURES).sort_values(ascending=False)
    return roc_auc_score(labels, scores), importances
