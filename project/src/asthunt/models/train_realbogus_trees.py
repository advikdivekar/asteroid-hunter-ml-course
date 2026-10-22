"""Train the classical real/bogus classifier: XGBoost on stamp features plus PCA of the stamp pixels."""

import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from asthunt.models.predict_realbogus_trees import core_pixels, tree_inputs

PCA_COMPONENTS = 20

# Chosen on the VALIDATION nights only (docs/classical_realbogus.md): depth 4 scored best of 4 / 5 / 6.
XGB_SETTINGS = {"n_estimators": 600, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}


def train_realbogus_trees(stamps, detections, labels, seed=42):
    """Fit and return the model as a dict: {"pca_mean", "pca_components", "xgb"}.

    stamps     : (n, 3, 31, 31) NORMALIZED stamps (dataset/normalize_stamps.py), the CNN's input
    detections : the matching detection rows (the blob finder's numbers are features too)
    labels     : 1 = real, 0 = bogus (balanced training set, as for the CNN)

    Two kinds of input, both classical ML:
      1. 29 hand-made features (features/compute_stamp_features.py): each a number with a physical meaning
      2. PCA of the central 15 x 15 pixels of all 3 channels: the 20 directions in which stamps differ most.
         PCA is unsupervised (it never sees the labels); it compresses 675 pixels into 20 numbers.
    Then gradient-boosted trees (XGBoost) learn which combinations mean "real".
    """
    pca = PCA(n_components=PCA_COMPONENTS, random_state=seed).fit(core_pixels(stamps))
    model = {"pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32),
             "pca_explained_variance": float(pca.explained_variance_ratio_.sum())}
    xgb = XGBClassifier(**XGB_SETTINGS, random_state=seed, n_jobs=-1, eval_metric="logloss")
    xgb.fit(tree_inputs(model, stamps, detections), np.asarray(labels))
    model["xgb"] = xgb
    return model
