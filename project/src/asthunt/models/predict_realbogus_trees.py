"""Score stamps with the classical real/bogus classifier (models/train_realbogus_trees.py)."""

import numpy as np

from asthunt.features.compute_stamp_features import compute_stamp_features


def core_pixels(stamps):
    """The central 15 x 15 pixels of all three channels, flattened: what PCA looks at (675 numbers per stamp)."""
    stamps = np.asarray(stamps, dtype=np.float32)
    return stamps[:, :, 8:23, 8:23].reshape(len(stamps), -1)


def tree_inputs(model, stamps, detections):
    """The table the trees read: 29 features, then the stamp's position along the 20 PCA directions."""
    features = compute_stamp_features(stamps, detections).to_numpy()
    projected = (core_pixels(stamps) - model["pca_mean"]) @ model["pca_components"].T
    return np.hstack([features, projected]).astype(np.float32)


def predict_realbogus_trees(model, stamps, detections, batch_size=20000):
    """Return probabilities (0..1) that each stamp is real. stamps must be normalized, as for training."""
    scores = [model["xgb"].predict_proba(tree_inputs(model, stamps[start:start + batch_size],
                                                     detections.iloc[start:start + batch_size]))[:, 1]
              for start in range(0, len(stamps), batch_size)]
    return np.concatenate(scores) if scores else np.zeros(0)
