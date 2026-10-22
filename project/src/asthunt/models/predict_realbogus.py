"""Score stamps with a trained real/bogus model: the classical trees (v6, the default) or the CNN (v1-v5, kept for comparison)."""

import numpy as np
import torch

from asthunt.models.predict_realbogus_trees import predict_realbogus_trees


def predict_realbogus(model, stamps, device="cpu", batch_size=2048, detections=None):
    """Return a numpy array of probabilities (0..1), one per stamp.

    stamps must already be normalized (see dataset/normalize_stamps.py).
    detections: the matching detection rows; the classical model needs them (the blob finder's
    numbers are among its features), the CNN ignores them.

    For the CNN, the model is switched to evaluation mode: dropout is turned off, so the
    same stamp always gets the same score.
    """
    if isinstance(model, dict):                        # classical model: models/train_realbogus_trees.py
        if detections is None:
            raise ValueError("the classical real/bogus model needs the detections table as well as the stamps")
        return predict_realbogus_trees(model, stamps, detections)

    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():                      # no training here, so no gradients needed
        for start in range(0, len(stamps), batch_size):
            batch = torch.as_tensor(np.asarray(stamps[start:start + batch_size], dtype=np.float32), device=device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)
