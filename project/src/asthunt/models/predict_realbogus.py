"""Score stamps with a trained real/bogus CNN."""

import numpy as np
import torch


def predict_realbogus(model, stamps, device="cpu", batch_size=2048):
    """Return a numpy array of probabilities (0..1), one per stamp.

    stamps must already be normalized (see dataset/normalize_stamps.py).
    The model is switched to evaluation mode: dropout is turned off, so the
    same stamp always gets the same score.
    """
    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():                      # no training here, so no gradients needed
        for start in range(0, len(stamps), batch_size):
            batch = torch.as_tensor(np.asarray(stamps[start:start + batch_size], dtype=np.float32), device=device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)
