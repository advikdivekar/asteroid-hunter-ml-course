"""Score tracklets with a trained multi-frame model."""

import numpy as np
import torch


def predict_multiframe(model, stamps, features, device="cpu", batch_size=512):
    """Return one probability (0..1) per tracklet. Stamps must already be normalized."""
    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():
        for start in range(0, len(stamps), batch_size):
            batch_stamps = torch.as_tensor(np.asarray(stamps[start:start + batch_size], dtype=np.float32), device=device)
            batch_features = torch.as_tensor(np.asarray(features[start:start + batch_size], dtype=np.float32), device=device)
            scores.append(torch.sigmoid(model(batch_stamps, batch_features)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)
