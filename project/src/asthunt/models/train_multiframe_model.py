"""Train the multi-frame tracklet model with augmentation, class weighting and early stopping."""

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from asthunt.dataset.augment_stamps import augment_stamps
from asthunt.models.define_multiframe_model import define_multiframe_model
from asthunt.models.predict_multiframe import predict_multiframe


def train_multiframe_model(train, validation, settings, device, seed):
    """Train a fresh model and return (best_model, history).

    train, validation : dicts with "stamps" (n, frames, 3, h, w), "features" (n, k), "labels" (n,)

    Same recipe as the v1 CNN (augment -> forward -> loss -> backward -> Adam;
    keep the epoch with the best validation average precision; stop early),
    with one difference: instead of throwing bogus examples away to balance
    the classes, every real example counts `pos_weight` times in the loss
    (pos_weight = bogus / real in the training data). No data is wasted.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    features = np.nan_to_num(train["features"]).astype(np.float32)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-6

    model = define_multiframe_model(features.shape[1], feature_mean, feature_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    labels = train["labels"].astype(np.float32)
    # float32: the Apple GPU (MPS) has no 64-bit floats
    pos_weight = torch.tensor(float((labels == 0).sum() / max((labels == 1).sum(), 1)), dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    validation_features = np.nan_to_num(validation["features"]).astype(np.float32)

    best_ap, best_model, epochs_without_improvement, history = -1.0, None, 0, []
    for epoch in range(settings["max_epochs"]):
        model.train()
        order = rng.permutation(len(labels))
        batch_losses = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            stamps = torch.as_tensor(augment_stamps(train["stamps"][rows].astype(np.float32), rng), device=device)
            optimizer.zero_grad()
            loss = loss_function(model(stamps, torch.as_tensor(features[rows], device=device)),
                                 torch.as_tensor(labels[rows], device=device))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        scores = predict_multiframe(model, validation["stamps"], validation_features, device)
        validation_ap = average_precision_score(validation["labels"], scores)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)), "validation_ap": float(validation_ap)})
        print(f"epoch {epoch + 1:2d}: train loss {history[-1]['train_loss']:.4f}  validation AP {validation_ap:.4f}", flush=True)

        if validation_ap > best_ap:
            best_ap, best_model, epochs_without_improvement = validation_ap, copy.deepcopy(model), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings["patience"]:
                print(f"no improvement for {settings['patience']} epochs: stopping early")
                break

    return best_model.cpu(), history
