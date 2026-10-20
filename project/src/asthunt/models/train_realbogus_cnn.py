"""Train the real/bogus CNN with augmentation and early stopping."""

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from asthunt.dataset.augment_stamps import augment_stamps
from asthunt.models.define_realbogus_cnn import define_realbogus_cnn
from asthunt.models.predict_realbogus import predict_realbogus


def train_realbogus_cnn(train_stamps, train_labels, validation_stamps, validation_labels, settings, device, seed):
    """Train a fresh CNN and return (best_model, history).

    One epoch = one pass over all training stamps, in random batches:
      1. augment the batch (random rotations / flips),
      2. forward pass: the model scores the batch,
      3. loss: binary cross-entropy between scores and labels,
      4. backward pass: compute how each weight should change,
      5. optimizer step: change the weights a little (Adam).
    After every epoch we score the whole validation set. We keep the model
    from the epoch with the best validation average precision, and stop
    early if it hasn't improved for `patience` epochs (it's starting to
    memorise the training data instead of learning general patterns).

    history: one dict per epoch with train_loss, validation_loss, validation_ap
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = define_realbogus_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    loss_function = nn.BCEWithLogitsLoss()

    best_ap, best_model, epochs_without_improvement = -1.0, None, 0
    history = []

    for epoch in range(settings["max_epochs"]):
        model.train()                                   # dropout on
        order = rng.permutation(len(train_stamps))
        batch_losses = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start + settings["batch_size"]]
            stamps = torch.as_tensor(augment_stamps(train_stamps[rows], rng), device=device)
            labels = torch.as_tensor(train_labels[rows], dtype=torch.float32, device=device)

            optimizer.zero_grad()                       # forget the previous batch's gradients
            loss = loss_function(model(stamps), labels)
            loss.backward()                             # backpropagation
            optimizer.step()                            # update the weights
            batch_losses.append(loss.item())

        validation_scores = predict_realbogus(model, validation_stamps, device)
        validation_ap = average_precision_score(validation_labels, validation_scores)
        clipped = np.clip(validation_scores, 1e-7, 1 - 1e-7)
        validation_loss = -np.mean(validation_labels * np.log(clipped) + (1 - validation_labels) * np.log(1 - clipped))

        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)),
                        "validation_loss": float(validation_loss), "validation_ap": float(validation_ap)})
        print(f"epoch {epoch + 1:2d}: train loss {history[-1]['train_loss']:.4f}  "
              f"validation loss {validation_loss:.4f}  validation AP {validation_ap:.4f}", flush=True)

        if validation_ap > best_ap:
            best_ap, best_model, epochs_without_improvement = validation_ap, copy.deepcopy(model), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings["patience"]:
                print(f"no improvement for {settings['patience']} epochs: stopping early")
                break

    return best_model.cpu(), history
