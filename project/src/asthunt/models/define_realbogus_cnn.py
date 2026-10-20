"""A small convolutional neural network that scores a stamp triplet as real or bogus."""

from torch import nn


def define_realbogus_cnn():
    """Return an untrained CNN: input (batch, 3, 31, 31) -> output (batch,) logits.

    A logit is a raw score; sigmoid(logit) turns it into a probability.

    Layout (sizes for a 31x31 stamp):
      block 1: two 3x3 convolutions, 16 filters  -> 31x31, then max-pool -> 15x15
      block 2: two 3x3 convolutions, 32 filters  -> 15x15, then max-pool -> 7x7
      block 3: one 3x3 convolution,  64 filters  ->  7x7,  then max-pool -> 3x3
      head   : flatten (64*3*3 = 576) -> dropout -> 64 neurons -> 1 output

    Convolutions look for small local patterns (a round blob, a sharp spike,
    a dipole); pooling halves the size so deeper layers see larger areas.
    Dropout randomly switches off neurons during training, so the network
    can't rely on any single one (less overfitting).
    """
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64 * 3 * 3, 64), nn.ReLU(),
        nn.Linear(64, 1),
        nn.Flatten(start_dim=0),          # (batch, 1) -> (batch,)
    )
