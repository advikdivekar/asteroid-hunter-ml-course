"""A network that scores a whole tracklet: the stamps from every frame plus the tracklet's numbers."""

import torch
from torch import nn


class MultiFrameModel(nn.Module):
    """Input:  stamps   (batch, frames, 3, 21, 21)  science / reference / difference along the predicted path
               features (batch, n_features)         tracklet numbers (speed, straightness, brightness scatter, ...)
       Output: one logit per tracklet (sigmoid -> probability it is a real moving object)

    How it works:
      1. ONE small CNN (the "encoder") turns each frame's stamp into 32 numbers.
      2. The SAME encoder also looks at the STACK: the average of all frames'
         stamps. Along the right path a faint asteroid adds up while noise
         averages out (a learned mini shift-and-stack).
      3. Per-frame results are summarised by their mean and their maximum, so
         the model works for any number of frames and doesn't care about order.
      4. A small dense head combines [mean, max, stack, features] into one score.
    Features are standardised inside the model with the training set's mean and
    standard deviation, stored with the weights, so they can never get lost.
    """

    def __init__(self, n_features, feature_mean, feature_std, stamp_size=21):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 21 -> 10
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 10 -> 5
            nn.Flatten(),
            nn.Linear(32 * (stamp_size // 4) ** 2, 32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 3 + n_features, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))

    def forward(self, stamps, features):
        batch, frames = stamps.shape[:2]
        per_frame = self.encoder(stamps.reshape(batch * frames, *stamps.shape[2:])).reshape(batch, frames, -1)
        stacked = self.encoder(stamps.mean(dim=1))
        features = (features - self.feature_mean) / self.feature_std
        combined = torch.cat([per_frame.mean(dim=1), per_frame.max(dim=1).values, stacked, features], dim=1)
        return self.head(combined).squeeze(1)


def define_multiframe_model(n_features, feature_mean, feature_std):
    """Return an untrained MultiFrameModel (see the class above)."""
    return MultiFrameModel(n_features, feature_mean, feature_std)
