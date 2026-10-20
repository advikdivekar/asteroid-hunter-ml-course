"""Make extra training examples by rotating and flipping stamps."""

import numpy as np


def augment_stamps(stamps, rng):
    """Return a copy where each stamp is randomly rotated (0/90/180/270 deg) and maybe mirrored.

    An asteroid seen upside down is still an asteroid, and the sky has no
    preferred direction. Showing the model random orientations teaches it
    exactly that, and gives it "new" examples for free.
    Use only on TRAINING data: validation and test must stay untouched.

    Works for single stamps (n, channels, h, w) and for tracklets
    (n, frames, channels, h, w): the rotation acts on the last two axes, so
    every frame of one tracklet is turned the same way and the motion stays a
    straight line.
    """
    augmented = np.empty_like(stamps)
    for i in range(len(stamps)):
        turned = np.rot90(stamps[i], k=rng.integers(4), axes=(-2, -1))
        if rng.random() < 0.5:
            turned = turned[..., ::-1]
        augmented[i] = turned
    return augmented
