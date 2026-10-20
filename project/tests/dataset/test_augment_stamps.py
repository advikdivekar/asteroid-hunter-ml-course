import numpy as np

from asthunt.dataset.augment_stamps import augment_stamps


def test_keeps_shape_and_pixel_values():
    stamps = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    augmented = augment_stamps(stamps, np.random.default_rng(0))

    assert augmented.shape == stamps.shape
    for original, changed in zip(stamps, augmented):
        assert sorted(original.ravel()) == sorted(changed.ravel())   # same pixels, new arrangement


def test_all_frames_of_a_tracklet_turn_together():
    stamps = np.zeros((1, 3, 1, 5, 5))
    for frame in range(3):
        stamps[0, frame, 0, 0, frame] = 1.0          # a dot moving right along the bottom row
    turned = augment_stamps(stamps, np.random.default_rng(1))[0, :, 0]
    positions = [np.argwhere(turned[frame] == 1.0)[0] for frame in range(3)]
    steps = {tuple(positions[1] - positions[0]), tuple(positions[2] - positions[1])}
    assert len(steps) == 1                            # still a straight line at constant speed
