import numpy as np

from asthunt.detect.cut_triplet_stamps import cut_triplet_stamps


def test_stamp_is_centred_and_has_three_channels():
    science = np.zeros((50, 50)); science[20, 30] = 1.0
    reference = np.zeros((50, 50))
    difference = science - reference

    stamps = cut_triplet_stamps(science, reference, difference, x_positions=[30], y_positions=[20], size=11)

    assert stamps.shape == (1, 3, 11, 11)
    assert stamps[0, 0, 5, 5] == 1.0      # centre pixel of the science stamp
    assert stamps[0, 2, 5, 5] == 1.0      # and of the difference stamp


def test_edges_are_padded_with_zero():
    image = np.ones((20, 20))
    stamps = cut_triplet_stamps(image, image, image, x_positions=[0], y_positions=[0], size=5)
    assert stamps[0, 0].sum() == 9        # only a 3x3 corner overlaps the image
