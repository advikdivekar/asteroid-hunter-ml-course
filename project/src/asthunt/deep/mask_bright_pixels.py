"""Hide everything single frames can already see, so the deep search only stacks faint signal."""

import numpy as np
from scipy.ndimage import binary_dilation


def mask_bright_pixels(snr_images, good_masks, sigma, radius_pixels):
    """Return (snr_images, good_masks) with bright pixels (|SNR| > sigma) and a border around them removed.

    Why: a bright asteroid also lies partly along WRONG trial motions, so it
    creates "echo" peaks offset by a few to tens of pixels: ghost tracklets
    that move just like it. (On a real night, 151 of 242 unexplained deep
    tracklets were within 30 px of a known asteroid; random positions would be ~190 px away.)
    Bright objects are found by the regular single-frame pipeline anyway; the
    deep search exists for signals too faint to show up in any single frame.
    """
    snr_images = snr_images.copy()
    good_masks = good_masks.copy()
    structure = np.ones((2 * radius_pixels + 1, 2 * radius_pixels + 1), dtype=bool)
    for k in range(len(snr_images)):
        bright = binary_dilation(np.abs(snr_images[k]) > sigma, structure=structure)
        snr_images[k][bright] = 0.0
        good_masks[k] &= ~bright
    return snr_images, good_masks
