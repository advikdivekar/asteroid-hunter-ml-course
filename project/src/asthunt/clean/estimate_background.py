"""Estimate the smooth sky level and its noise across an image."""

import numpy as np
import sep


def estimate_background(image, bad_pixel_mask=None, box_size=64):
    """Return (background_map, noise_map), both the same shape as the image.

    The sky is not perfectly flat: moonlight, clouds and scattered light make
    it vary slowly across the chip. SEP splits the image into boxes of
    `box_size` pixels, measures the typical sky level in each box (ignoring
    stars), and interpolates a smooth map. `noise_map` is the scatter
    (standard deviation) of the sky in each place, i.e. "1 sigma".

    Bad pixels are ignored so saturated stars don't bias the sky estimate.
    """
    data = np.ascontiguousarray(image, dtype=np.float32)   # SEP needs this memory layout
    if bad_pixel_mask is not None:
        bad_pixel_mask = np.ascontiguousarray(bad_pixel_mask, dtype=bool)

    background = sep.Background(data, mask=bad_pixel_mask, bw=box_size, bh=box_size)
    return background.back(), background.rms()
