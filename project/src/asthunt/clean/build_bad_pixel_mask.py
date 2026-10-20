"""Mark pixels that must never be trusted (True = bad)."""

import numpy as np


def build_bad_pixel_mask(image, saturation_level, ztf_mask=None, bad_bits=(), grow_saturated_by=2):
    """Return a boolean array, True where a pixel is bad.

    A pixel is bad if:
      - it is NaN (no data),
      - it is at or above the saturation level (the detector was full, so
        the true brightness is unknown). We also mark a small border of
        `grow_saturated_by` pixels around saturated areas, because bright
        stars bleed into their neighbours.
      - its ZTF mask value has any of the `bad_bits` switched on.
        ZTF's mask files store several yes/no flags per pixel as bits;
        their meaning is listed in the mask file's own header.
    """
    bad = np.isnan(image)

    saturated = image >= saturation_level
    saturated = _grow(saturated, grow_saturated_by)
    bad = bad | saturated

    if ztf_mask is not None and len(bad_bits) > 0:
        bad_bit_pattern = sum(1 << bit for bit in bad_bits)   # e.g. bits (0, 4) -> 0b10001 = 17
        bad = bad | ((ztf_mask.astype(np.int64) & bad_bit_pattern) != 0)

    return bad


def _grow(mask, pixels):
    """Expand True areas of a mask by `pixels` in every direction."""
    grown = mask.copy()
    for _ in range(pixels):
        neighbour = grown.copy()
        neighbour[1:, :] |= grown[:-1, :]
        neighbour[:-1, :] |= grown[1:, :]
        neighbour[:, 1:] |= grown[:, :-1]
        neighbour[:, :-1] |= grown[:, 1:]
        grown = neighbour
    return grown
