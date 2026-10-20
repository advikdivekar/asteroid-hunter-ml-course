"""Judge every pixel against its own behaviour over the night (a per-pixel z-score over time)."""

import numpy as np


def standardize_over_night(snr_images, good_masks, offsets, row_block=256):
    """Return maps (same shape) where each pixel = (value - its night median) / its night scatter.

    snr_images : (frames, h, w) matched-filter SNR maps, each on its own frame's pixel grid
    good_masks : (frames, h, w) True where usable
    offsets    : (frames, 2) integer (dx, dy): reference pixel + offset = same sky spot on frame k

    Why: difference images are not pure noise.
      - STATIC structure (star halos, galaxies, bleed trails) is the same in
        every frame -> the per-pixel median over the night removes it.
      - VARYING structure (star residuals that change with the seeing) wobbles
        from frame to frame -> the per-pixel scatter over the night is large
        there, so dividing by it turns those wobbles down.
    A moving object sits on a given sky pixel in only one or two frames, so the
    median and the robust scatter (MAD) barely notice it.
    The scatter is floored at 1: quiet pixels are never boosted.
    """
    frames, height, width = snr_images.shape
    pad = int(np.abs(offsets).max()) + 1

    # 1. Every frame on the reference grid (NaN where the frame has no data)
    aligned = np.full((frames, height, width), np.nan, dtype=np.float32)
    for k in range(frames):
        dx, dy = int(offsets[k][0]), int(offsets[k][1])
        padded = np.pad(np.where(good_masks[k], snr_images[k], np.nan), pad, constant_values=np.nan)
        aligned[k] = padded[pad + dy:pad + dy + height, pad + dx:pad + dx + width]

    # 2. Per-pixel median and robust scatter over the night (blocks of rows to limit memory)
    median = np.zeros((height, width), dtype=np.float32)
    scatter = np.ones((height, width), dtype=np.float32)
    for start in range(0, height, row_block):
        block = aligned[:, start:start + row_block]
        enough = np.sum(~np.isnan(block), axis=0) >= 3
        filled = np.where(enough[None], block, 0.0)
        block_median = np.nanmedian(filled, axis=0)
        block_scatter = 1.4826 * np.nanmedian(np.abs(filled - block_median), axis=0)
        median[start:start + row_block] = np.where(enough, block_median, 0.0)
        scatter[start:start + row_block] = np.where(enough, np.maximum(block_scatter, 1.0), 1.0)

    # 3. Standardise each frame with its own pixels' night statistics (moved back to that frame's grid)
    standardized = np.zeros_like(snr_images)
    median_padded, scatter_padded = np.pad(median, pad), np.pad(scatter, pad, constant_values=1.0)
    for k in range(frames):
        dx, dy = int(offsets[k][0]), int(offsets[k][1])
        window = (slice(pad - dy, pad - dy + height), slice(pad - dx, pad - dx + width))
        frame = (snr_images[k] - median_padded[window]) / scatter_padded[window]
        standardized[k] = np.where(good_masks[k], frame, 0.0)
    return standardized.astype(np.float32)
