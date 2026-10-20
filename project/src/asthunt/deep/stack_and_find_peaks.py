"""Shift-and-stack SNR maps along trial motions and keep the peaks."""

import numpy as np
import pandas as pd
import torch


def stack_and_find_peaks(snr_images, good_masks, hours, offsets, velocities, threshold, device="cpu", peak_size=5,
                         clip_sigma=None):
    """Return a table of peaks: x, y (reference grid, at hours = 0), vx, vy (px/h), snr, coverage.

    snr_images : (frames, h, w) float32 matched-filter SNR maps (noise ~ N(0, 1))
    good_masks : (frames, h, w) bool, True where a frame has usable pixels
    hours      : (frames,) time of each frame in hours since the reference time
    offsets    : (frames, 2) integer (dx, dy) pointing shift of each frame vs the reference grid
    velocities : (n, 2) trial motions in pixels/hour (make_velocity_grid)

    For every trial motion: move each frame's map back along the motion,
    add them, divide by sqrt(number of frames covering each pixel). An object
    really moving at that velocity piles up in one spot; noise doesn't. Pure
    noise stays ~N(0, 1) after stacking, so `threshold` is in sigma.

    We keep, for every pixel, the BEST stacked SNR over all trial motions (and
    which motion gave it), and look for peaks once at the end: local maxima of
    that best-SNR image above `threshold`, covered by at least half the frames.
    (Finding peaks after every motion would mean thousands of slow GPU->CPU trips.)

    clip_sigma: robust stacking. Each frame's SNR is clipped to [-clip, +clip]
    before adding. Real difference images contain bright artifacts (star
    residuals, bleed edges): a 50-sigma blip in ONE frame would otherwise add
    50/sqrt(8) = 18 sigma to every trial motion passing through it. Faint
    targets (1-4 sigma per frame) are untouched by a clip at ~5.
    """
    frames, height, width = snr_images.shape
    max_shift = int(np.ceil(np.abs(velocities).max() * np.abs(hours).max() + np.abs(offsets).max())) + 2
    pad = (max_shift, max_shift, max_shift, max_shift)
    snr = torch.as_tensor(snr_images, device=device)
    if clip_sigma is not None:
        snr = snr.clamp(-clip_sigma, clip_sigma)
    snr = torch.nn.functional.pad(snr, pad)
    good = torch.nn.functional.pad(torch.as_tensor(good_masks.astype(np.float32), device=device), pad)

    best = torch.full((height, width), -1e9, device=device)
    best_velocity = torch.zeros((height, width), dtype=torch.long, device=device)
    best_coverage = torch.zeros((height, width), device=device)
    for index, (vx, vy) in enumerate(velocities):
        total = torch.zeros((height, width), device=device)
        coverage = torch.zeros((height, width), device=device)
        for k in range(frames):
            dx = int(round(vx * hours[k])) + int(offsets[k][0])
            dy = int(round(vy * hours[k])) + int(offsets[k][1])
            window = (k, slice(max_shift + dy, max_shift + dy + height), slice(max_shift + dx, max_shift + dx + width))
            total += snr[window]
            coverage += good[window]
        stacked = total / torch.sqrt(coverage.clamp(min=1.0))
        better = stacked > best
        best = torch.where(better, stacked, best)
        best_velocity = torch.where(better, torch.full_like(best_velocity, index), best_velocity)
        best_coverage = torch.where(better, coverage, best_coverage)

    local_max = torch.nn.functional.max_pool2d(best[None, None], peak_size, stride=1, padding=peak_size // 2)[0, 0]
    is_peak = (best == local_max) & (best > threshold) & (best_coverage >= frames / 2)
    ys, xs = torch.nonzero(is_peak, as_tuple=True)
    if len(xs) == 0:
        return pd.DataFrame(columns=["x", "y", "vx", "vy", "snr", "coverage"])

    chosen = velocities[best_velocity[ys, xs].cpu().numpy()]
    return pd.DataFrame({"x": xs.cpu().numpy(), "y": ys.cpu().numpy(), "vx": chosen[:, 0], "vy": chosen[:, 1],
                         "snr": best[ys, xs].cpu().numpy(), "coverage": best_coverage[ys, xs].cpu().numpy()})
