"""Find trails in a difference image with a bank of trailed matched filters."""

import numpy as np
import pandas as pd
from scipy import fft
from scipy.ndimage import maximum_filter


def find_streaks(difference, bad_pixel_mask, psf, streak_kernels, threshold, peak_size=7):
    """Return a table of streak detections: x, y (centre), length, angle, snr, point_snr.

    For every kernel (a PSF smeared along a line), the matched-filter SNR map
    tells how strongly each pixel looks like the centre of such a trail.
    We keep, per pixel, the best kernel. The round PSF itself is also tested
    (point_snr): a detection counts as a STREAK only if a trailed kernel beats
    the round one, so ordinary stars and point-like asteroids aren't reported.

    Speed trick: convolution = multiplication in Fourier space. The image's
    Fourier transform is computed once and reused for every kernel.

    Peaks closer to the image edge than half the largest kernel are dropped:
    there the kernel hangs over the edge, and a long kernel lying along the
    border responds to the border itself (found as a false "fast mover" sliding
    along the bottom row of 618/c02/q3 on 2019-01-23).
    """
    good = ~bad_pixel_mask
    image = np.where(good, difference, 0.0)
    values = image[good]
    sigma = 1.4826 * np.median(np.abs(values - np.median(values)))

    height, width = image.shape
    largest = max([k["kernel"].shape[0] for k in streak_kernels] + [psf.shape[0]])
    shape = (height + largest, width + largest)
    image_fft = fft.rfft2(image, shape, workers=-1)

    def snr_map(kernel):
        convolved = fft.irfft2(image_fft * fft.rfft2(kernel, shape, workers=-1), shape, workers=-1)
        half = kernel.shape[0] // 2
        return convolved[half:half + height, half:half + width] / (sigma * np.sqrt(np.sum(kernel ** 2)))

    point_snr = snr_map(psf / psf.sum())
    best = np.full(image.shape, -np.inf)
    best_index = np.zeros(image.shape, dtype=int)
    for index, entry in enumerate(streak_kernels):
        snr = snr_map(entry["kernel"])
        better = snr > best
        best[better], best_index[better] = snr[better], index
    best[bad_pixel_mask] = 0.0

    is_peak = (best == maximum_filter(best, size=peak_size)) & (best > threshold) & (best > point_snr)
    edge = largest // 2
    inside = np.zeros(image.shape, dtype=bool)
    inside[edge:height - edge, edge:width - edge] = True
    is_peak &= inside
    ys, xs = np.nonzero(is_peak)
    return pd.DataFrame({
        "x": xs, "y": ys,
        "length": [streak_kernels[i]["length"] for i in best_index[ys, xs]],
        "angle": [streak_kernels[i]["angle"] for i in best_index[ys, xs]],
        "snr": best[ys, xs],
        "point_snr": point_snr[ys, xs],
    })
