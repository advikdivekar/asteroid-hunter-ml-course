"""Turn a difference image into a map of 'how many sigma of point-source signal is here'."""

import numpy as np
from scipy.signal import fftconvolve


def build_snr_image(difference, bad_pixel_mask, psf):
    """Return a float32 image where each pixel is the signal-to-noise of a point source centred there.

    This is a MATCHED FILTER: convolving with the PSF is the best linear way to
    find a known shape (a star-like blob) in noise. Each output pixel is a
    weighted sum of its neighbours, weighted by the PSF.

    Scaling: if the image noise is sigma per pixel, the filtered image's noise
    is sigma * sqrt(sum(psf^2)). Dividing by that makes pure noise come out as
    standard normal (mean 0, std 1), on every night. That's what lets us simply
    ADD maps from different frames later: signal adds up, noise grows as sqrt(N).

    Bad pixels are set to 0 before filtering (no signal, no noise) and the
    output is 0 there too.
    """
    good = ~bad_pixel_mask
    image = np.where(good, difference, 0.0)
    values = image[good]
    sigma = 1.4826 * np.median(np.abs(values - np.median(values)))     # robust noise per pixel

    filtered = fftconvolve(image, psf, mode="same")
    snr = filtered / (sigma * np.sqrt(np.sum(psf ** 2)))
    snr[bad_pixel_mask] = 0.0
    return snr.astype(np.float32)
