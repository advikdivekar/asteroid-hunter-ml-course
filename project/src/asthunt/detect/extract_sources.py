"""Find candidate sources (positive blobs) in a difference image."""

import numpy as np
import pandas as pd
import sep

from asthunt.clean.estimate_background import estimate_background


def extract_sources(difference_image, bad_pixel_mask, threshold_sigma=2.0, min_area_pixels=5):
    """Return a table with one row per detected blob.

    In a difference image (science - reference) stars cancel out, so what
    remains is either something new (an asteroid, a variable star, a
    supernova) or an artifact (bad subtraction, cosmic ray, ghost).
    We keep only POSITIVE blobs: a moving asteroid is in the science
    image but not in the reference.

    Steps:
      1. Mark NaN pixels as bad and set them to 0 (SEP can't handle NaN).
      2. Estimate the remaining background and noise.
      3. SEP smooths the image with a small kernel (which boosts faint,
         star-shaped signals) and finds groups of at least
         `min_area_pixels` connected pixels above threshold_sigma x noise.
      4. Measure flux in a small circle and turn it into a signal-to-noise ratio.

    Columns: x, y (pixels, 0-indexed), flux, flux_error, snr, peak,
             a, b (blob shape in pixels), npix (number of pixels).
    """
    bad = bad_pixel_mask | np.isnan(difference_image)
    image = np.where(bad, 0.0, difference_image)

    background, noise = estimate_background(image, bad)
    image = np.ascontiguousarray(image - background, dtype=np.float32)
    noise = np.ascontiguousarray(noise, dtype=np.float32)

    sources = sep.extract(image, thresh=threshold_sigma, err=noise, mask=bad, minarea=min_area_pixels)

    # Aperture photometry: sum the flux inside a 3-pixel radius circle.
    flux, flux_error, _ = sep.sum_circle(image, sources["x"], sources["y"], 3.0, err=noise, mask=bad)

    table = pd.DataFrame({
        "x": sources["x"],
        "y": sources["y"],
        "flux": flux,
        "flux_error": flux_error,
        "peak": sources["peak"],
        "a": sources["a"],
        "b": sources["b"],
        "npix": sources["npix"],
    })
    # If every pixel in the aperture is masked, the flux can't be measured (NaN).
    # Record it as "no measurable flux" so later steps never see NaN.
    unmeasured = ~np.isfinite(table["flux"]) | ~np.isfinite(table["flux_error"])
    table.loc[unmeasured, "flux"] = 0.0
    table.loc[unmeasured, "flux_error"] = table.loc[~unmeasured, "flux_error"].median()
    table["snr"] = table["flux"] / table["flux_error"]
    return table
