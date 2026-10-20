import numpy as np

from asthunt.detect.extract_sources import extract_sources


def add_gaussian_blob(image, x, y, total_flux, sigma=1.0):
    yy, xx = np.indices(image.shape)
    blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    image += total_flux * blob / blob.sum()


def test_finds_injected_blob_and_ignores_negative_one():
    rng = np.random.default_rng(1)
    image = rng.normal(0, 5, size=(200, 200))
    add_gaussian_blob(image, 60.3, 140.7, total_flux=2000)    # bright positive source
    add_gaussian_blob(image, 150, 50, total_flux=-2000)       # negative: must be ignored
    bad = np.zeros(image.shape, dtype=bool)

    sources = extract_sources(image, bad, threshold_sigma=3)

    distance = np.hypot(sources["x"] - 60.3, sources["y"] - 140.7)
    assert distance.min() < 0.5
    assert sources.loc[distance.idxmin(), "snr"] > 20
    assert (np.hypot(sources["x"] - 150, sources["y"] - 50) > 5).all()


def test_nan_pixels_do_not_crash():
    image = np.zeros((100, 100))
    image[10:20, 10:20] = np.nan
    sources = extract_sources(image, np.zeros(image.shape, dtype=bool))
    assert len(sources) == 0
