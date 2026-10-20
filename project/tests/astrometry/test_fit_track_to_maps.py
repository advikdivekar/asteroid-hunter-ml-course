import numpy as np
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from asthunt.astrometry.fit_track_to_maps import fit_track_to_maps

SIZE, PIXEL = 220, 1.0                                       # arcsec per pixel


def make_wcs(crpix_x, crpix_y):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [125.0, 20.0]
    wcs.wcs.crpix = [crpix_x, crpix_y]
    wcs.wcs.cdelt = [-PIXEL / 3600, PIXEL / 3600]
    return wcs


def synthetic_night(rng, amplitude, frames=20, v_ra=-38.0, v_dec=6.0, ra0=125.0, dec0=20.0):
    """Unit-variance correlated noise plus a blob that follows a known sky track, with pointing jitter between frames."""
    hours = (np.arange(frames) - (frames - 1) / 2) * 4 / 60
    maps, wcs_list = [], []
    for h in hours:
        wcs = make_wcs(110 + rng.uniform(-3, 3), 110 + rng.uniform(-3, 3))       # a different pointing every frame
        noise = gaussian_filter(rng.normal(size=(SIZE, SIZE)), 1.2)
        noise /= noise.std()
        x, y = wcs.world_to_pixel_values(ra0 + v_ra * h / 3600 / np.cos(np.radians(dec0)), dec0 + v_dec * h / 3600)
        yy, xx = np.indices(noise.shape)
        maps.append((noise + amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2))).astype(np.float32))
        wcs_list.append(wcs)
    return np.array(maps), wcs_list, 2458511.8 + hours / 24


def test_the_fit_recovers_a_faint_track_through_changing_pointings():
    rng = np.random.default_rng(1)
    maps, wcs_list, times = synthetic_night(rng, amplitude=2.5)
    start = {"jd": times.mean(), "ra": 125.0 + 2.0 / 3600, "dec": 20.0 - 3.0 / 3600, "v_ra": -36.0, "v_dec": 7.0}   # a poor first guess

    fit = fit_track_to_maps(maps, wcs_list, times, start)

    sigma = np.sqrt(np.diag(fit["cov"]))
    assert abs(fit["ra"] - 125.0) * np.cos(np.radians(20)) * 3600 < 4 * sigma[0] + 0.3
    assert abs(fit["dec"] - 20.0) * 3600 < 4 * sigma[1] + 0.3
    assert abs(fit["v_ra"] + 38.0) < 4 * sigma[2] + 0.3 and abs(fit["v_dec"] - 6.0) < 4 * sigma[3] + 0.3
    assert fit["snr"] > 8 and fit["frames"] == 20


def test_the_reported_uncertainty_is_about_right():
    z = []
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        maps, wcs_list, times = synthetic_night(rng, amplitude=2.5, frames=16)
        start = {"jd": times.mean(), "ra": 125.0, "dec": 20.0, "v_ra": -38.0, "v_dec": 6.0}
        fit = fit_track_to_maps(maps, wcs_list, times, start)
        sigma = np.sqrt(np.diag(fit["cov"]))
        z += [(fit["ra"] - 125.0) * np.cos(np.radians(20)) * 3600 / sigma[0], (fit["dec"] - 20.0) * 3600 / sigma[1],
              (fit["v_ra"] + 38.0) / sigma[2], (fit["v_dec"] - 6.0) / sigma[3]]
    rms = np.sqrt(np.mean(np.square(z)))
    assert 0.5 < rms < 1.5                                    # honest errors: z-scores of about 1


def test_a_bright_source_gets_a_correspondingly_tiny_uncertainty():
    rng = np.random.default_rng(7)
    maps, wcs_list, times = synthetic_night(rng, amplitude=30.0)
    start = {"jd": times.mean(), "ra": 125.0, "dec": 20.0, "v_ra": -38.0, "v_dec": 6.0}

    fit = fit_track_to_maps(maps, wcs_list, times, start, clip_sigma=1e6)      # no clipping: it would flatten the peak

    sigma = np.sqrt(np.diag(fit["cov"]))
    assert sigma[0] < 0.02 and sigma[1] < 0.02                  # hundredths of an arcsec
    assert abs(fit["ra"] - 125.0) * np.cos(np.radians(20)) * 3600 < 5 * sigma[0]


def test_too_few_frames_inside_the_image_is_an_error():
    rng = np.random.default_rng(2)
    maps, wcs_list, times = synthetic_night(rng, amplitude=2.5, frames=2)
    start = {"jd": times.mean(), "ra": 125.0, "dec": 20.0, "v_ra": -38.0, "v_dec": 6.0}
    try:
        fit_track_to_maps(maps, wcs_list, times, start)
    except ValueError as error:
        assert "not enough" in str(error)
    else:
        raise AssertionError("expected a ValueError")
