"""Fit a moving object's sky position and velocity to a night's matched-filter maps."""

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import minimize

WINDOW = 40                      # pixels kept around the track in every frame: far more than the search ever moves


def fit_track_to_maps(snr_maps, wcs_list, times_jd, start, clip_sigma=5.0, search_arcsec=12.0,
                      search_arcsec_per_hour=5.0, coarse_step=1.0):
    """Return the object's attributable at start["jd"] with its uncertainty.

    snr_maps : (frames, h, w) matched-filter maps (unit-variance noise), each on its OWN frame's pixels
    wcs_list : the WCS of every frame: the track is followed in sky coordinates through each of them,
               so frame-to-frame pointing changes need no rounding to whole pixels
    times_jd : mid-exposure time of every frame
    start    : {"jd", "ra", "dec", "v_ra", "v_dec"}: a first guess (v_ra includes cos dec, arcsec/hour)

    How: with the source's flux free and a known PSF shape, the log-likelihood of a track is S^2 / 2,
    where S is the stacked SNR along it (sum of each frame's map at the track's pixel / sqrt(frames)).
      1. S on a coarse grid of position and velocity offsets (linear interpolation: fast), take the best;
      2. polish it by maximising S with smooth (cubic spline) interpolation, so the maximum is not
         pinned to a pixel grid;
      3. the uncertainty is the inverse curvature (Hessian) of -S^2 / 2 at the maximum, measured with
         steps matched to the answer's own size, so a 5-sigma source and a 500-sigma source both work.

    Returns {"jd", "ra", "dec", "v_ra", "v_dec", "cov", "snr", "frames", "kept"} (kept: indices of the frames
    that cover the track and were used); cov is 4x4 in
    (arcsec, arcsec, arcsec/hour, arcsec/hour) for (RA*cos dec, Dec, v_ra, v_dec). Position and
    velocity are nearly uncorrelated when start["jd"] is the mean frame time.
    """
    times_jd = np.asarray(times_jd, dtype=float)
    hours = (times_jd - start["jd"]) * 24
    cos_dec = np.cos(np.radians(start["dec"]))
    windows, centres, jacobians, kept = _prepare_frames(snr_maps, wcs_list, hours, start, cos_dec, clip_sigma)
    if len(windows) < 3:
        raise ValueError(f"only {len(windows)} frames cover the track: not enough to fit")
    splines = [spline_filter(window, order=3) for window in windows]
    kept_hours = hours[kept]

    def stacked_snr(offsets, order):
        """S for every row of offsets (d_ra*cos, d_dec, d_v_ra, d_v_dec)."""
        offsets = np.atleast_2d(offsets)
        total = np.zeros(len(offsets))
        for window, spline, centre, jacobian, h in zip(windows, splines, centres, jacobians, kept_hours):
            move = np.column_stack([offsets[:, 0] + offsets[:, 2] * h, offsets[:, 1] + offsets[:, 3] * h])   # arcsec
            xy = centre + move @ jacobian.T                                                                 # pixels
            total += map_coordinates(spline if order == 3 else window, [xy[:, 1], xy[:, 0]], order=order,
                                     prefilter=False, mode="constant", cval=0.0)
        return total / np.sqrt(len(windows))

    coarse = _grid([search_arcsec, search_arcsec, search_arcsec_per_hour, search_arcsec_per_hour], coarse_step)
    first_guess = coarse[np.argmax(stacked_snr(coarse, order=1))]
    negative_log_likelihood = lambda offsets: -stacked_snr(offsets, order=3)[0] ** 2 / 2
    best = minimize(negative_log_likelihood, first_guess, method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-8, "initial_simplex": _simplex(first_guess, 0.4)}).x
    best = minimize(negative_log_likelihood, best, method="Nelder-Mead",           # restart: Nelder-Mead can stall early
                    options={"xatol": 1e-5, "fatol": 1e-10, "initial_simplex": _simplex(best, 0.05)}).x

    cov = _inverse_curvature(negative_log_likelihood, best)
    return {"jd": start["jd"], "ra": start["ra"] + best[0] / 3600 / cos_dec, "dec": start["dec"] + best[1] / 3600,
            "v_ra": start["v_ra"] + best[2], "v_dec": start["v_dec"] + best[3], "cov": cov,
            "snr": float(stacked_snr(best, order=3)[0]), "frames": len(windows), "kept": kept}


def _inverse_curvature(function, point, rounds=3):
    """Covariance = inverse Hessian of `function` at `point`, with finite-difference steps that shrink to the answer's size."""
    steps = np.full(4, 0.05)
    for _ in range(rounds):
        hessian = np.zeros((4, 4))
        for i in range(4):
            for j in range(i, 4):
                ei, ej = np.eye(4)[i] * steps[i], np.eye(4)[j] * steps[j]
                hessian[i, j] = hessian[j, i] = (function(point + ei + ej) - function(point + ei - ej)
                                                 - function(point - ei + ej) + function(point - ei - ej)) / (4 * steps[i] * steps[j])
        covariance = np.linalg.inv(hessian)
        if not np.all(np.linalg.eigvalsh(hessian) > 0):
            raise ValueError("the likelihood has no clear maximum here: the source is too faint or the start too far")
        steps = np.clip(np.sqrt(np.diag(covariance)) / 2, 1e-3, 0.3)
    return covariance


def _simplex(centre, size):
    """A starting simplex for Nelder-Mead: the centre plus one step along each axis."""
    return np.vstack([centre] + [centre + np.eye(4)[i] * size for i in range(4)])


def _prepare_frames(snr_maps, wcs_list, hours, start, cos_dec, clip_sigma):
    """Per usable frame: a clipped window around the guessed track, the track's pixel, and the arcsec->pixel matrix."""
    windows, centres, jacobians, kept = [], [], [], []
    for k, (snr_map, wcs, h) in enumerate(zip(snr_maps, wcs_list, hours)):
        ra = start["ra"] + start["v_ra"] * h / 3600 / cos_dec
        dec = start["dec"] + start["v_dec"] * h / 3600
        x, y = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        height, width = snr_map.shape
        if not (WINDOW <= x < width - WINDOW and WINDOW <= y < height - WINDOW):
            continue                                         # the track leaves the image in this frame
        x0, y0 = int(round(x)) - WINDOW, int(round(y)) - WINDOW
        window = np.clip(snr_map[y0:y0 + 2 * WINDOW + 1, x0:x0 + 2 * WINDOW + 1], -clip_sigma, clip_sigma)
        # How a shift of 1 arcsec in RA*cos(dec) and in Dec moves the track in this frame's pixels
        east_x, east_y = (float(v) for v in wcs.world_to_pixel_values(ra + 1 / 3600 / cos_dec, dec))
        north_x, north_y = (float(v) for v in wcs.world_to_pixel_values(ra, dec + 1 / 3600))
        windows.append(window.astype(np.float64))
        centres.append(np.array([x - x0, y - y0]))
        jacobians.append(np.array([[east_x - x, north_x - x], [east_y - y, north_y - y]]))
        kept.append(k)
    return windows, centres, jacobians, np.array(kept, dtype=int)


def _grid(half_widths, step):
    """All combinations of offsets within half_widths of zero, as rows."""
    axes = [np.arange(-w, w + step / 2, step) for w in half_widths]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 4)
