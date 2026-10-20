"""What known asteroids say about our measurement errors, and how to apply it to a new object."""

import numpy as np

DIFFERENCES = ["d_ra", "d_dec", "d_v_ra", "d_v_dec"]
SIGMAS = ["sigma_ra", "sigma_dec", "sigma_v_ra", "sigma_v_dec"]
RIGHT_OBJECT_ARCSEC = 1.0          # a fit this close to Horizons found the object; farther means it locked onto something else
MIN_SIGMA_ARCSEC_TIMES_SNR = 0.3   # a fitted position error below 0.3 arcsec / SNR is not believable (a spike, not a source)
WELL_MEASURED_SNR, BRIGHT_SNR = 15.0, 30.0


def summarize_calibration(table):
    """Return the bias, systematic floor and error scale that the known asteroids in `table` imply.

    table : one row per known asteroid with name, snr, d_* (fit minus Horizons: arcsec, arcsec/hour, for
            RA*cos dec, Dec, v_ra, v_dec) and sigma_* (our statistical error on each)

    Rows are left out when the fit found a different object (position off by more than 1 arcsec) or its error is
    implausibly small; they are listed so nothing is hidden.
      bias  : median (fit - Horizons) of well measured objects (SNR >= 15): a frame-level offset, not noise;
      floor : scatter of the bright ones (SNR >= 30) around the bias, less their statistical error: what the method
              cannot beat however bright the source (per-frame astrometric jitter; shows up mostly in the velocity);
      scale : how much to multiply the statistical errors of FAINT objects (SNR < 30, like a candidate), a robust
              estimate from (difference - bias) / sigma, never below 1.
    """
    position_error = np.hypot(table["d_ra"], table["d_dec"])
    smallest_sigma = table[["sigma_ra", "sigma_dec"]].min(axis=1)
    implausible = smallest_sigma < MIN_SIGMA_ARCSEC_TIMES_SNR / table["snr"]
    usable = table[(position_error < RIGHT_OBJECT_ARCSEC) & ~implausible]
    left_out = table.loc[~table.index.isin(usable.index), "name"].tolist()

    well = usable[usable["snr"] >= WELL_MEASURED_SNR]
    bias = well[DIFFERENCES].median().to_numpy()
    bias_error = 1.2533 * well[DIFFERENCES].std().to_numpy() / np.sqrt(len(well))          # standard error of a median
    bright = usable[usable["snr"] >= BRIGHT_SNR]
    scatter = 1.4826 * (bright[DIFFERENCES] - bias).abs().median().to_numpy()
    floor = np.sqrt(np.maximum(scatter ** 2 - bright[SIGMAS].median().to_numpy() ** 2, 0.0))
    faint = usable[usable["snr"] < BRIGHT_SNR]
    z = ((faint[DIFFERENCES].to_numpy() - bias) / faint[SIGMAS].to_numpy()).ravel()
    scale = max(1.0, float(1.4826 * np.median(np.abs(z - np.median(z))))) if len(z) else 1.0
    return {"n_usable": len(usable), "n_well": len(well), "n_bright": len(bright), "n_faint": len(faint), "left_out": left_out,
            "bias": bias.tolist(), "bias_error": bias_error.tolist(), "floor": floor.tolist(), "scale": scale}


def calibrate_fit(fit, summary):
    """The fit with the bias taken out and an error budget: scaled statistical errors + floor + the bias's own uncertainty."""
    cos_dec = np.cos(np.radians(fit["dec"]))
    bias = np.array(summary["bias"])
    covariance = summary["scale"] ** 2 * np.array(fit["cov"]) + np.diag(np.square(summary["floor"]) + np.square(summary["bias_error"]))
    return {**fit, "ra": fit["ra"] - bias[0] / 3600 / cos_dec, "dec": fit["dec"] - bias[1] / 3600,
            "v_ra": fit["v_ra"] - bias[2], "v_dec": fit["v_dec"] - bias[3], "cov": covariance}
