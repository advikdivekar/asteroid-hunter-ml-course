"""End-to-end fast-mover search on 3 synthetic frames: a trailed fake must come out as one consistent tracklet."""

import numpy as np
import pandas as pd
from astropy.io import fits

from asthunt.pipeline.run_fast_search import run_fast_search

NIGHT = "2020-01-01"
UNIT = {"field": 618, "ccd": 3, "quadrant": 4, "night": NIGHT}
ACQUIRE = {"filter": "zr", "bad_mask_bits": [0]}
FAST = {"exposure_seconds": 30, "standardize_over_night": True, "streak_lengths_pixels": [4, 8, 16], "angle_step_degrees": 15, "detection_threshold_sigma": 7.0,
        "link": {"min_rate_arcsec_per_hour": 300.0, "max_rate_arcsec_per_hour": 15000.0, "match_radius_pixels": 4.0,
                 "min_detections": 3, "max_residual_pixels": 3.0},
        "max_angle_difference_degrees": 20, "length_ratio_range": [0.5, 2.0]}


def write_frame(folder, fracday, obsjd, rng, shift):
    prefix = folder / f"ztf_{fracday}_000618_zr_c03_o_q4_"
    header = fits.Header({"SATURATE": 60000.0, "OBSJD": obsjd, "MAGZP": 26.0, "GAIN": 6.2, "SEEING": 2.0,
                          "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 135.6, "CRVAL2": 22.8,
                          "CRPIX1": 150.0 + shift, "CRPIX2": 150.0, "CD1_1": -1.012 / 3600, "CD2_2": 1.012 / 3600})
    yy, xx = np.indices((15, 15)) - 7
    psf = np.exp(-(xx**2 + yy**2) / 2.0)
    fits.PrimaryHDU(500 + rng.normal(0, 5, (300, 300)), header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(rng.normal(0, 5, (300, 300)).astype(np.float32), header)]) \
        .writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(np.zeros((300, 300), dtype=np.int16)).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(psf / psf.sum()).writeto(f"{prefix}diffimgpsf.fits")


def run_with_fake(tmp_path, angle, fast):
    """Plant one trailed fake (1,200 "/hr, ~10 px trail, ~80 px between frames) heading `angle` degrees and run the search."""
    rng = np.random.default_rng(0)
    paths = {"raw": tmp_path / "raw"}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    times = 2458849.9 + np.arange(3) * 4 / (24 * 60)
    for i, t in enumerate(times):
        write_frame(night_dir, 20200101400000 + i, t, rng, shift=3.0 * i)
    metadata = pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0, "obsjd": times,
                             "filefracday": [20200101400000 + i for i in range(3)]})
    hours = (times - times[0]) * 24
    pixels_per_hour = 1200 / 1.012
    fakes = pd.DataFrame({"injection_id": 0, "frame_index": range(3), "mag": 17.0, "rate_arcsec_per_hour": 1200.0,
                          "angle_degrees": float(angle),
                          "x": 60 + pixels_per_hour * np.cos(np.radians(angle)) * hours,
                          "y": 80 + pixels_per_hour * np.sin(np.radians(angle)) * hours})
    result = run_fast_search(UNIT, metadata, ACQUIRE, fast, paths, fakes=fakes, rng=rng, max_frames=3)
    return result["tracklets"][result["tracklets"]["consistent"]]


def test_trailed_fake_is_linked_and_consistent(tmp_path):
    consistent = run_with_fake(tmp_path, 30, FAST)
    assert len(consistent) == 1
    assert abs(consistent.iloc[0]["rate_arcsec_per_hour"] - 1200) < 100


def test_a_trail_along_the_column_is_dropped_only_when_the_veto_is_configured(tmp_path):
    veto = {"column_veto": {"max_angle_from_column_degrees": 3.5, "max_column_spread_pixels": 12}}
    assert len(run_with_fake(tmp_path / "off", 90, FAST)) == 1
    assert len(run_with_fake(tmp_path / "on", 90, {**FAST, **veto})) == 0
    assert len(run_with_fake(tmp_path / "other_direction", 30, {**FAST, **veto})) == 1          # the veto leaves other directions alone
