"""End-to-end deep search on 12 tiny synthetic frames: a faint mover is stacked, then linked across stacks."""

import numpy as np
import pandas as pd
from astropy.io import fits

from asthunt.pipeline.run_deep_search import run_deep_search
from asthunt.simulate.plan_injections import plan_injections

NIGHT = "2020-01-01"
UNIT = {"field": 618, "ccd": 3, "quadrant": 4, "night": NIGHT}
ACQUIRE = {"filter": "zr", "bad_mask_bits": [0]}
DEEP = {"frames_per_stack": 4, "max_gap_minutes": 15, "max_rate_arcsec_per_hour": 60.0, "velocity_step_pixels_per_hour": 3.0,
        "peak_threshold_sigma": 5.0, "merge_radius_pixels": 3.0, "standardize_over_night": True, "clip_sigma": 5.0, "max_velocity_mismatch_pixels_per_hour": 6.0,
        "link": {"min_rate_arcsec_per_hour": 1.0, "max_rate_arcsec_per_hour": 60.0, "match_radius_pixels": 3.0,
                 "min_detections": 3, "max_residual_pixels": 2.5}}


def write_frame(folder, fracday, obsjd, rng, shift):
    prefix = folder / f"ztf_{fracday}_000618_zr_c03_o_q4_"
    header = fits.Header({"SATURATE": 60000.0, "OBSJD": obsjd, "MAGZP": 26.0, "GAIN": 6.2, "SEEING": 2.0,
                          "MAGLIM": 20.5, "PIXSCALE": 1.0, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                          "CRVAL1": 135.6, "CRVAL2": 22.8, "CRPIX1": 80.0 + shift, "CRPIX2": 80.0 - shift,
                          "CD1_1": -1 / 3600, "CD2_2": 1 / 3600})
    yy, xx = np.indices((15, 15)) - 7
    psf = np.exp(-(xx**2 + yy**2) / 2.0)
    fits.PrimaryHDU(500 + rng.normal(0, 5, (160, 160)), header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(rng.normal(0, 5, (160, 160)).astype(np.float32), header)]) \
        .writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(np.zeros((160, 160), dtype=np.int16)).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(psf / psf.sum()).writeto(f"{prefix}diffimgpsf.fits")


def test_faint_mover_is_found_by_stacking_and_linking(tmp_path):
    rng = np.random.default_rng(0)
    paths = {"raw": tmp_path / "raw"}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    times = 2458849.9 + np.arange(12) * 4 / (24 * 60)                    # 12 frames, 4 minutes apart
    for i, t in enumerate(times):
        write_frame(night_dir, 20200101400000 + i, t, rng, shift=(i % 3) - 1)
    metadata = pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0, "obsjd": times,
                             "filefracday": [20200101400000 + i for i in range(12)]})

    # One fake at mag 21.5: about 3.6 sigma per frame after matched filtering, 30 "/hr
    fakes = plan_injections(1, (160, 160), list(times), (21.5, 21.5), (30.0, 30.0), 1.0, 40, rng)
    result = run_deep_search(UNIT, metadata, ACQUIRE, DEEP, paths, device="cpu", fakes=fakes, rng=rng)

    assert len(result["stacks"]) == 3
    assert len(result["tracklets"]) >= 1
    best = result["tracklets"].loc[result["tracklets"]["n_detections"].idxmax()]
    assert best["n_detections"] == 3
    assert abs(best["rate_arcsec_per_hour"] - 30.0) < 5
