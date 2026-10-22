"""End-to-end run of the deployed pipeline on 3 tiny fake frames (no network needed)."""


import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from asthunt.models.define_realbogus_cnn import define_realbogus_cnn
from asthunt.models.train_realbogus_trees import train_realbogus_trees
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier
from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS
from asthunt.pipeline.run_night_pipeline import run_night_pipeline

NIGHT = "2020-01-01"
CONFIGS = {
    "acquire": {"field": 618, "ccd": 3, "quadrant": 4, "filter": "zr", "frames_per_night": 3, "bad_mask_bits": [0]},
    "inject": {"asteroids_per_night": 4, "magnitude_range": [16.0, 16.5],
               "rate_range_arcsec_per_hour": [60.0, 60.0], "edge_margin_pixels": 40, "match_radius_pixels": 2.0,
               "known_match_radius_pixels": 4.0, "known_asteroid_max_v_mag": 20.5},
    "detect": {"threshold_sigma": 3.0, "min_area_pixels": 5, "stamp_size": 31},
    "link": {"pixel_scale_arcsec": 1.0, "min_rate_arcsec_per_hour": 8.0, "max_rate_arcsec_per_hour": 120.0,
             "match_radius_pixels": 2.0, "min_detections": 3, "max_residual_pixels": 1.5},
}


def write_frame(folder, fracday, obsjd, rng, shift):
    prefix = folder / f"ztf_{fracday}_000618_zr_c03_o_q4_"
    header = fits.Header({"SATURATE": 60000.0, "OBSJD": obsjd, "MAGZP": 26.0, "GAIN": 6.2, "SEEING": 2.0,
                          "MAGLIM": 20.5, "PIXSCALE": 1.0, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                          "CRVAL1": 135.6, "CRVAL2": 22.8, "CRPIX1": 150.0 + shift, "CRPIX2": 150.0 + shift,
                          "CD1_1": -1 / 3600, "CD2_2": 1 / 3600})
    yy, xx = np.indices((25, 25)) - 12
    psf = np.exp(-(xx**2 + yy**2) / 2.0)
    fits.PrimaryHDU(500 + rng.normal(0, 5, (300, 300)), header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(rng.normal(0, 5, (300, 300)).astype(np.float32), header)]) \
        .writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(np.zeros((300, 300), dtype=np.int16)).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(psf / psf.sum()).writeto(f"{prefix}diffimgpsf.fits")
    return f"ztf_{fracday}_000618_zr_c03_o_q4"


@pytest.mark.parametrize("stage1", ["classical", "cnn"])
def test_pipeline_finds_planted_asteroids_and_never_reports_them_as_candidates(tmp_path, stage1):
    rng = np.random.default_rng(0)
    paths = {name: tmp_path / name for name in ["raw", "interim", "processed", "reports"]}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    times = 2458849.9 + np.arange(3) * 10 / (24 * 60)                  # 3 frames, 10 minutes apart
    names = [write_frame(night_dir, 20200101400000 + i, t, rng, shift=5.0 * i) for i, t in enumerate(times)]
    metadata = pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0, "obsjd": times,
                             "filefracday": [20200101400000 + i for i in range(3)]})
    (paths["interim"] / "known_asteroids").mkdir(parents=True)
    for name in names:
        pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"]).to_csv(
            paths["interim"] / "known_asteroids" / f"known_asteroids__{name}.csv", index=False)

    # A dummy bundle: an untrained CNN or a classical model fitted to noise (the deployed default since v6);
    # threshold 0 keeps every detection; a tracklet model that accepts everything
    features = rng.uniform(size=(50, len(FEATURE_COLUMNS)))
    noise = rng.normal(size=(40, 3, 31, 31)).astype(np.float32)
    blobs = pd.DataFrame({c: rng.uniform(1, 2, 40) for c in ["snr", "peak", "npix", "a", "b", "flux"]})
    stage1_model = define_realbogus_cnn() if stage1 == "cnn" else train_realbogus_trees(noise, blobs, np.arange(40) % 2)
    bundle = {"realbogus_model": stage1_model,
              "tracklet_model": train_tracklet_classifier(features, np.arange(50) % 2),
              "thresholds": {"realbogus": 0.0, "tracklet": 0.0}, "info": {}}

    unit = {"field": 618, "ccd": 3, "quadrant": 4, "night": NIGHT}
    result = run_night_pipeline(unit, metadata, CONFIGS, bundle, paths, rng, with_injections=True)

    assert {"detections", "tracklets", "members", "candidates", "injections", "known"} <= set(result)
    assert (result["tracklets"]["label"] == 1).sum() >= 3                # bright fakes linked as clean tracklets
    assert not result["candidates"]["contains_fake"].any()               # fakes are never "discoveries"
    assert (paths["reports"] / f"000618_c03_q4_{NIGHT}" / "index.html").exists()
