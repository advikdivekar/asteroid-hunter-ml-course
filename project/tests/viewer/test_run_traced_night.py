"""The live viewer must show what the pipeline really does: on tiny fake frames both give the same detections, scores and tracks."""

import numpy as np
import pandas as pd
from astropy.io import fits

from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.features.compute_tracklet_features import FEATURE_COLUMNS
from asthunt.models.define_realbogus_cnn import define_realbogus_cnn
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier
from asthunt.pipeline.run_night_pipeline import run_night_pipeline
from asthunt.viewer.live_state import LiveState
from asthunt.viewer.run_traced_night import STAGE_PLAN, run_traced_night

NIGHT = "2020-01-01"
UNIT = {"field": 618, "ccd": 3, "quadrant": 4, "night": NIGHT}
CONFIGS = {
    "acquire": {"field": 618, "ccd": 3, "quadrant": 4, "filter": "zr", "frames_per_night": 3, "bad_mask_bits": [0]},
    "inject": {"asteroids_per_night": 4, "magnitude_range": [16.0, 16.5], "rate_range_arcsec_per_hour": [60.0, 60.0], "edge_margin_pixels": 40,
               "match_radius_pixels": 2.0, "known_match_radius_pixels": 4.0, "known_asteroid_max_v_mag": 20.5},
    "detect": {"threshold_sigma": 3.0, "min_area_pixels": 5, "stamp_size": 31},
    "link": {"pixel_scale_arcsec": 1.0, "min_rate_arcsec_per_hour": 8.0, "max_rate_arcsec_per_hour": 120.0,
             "match_radius_pixels": 2.0, "min_detections": 3, "max_residual_pixels": 1.5},
}


def write_frame(folder, fracday, obsjd, rng, shift):
    prefix = folder / f"ztf_{fracday}_000618_zr_c03_o_q4_"
    header = fits.Header({"SATURATE": 60000.0, "OBSJD": obsjd, "MAGZP": 26.0, "GAIN": 6.2, "SEEING": 2.0, "MAGLIM": 20.5, "PIXSCALE": 1.0,
                          "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 135.6, "CRVAL2": 22.8, "CRPIX1": 150.0 + shift,
                          "CRPIX2": 150.0 + shift, "CD1_1": -1 / 3600, "CD2_2": 1 / 3600})
    yy, xx = np.indices((25, 25)) - 12
    psf = np.exp(-(xx**2 + yy**2) / 2.0)
    fits.PrimaryHDU(500 + rng.normal(0, 5, (300, 300)), header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(rng.normal(0, 5, (300, 300)).astype(np.float32), header)]).writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(np.zeros((300, 300), dtype=np.int16)).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(psf / psf.sum()).writeto(f"{prefix}diffimgpsf.fits")
    return f"ztf_{fracday}_000618_zr_c03_o_q4"


def test_the_traced_run_matches_the_pipeline_and_reports_every_stage(tmp_path):
    rng = np.random.default_rng(0)
    paths = {name: tmp_path / name for name in ["raw", "interim", "processed", "reports"]}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    times = 2458849.9 + np.arange(3) * 10 / (24 * 60)
    names = [write_frame(night_dir, 20200101400000 + i, t, rng, shift=5.0 * i) for i, t in enumerate(times)]
    pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0, "obsjd": times, "filefracday": [20200101400000 + i for i in range(3)]}) \
        .to_csv(paths["raw"] / "ztf" / "metadata__000618_c03_q4.csv", index=False)
    (paths["interim"] / "known_asteroids").mkdir(parents=True)
    for name in names:
        pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"]).to_csv(paths["interim"] / "known_asteroids" / f"known_asteroids__{name}.csv", index=False)
    features = rng.uniform(size=(50, len(FEATURE_COLUMNS)))
    bundle = {"realbogus_model": define_realbogus_cnn(), "tracklet_model": train_tracklet_classifier(features, np.arange(50) % 2),
              "thresholds": {"realbogus": 0.0, "tracklet": 0.0}, "info": {}}

    metadata = pd.read_csv(paths["raw"] / "ztf" / "metadata__000618_c03_q4.csv")
    expected = run_night_pipeline(UNIT, metadata, CONFIGS, bundle, paths, make_unit_rng(42, UNIT), with_injections=True, write_reports=False)
    # the pipeline caches its tables; the viewer builds from scratch, so give it a clean processed folder
    state = LiveState()
    state.reset(STAGE_PLAN, UNIT, step_mode=False)
    seen = run_traced_night(UNIT, state, CONFIGS, bundle, {**paths, "processed": tmp_path / "other"}, 4, 42, "cpu")

    np.testing.assert_allclose(seen["detections"][["x", "y"]].to_numpy(), expected["detections"][["x", "y"]].to_numpy())
    np.testing.assert_allclose(seen["detections"]["score"].to_numpy(), expected["detections"]["score"].to_numpy(), atol=1e-6)
    assert len(seen["tracklets"]) == len(expected["tracklets"]) > 0
    np.testing.assert_allclose(seen["tracklets"]["tracklet_score"].to_numpy(), expected["tracklets"]["tracklet_score"].to_numpy(), atol=1e-6)
    snapshot = state.snapshot()
    assert snapshot["status"] == "finished"
    assert [s["status"] for s in snapshot["stages"]] == ["done", "done", "done", "done", "done", "done", "done", "done", "done", "done", "done", "done"]
    assert sum(len(s["images"]) for s in snapshot["stages"]) >= 10            # the pictures really were drawn


def test_a_night_without_tracks_skips_the_later_stages(tmp_path):
    rng = np.random.default_rng(1)
    paths = {name: tmp_path / name for name in ["raw", "interim", "processed", "reports"]}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    times = 2458849.9 + np.arange(3) * 10 / (24 * 60)
    names = [write_frame(night_dir, 20200101400000 + i, t, rng, shift=5.0 * i) for i, t in enumerate(times)]
    pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0, "obsjd": times, "filefracday": [20200101400000 + i for i in range(3)]}) \
        .to_csv(paths["raw"] / "ztf" / "metadata__000618_c03_q4.csv", index=False)
    (paths["interim"] / "known_asteroids").mkdir(parents=True)
    for name in names:
        pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"]).to_csv(paths["interim"] / "known_asteroids" / f"known_asteroids__{name}.csv", index=False)
    features = rng.uniform(size=(50, len(FEATURE_COLUMNS)))
    bundle = {"realbogus_model": define_realbogus_cnn(), "tracklet_model": train_tracklet_classifier(features, np.arange(50) % 2),
              "thresholds": {"realbogus": 1.01, "tracklet": 0.0}, "info": {}}          # a CNN cut nothing can pass: no detection is kept

    state = LiveState()
    state.reset(STAGE_PLAN, UNIT, step_mode=False)
    run_traced_night(UNIT, state, CONFIGS, bundle, paths, 0, 42, "cpu")

    statuses = {s["id"]: s["status"] for s in state.snapshot()["stages"]}
    assert statuses["link"] == "done" and statuses["features"] == statuses["classify"] == statuses["known"] == "skipped"
    assert statuses["plant"] == "skipped" and statuses["result"] == "done"
