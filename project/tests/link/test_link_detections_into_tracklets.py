import numpy as np
import pandas as pd

from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets

SETTINGS = dict(pixel_scale=1.0, min_rate=8.0, max_rate=120.0, match_radius=2.0, min_detections=4, max_residual=1.5)
TIMES_JD = 2459000.0 + np.array([0, 7, 14, 21, 28]) / (24 * 60)        # 5 frames, 7 minutes apart


def night_with_one_asteroid(rng, n_noise=300, skip_frame=None):
    rows = []
    for frame, jd in enumerate(TIMES_JD):
        hours = (jd - TIMES_JD[0]) * 24
        if frame != skip_frame:
            rows.append({"x": 500 + 40 * hours, "y": 300 - 20 * hours, "frame_index": frame, "obsjd": jd, "true": 1})
        for x, y in rng.uniform(0, 1000, size=(n_noise, 2)):
            rows.append({"x": x, "y": y, "frame_index": frame, "obsjd": jd, "true": 0})
    return pd.DataFrame(rows)


def test_finds_the_moving_object_among_noise():
    detections = night_with_one_asteroid(np.random.default_rng(0))

    tracklets, members = link_detections_into_tracklets(detections, **SETTINGS)

    real_ids = set(detections.index[detections["true"] == 1])
    member_sets = [set(group["detection_id"]) for _, group in members.groupby("tracklet_id")]
    assert real_ids in member_sets
    best = tracklets.loc[tracklets["n_detections"].idxmax()]
    assert abs(best["rate_arcsec_per_hour"] - np.hypot(40, 20)) < 1


def test_one_missing_detection_is_allowed():
    detections = night_with_one_asteroid(np.random.default_rng(1), skip_frame=2)
    _, members = link_detections_into_tracklets(detections, **SETTINGS)
    real_ids = set(detections.index[detections["true"] == 1])
    assert real_ids in [set(g["detection_id"]) for _, g in members.groupby("tracklet_id")]


def test_stationary_source_is_not_linked():
    rows = [{"x": 100.0, "y": 100.0, "frame_index": f, "obsjd": jd} for f, jd in enumerate(TIMES_JD)]
    tracklets, _ = link_detections_into_tracklets(pd.DataFrame(rows), **SETTINGS)
    assert len(tracklets) == 0


def test_no_detection_is_shared():
    detections = night_with_one_asteroid(np.random.default_rng(2), n_noise=2000)
    _, members = link_detections_into_tracklets(detections, **SETTINGS)
    assert members["detection_id"].is_unique


def test_jittery_stationary_source_is_not_linked():
    # A variable star or bad subtraction: same spot every frame, centroid wobbling by ~0.5 px
    rng = np.random.default_rng(3)
    rows = [{"x": 100.0 + rng.normal(0, 0.5), "y": 100.0 + rng.normal(0, 0.5), "frame_index": f, "obsjd": jd}
            for f, jd in enumerate(TIMES_JD)]
    tracklets, _ = link_detections_into_tracklets(pd.DataFrame(rows), **SETTINGS)
    assert len(tracklets) == 0


def test_returns_the_fitted_line():
    detections = night_with_one_asteroid(np.random.default_rng(0), n_noise=50)
    tracklets, members = link_detections_into_tracklets(detections, **SETTINGS)
    best = tracklets.loc[tracklets["n_detections"].idxmax()]
    assert abs(best["x0"] - 500) < 0.1 and abs(best["y0"] - 300) < 0.1      # where it was at t0
    assert abs(best["t0_jd"] - TIMES_JD[0]) < 1e-9
