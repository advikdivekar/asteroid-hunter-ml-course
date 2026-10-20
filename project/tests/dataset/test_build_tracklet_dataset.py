import numpy as np
import pandas as pd

from asthunt.dataset.build_tracklet_dataset import build_tracklet_dataset

LINK = {"pixel_scale_arcsec": 1.0, "min_rate_arcsec_per_hour": 3.0, "max_rate_arcsec_per_hour": 120.0,
        "match_radius_pixels": 2.0, "min_detections": 4, "max_residual_pixels": 1.5}


def test_links_labels_and_saves_a_fake_asteroid(tmp_path):
    rng = np.random.default_rng(0)
    times = 2459000.0 + np.arange(5) * 7 / (24 * 60)
    rows = []
    for frame, jd in enumerate(times):
        hours = (jd - times[0]) * 24
        rows.append({"x": 400 + 40 * hours, "y": 300.0, "frame_index": frame, "obsjd": jd, "injection_id": 3})
        for x, y in rng.uniform(0, 1000, size=(100, 2)):
            rows.append({"x": x, "y": y, "frame_index": frame, "obsjd": jd, "injection_id": -1})
    detections = pd.DataFrame(rows)
    detections["x_common"], detections["y_common"] = detections["x"], detections["y"]   # pointing didn't move
    detections["known_asteroid"] = ""
    for column, value in [("mag", 20.0), ("snr", 8.0), ("npix", 10), ("a", 1.1), ("b", 1.0)]:
        detections[column] = value
    detections.to_parquet(tmp_path / "detections__000618_c03_q4_n1.parquet")
    np.save(tmp_path / "realbogus_scores__000618_c03_q4_n1.npy", np.ones(len(detections)))

    unit = {"field": 618, "ccd": 3, "quadrant": 4, "night": "n1"}
    tracklets, members = build_tracklet_dataset(unit, tmp_path, realbogus_threshold=0.5, link=LINK)

    assert (tracklets["label"] == 1).sum() == 1
    assert tracklets.loc[tracklets["label"] == 1, "injection_id"].item() == 3
    assert (tmp_path / "tracklets__000618_c03_q4_n1.parquet").exists()
