import numpy as np
import pandas as pd

from asthunt.viewer import draw_stage_figures as draw

PNG = b"\x89PNG"


def test_every_picture_comes_back_as_a_png():
    rng = np.random.default_rng(0)
    image = rng.normal(0, 5, (300, 320))
    bad = np.zeros(image.shape, bool)
    detections = pd.DataFrame({"x": [50.0, 150.0, 200.0], "y": [60.0, 120.0, 90.0], "snr": [5.0, 9.0, 7.0],
                               "injection_id": [-1, 2, -1], "known_asteroid": ["", "", "1999 AB"]})
    stamps = rng.normal(size=(5, 3, 31, 31))
    kept = pd.DataFrame({"x_common": [10.0, 20.0, 30.0], "y_common": [10.0, 20.0, 30.0], "frame_index": [0, 1, 2]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0], "detection_id": [0, 1, 2]})
    tracklets = pd.DataFrame({"tracklet_id": [0], "n_detections": [3], "max_residual": [0.1], "rate_arcsec_per_hour": [50.0],
                              "tracklet_score": [0.9], "contains_fake": [False], "known_name": [""]})
    results = [
        draw.draw_four_panels(image, image, image, bad, "x"),
        draw.draw_four_panels(None, None, image, bad, "light mode"),
        draw.draw_noise_histogram(image.ravel(), 5.0, 5.1),
        draw.draw_detections(image, detections, "x"),
        draw.draw_detections(image, detections.iloc[:0], "none found"),
        draw.draw_stamp_mosaic(stamps, [f"s{i}" for i in range(5)], "x"),
        draw.draw_score_histogram(rng.uniform(size=200), 0.5, [("fakes", draw.TEAL, np.arange(200) < 10)]),
        draw.draw_tracklets(kept, members, tracklets, 3, "x"),
        draw.draw_tracklet_scores(tracklets, 0.5, "x"),
    ]
    assert all(png.startswith(PNG) for png in results)


def test_stretch_handles_all_nan_and_block_mean_shrinks():
    assert np.all(draw.stretch(np.full((4, 4), np.nan)) == 0)
    assert draw.block_mean(np.ones((10, 10)), 5).shape == (2, 2)
