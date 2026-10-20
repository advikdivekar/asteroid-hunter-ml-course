import numpy as np
import pandas as pd

from asthunt.report.write_candidate_report import write_candidate_report


def test_writes_page_and_one_image_per_candidate(tmp_path):
    detections = pd.DataFrame({"obsjd": [1.0, 2.0], "frame_index": [0, 1], "snr": [5.0, 6.0],
                               "mag": [20.0, 20.1], "ra": [135.6, 135.61], "dec": [22.8, 22.8]})
    stamps = np.zeros((2, 3, 31, 31))
    candidates = pd.DataFrame({"tracklet_id": [7], "n_detections": [2], "rate_arcsec_per_hour": [30.0], "tracklet_score": [0.9]})
    members = pd.DataFrame({"tracklet_id": [7, 7], "detection_id": [0, 1]})

    page = write_candidate_report(candidates, members, detections, stamps, tmp_path, "test night")

    assert "Tracklet 7" in page.read_text()
    assert (tmp_path / "tracklet_7.png").exists()
