import pandas as pd

from asthunt.report.write_ades_report import write_ades_report


def test_writes_header_and_mid_exposure_times(tmp_path):
    detections = pd.DataFrame({"obsjd": [2459000.5], "ra": [135.6], "dec": [22.8], "mag": [20.1]})
    candidates = pd.DataFrame({"tracklet_id": [0]})
    members = pd.DataFrame({"tracklet_id": [0], "detection_id": [0]})

    path = write_ades_report(candidates, members, detections, tmp_path / "draft.psv")
    lines = path.read_text().splitlines()

    assert lines[0] == "# version=2017"
    assert "NOT FOR SUBMISSION" in path.read_text()
    assert lines[-1].startswith("ah00001|CCD|I41|2020-05-31T00:00:15.000Z|135.600000|22.800000")
