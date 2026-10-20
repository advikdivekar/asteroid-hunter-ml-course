import pandas as pd

from asthunt.acquire.load_unit_metadata import load_unit_metadata


def test_exposures_missing_from_the_archive_are_left_out(tmp_path):
    folder = tmp_path / "ztf"
    folder.mkdir()
    pd.DataFrame({"filefracday": [1, 2, 3], "night": ["2019-01-28"] * 3}).to_csv(folder / "metadata__000567_c10_q1.csv", index=False)
    pd.DataFrame([{"field": 567, "ccdid": 10, "qid": 1, "filefracday": 2, "product": "diffimgpsf.fits"},
                  {"field": 567, "ccdid": 11, "qid": 1, "filefracday": 3, "product": "diffimgpsf.fits"}]   # another CCD
                 ).to_csv(folder / "missing_exposures.csv", index=False)

    metadata = load_unit_metadata({"field": 567, "ccd": 10, "quadrant": 1}, tmp_path)

    assert metadata["filefracday"].tolist() == [1, 3]
