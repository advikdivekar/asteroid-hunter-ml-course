import pandas as pd

from asthunt.acquire.select_night_exposures import select_night_exposures


def test_keeps_good_frames_of_one_night_and_filter_in_time_order():
    metadata = pd.DataFrame({
        "night":      ["n1", "n1", "n1", "n1", "n2"],
        "filtercode": ["zr", "zr", "zg", "zr", "zr"],
        "infobits":   [0,    0,    0,    5,    0],
        "obsjd":      [3.0,  1.0,  2.0,  4.0,  5.0],
    })

    chosen = select_night_exposures(metadata, night="n1", filter_code="zr", max_frames=5)

    assert chosen["obsjd"].tolist() == [1.0, 3.0]   # zg, bad-quality and other-night rows removed


def test_limits_number_of_frames():
    metadata = pd.DataFrame({"night": ["n1"] * 4, "filtercode": ["zr"] * 4,
                             "infobits": [0] * 4, "obsjd": [1.0, 2.0, 3.0, 4.0]})
    assert len(select_night_exposures(metadata, "n1", "zr", max_frames=2)) == 2
