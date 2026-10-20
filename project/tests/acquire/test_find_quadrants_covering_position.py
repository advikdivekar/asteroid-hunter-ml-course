import pandas as pd

from asthunt.acquire.find_quadrants_covering_position import find_quadrants_covering_position


def square(qid, ra, dec, half=0.43):
    return {"qid": qid, "ra": ra, "dec": dec,
            "ra1": ra - half, "dec1": dec - half, "ra2": ra + half, "dec2": dec - half,
            "ra3": ra + half, "dec3": dec + half, "ra4": ra - half, "dec4": dec + half}


def test_position_is_found_in_the_right_quadrant_only():
    metadata = pd.DataFrame([square(1, 130.0, 0.0), square(2, 130.9, 0.0)])
    assert find_quadrants_covering_position(metadata, 130.1, 0.2)["qid"].tolist() == [1]
    assert find_quadrants_covering_position(metadata, 130.8, -0.1)["qid"].tolist() == [2]
    assert find_quadrants_covering_position(metadata, 132.0, 0.0).empty


def test_ra_wraps_around_zero():
    metadata = pd.DataFrame([square(3, 359.9, 10.0)])
    metadata.loc[0, ["ra2", "ra3"]] = [0.33, 0.33]           # the right edge is past 0 deg
    assert find_quadrants_covering_position(metadata, 0.1, 10.0)["qid"].tolist() == [3]
