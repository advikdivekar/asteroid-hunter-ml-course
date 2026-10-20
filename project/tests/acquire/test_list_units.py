from asthunt.acquire.list_units import list_units

ACQUIRE = {
    "field": 618, "ccd": 3, "quadrant": 4,
    "nights": {"train": ["n1", "n2"], "validation": ["n3"], "test": ["n4"]},
    "holdout_nights": ["n5"],
    "extra_quadrants": [[3, 1]],
    "extra_sources": [{"field": 567, "ccd": 10, "quadrant": 1, "split": "train", "nights": ["m1"]}],
}


def test_extra_quadrants_follow_their_nights_and_sources_their_split():
    train = list_units(ACQUIRE, "train")
    assert len(train) == 2 * 2 + 1
    assert {"field": 567, "ccd": 10, "quadrant": 1, "night": "m1"} in train
    assert {u["night"] for u in list_units(ACQUIRE, "validation")} == {"n3"}
    assert len(list_units(ACQUIRE, "holdout")) == 2


def test_original_quadrant_only():
    assert list_units(ACQUIRE, "train", include_extra=False) == [
        {"field": 618, "ccd": 3, "quadrant": 4, "night": "n1"}, {"field": 618, "ccd": 3, "quadrant": 4, "night": "n2"}]
