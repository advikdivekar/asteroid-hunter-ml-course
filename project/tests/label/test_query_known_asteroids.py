import pandas as pd

from asthunt.label.query_known_asteroids import _keep_inside_image


def test_keeps_only_objects_on_the_image():
    table = pd.DataFrame({"name": ["in", "left", "above"], "x": [5.0, -3.0, 5.0], "y": [5.0, 5.0, 200.0]})
    kept = _keep_inside_image(table, image_shape=(100, 100))
    assert kept["name"].tolist() == ["in"]


def test_reads_cached_answer_without_network(tmp_path):
    from asthunt.label.query_known_asteroids import query_known_asteroids

    cache = tmp_path / "known.csv"
    pd.DataFrame({"name": ["2002 RC176"], "ra": [136.0], "dec": [22.8], "v_mag": [20.4], "x": [112.0], "y": [1802.0]}).to_csv(cache, index=False)

    known = query_known_asteroids(header=None, image_shape=(3080, 3072), cache_file=cache)

    assert known["name"].tolist() == ["2002 RC176"]
