import pandas as pd

from asthunt.label.load_known_asteroids import load_known_asteroids


def write(folder, file_name, names, xs):
    pd.DataFrame({"name": names, "ra": 135.0, "dec": 22.0, "v_mag": 19.0, "x": xs, "y": 10.0}).to_csv(folder / file_name, index=False)


def test_horizons_where_refined_skybot_for_the_rest(tmp_path):
    folder = tmp_path / "known_asteroids"
    folder.mkdir()
    write(folder, "known_asteroids__e1.csv", ["Anius", "Faint"], [100.0, 300.0])
    assert load_known_asteroids("e1", tmp_path)["source"].tolist() == ["skybot", "skybot"]

    write(folder, "known_asteroids_horizons__e1.csv", ["Anius"], [97.5])
    table = load_known_asteroids("e1", tmp_path).set_index("name")
    assert table.loc["Anius", "source"] == "horizons" and table.loc["Anius", "x"] == 97.5
    assert table.loc["Faint", "source"] == "skybot" and table.loc["Faint", "x"] == 300.0
