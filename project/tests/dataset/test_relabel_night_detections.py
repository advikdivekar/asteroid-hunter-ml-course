import pandas as pd

from asthunt.dataset.relabel_night_detections import relabel_night_detections

INJECT = {"known_asteroid_max_v_mag": 20.5, "match_radius_pixels": 2.0, "known_match_radius_pixels": 2.0}


def test_new_known_positions_change_labels_but_not_row_order(tmp_path):
    paths = {"processed": tmp_path / "processed", "interim": tmp_path / "interim"}
    paths["processed"].mkdir(); (paths["interim"] / "known_asteroids").mkdir(parents=True)
    detections = pd.DataFrame({"x": [100.0, 50.0, 10.0], "y": [100.0, 50.0, 10.0], "frame_index": [0, 0, 0],
                               "exposure": "e0", "injection_id": -1, "known_asteroid": "", "label": 0},
                              index=[7, 3, 5])
    detections.to_parquet(paths["processed"] / "detections__000618_c03_q4_n1.parquet")
    pd.DataFrame({"injection_id": [0], "frame_index": [0], "x": [10.2], "y": [10.0]}) \
        .to_parquet(paths["processed"] / "injections__000618_c03_q4_n1.parquet")
    # SkyBoT's position is 3 px off (as on real data); the Horizons refinement is right
    for file_name, x in [("known_asteroids__e0.csv", 103.0), ("known_asteroids_horizons__e0.csv", 100.5)]:
        pd.DataFrame({"name": ["Anius"], "ra": [0.0], "dec": [0.0], "v_mag": [19.0], "x": [x], "y": [100.0]}) \
            .to_csv(paths["interim"] / "known_asteroids" / file_name, index=False)

    relabelled = relabel_night_detections({"field": 618, "ccd": 3, "quadrant": 4, "night": "n1"}, paths, INJECT)

    assert relabelled.index.tolist() == [7, 3, 5]                       # same order as the stamps
    assert relabelled["label"].tolist() == [1, 0, 1]
    assert relabelled["known_asteroid"].tolist() == ["Anius", "", ""]
    assert relabelled["injection_id"].tolist() == [-1, -1, 0]
