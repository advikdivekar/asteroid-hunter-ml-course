import pandas as pd

from asthunt.features.count_static_neighbours import count_static_neighbours


def test_mover_has_none_artifact_walker_has_many():
    rows = []
    for frame in range(3):
        rows.append({"x_common": 100 + 10 * frame, "y_common": 100.0, "frame_index": frame})     # a real mover
        for x in (500.0, 505.0, 510.0):                                                        # a faint line of blips,
            rows.append({"x_common": x, "y_common": 500.0, "frame_index": frame})               # present in every frame
    detections = pd.DataFrame(rows)
    # rows per frame: mover, 500, 505, 510 -> frame f starts at index 4*f
    tracklets = pd.DataFrame({"tracklet_id": [0, 1]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0, 1, 1, 1],
                            "detection_id": [0, 4, 8,            # the mover
                                             1, 6, 11]})         # the "walker": 500 (f0), 505 (f1), 510 (f2)

    table = count_static_neighbours(tracklets, members, detections).set_index("tracklet_id")

    assert table.loc[0, "static_neighbours_max"] == 0
    assert table.loc[1, "static_neighbours_mean"] == 2        # the line is still there at each spot in the other 2 frames


def test_a_slow_movers_own_detections_do_not_count():
    # Moves only 1.5 px per frame: its own detections in the other frames are within the 2 px circle
    detections = pd.DataFrame({"x_common": [100.0, 101.5, 103.0], "y_common": 100.0, "frame_index": [0, 1, 2]})
    tracklets = pd.DataFrame({"tracklet_id": [0]})
    members = pd.DataFrame({"tracklet_id": [0, 0, 0], "detection_id": [0, 1, 2]})

    table = count_static_neighbours(tracklets, members, detections)

    assert table.loc[0, "static_neighbours_max"] == 0
