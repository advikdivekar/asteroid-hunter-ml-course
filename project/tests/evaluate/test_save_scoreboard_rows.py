from asthunt.evaluate.save_scoreboard_rows import save_scoreboard_rows


def test_rerun_replaces_only_its_own_rows(tmp_path):
    path = tmp_path / "scoreboard.csv"
    save_scoreboard_rows([{"version": "v1", "split": "test", "ap": 0.90}], path)
    save_scoreboard_rows([{"version": "v2", "split": "test", "ap": 0.95}], path)
    table = save_scoreboard_rows([{"version": "v1", "split": "test", "ap": 0.91}], path)

    assert len(table) == 2
    assert table.set_index("version").loc["v1", "ap"] == 0.91
    assert table.set_index("version").loc["v2", "ap"] == 0.95
