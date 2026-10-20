from asthunt.evaluate.check_split_leakage import check_split_leakage


def test_clean_split_has_no_shared_groups():
    assert check_split_leakage(["n1", "n1", "n2"], ["n3", "n3"]) == []


def test_reports_every_shared_group_once():
    assert check_split_leakage(["n1", "n2", "n2"], ["n2", "n1", "n3"]) == ["n1", "n2"]
