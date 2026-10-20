from asthunt.dataset.make_unit_tag import make_unit_tag


def test_tag_includes_field_ccd_quadrant_night():
    assert make_unit_tag({"field": 618, "ccd": 3, "quadrant": 4, "night": "2019-02-08"}) == "000618_c03_q4_2019-02-08"
