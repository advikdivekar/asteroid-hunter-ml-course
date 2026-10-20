from astropy.table import Table

from asthunt.label.query_horizons_positions import _brightness


def test_asteroids_use_v_and_comets_use_total_magnitude():
    assert _brightness(Table({"V": [19.5, 20.1]})) == [19.5, 20.1]
    assert _brightness(Table({"Tmag": [17.0], "Nmag": [21.0]})) == [17.0]
    assert _brightness(Table({"RA": [1.0]}))[0] != _brightness(Table({"RA": [1.0]}))[0]      # NaN when unknown
