import numpy as np
import pandas as pd

from asthunt.evaluate.find_magnitude_at_completeness import find_magnitude_at_completeness


def test_interpolates_between_bins():
    table = pd.DataFrame({"bin_centre": [19.75, 20.25, 20.75], "completeness": [0.89, 0.79, 0.29]})
    # 0.79 -> 0.29 over half a magnitude; 0.5 is 58% of the way down: 20.25 + 0.58 * 0.5 = 20.54
    assert np.isclose(find_magnitude_at_completeness(table, 0.5), 20.54)


def test_never_drops_gives_nan():
    table = pd.DataFrame({"bin_centre": [18.0, 19.0], "completeness": [0.9, 0.8]})
    assert np.isnan(find_magnitude_at_completeness(table, 0.5))
