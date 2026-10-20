import numpy as np
import pandas as pd

from asthunt.evaluate.measure_completeness import measure_completeness


def test_fraction_found_per_bin():
    table = pd.DataFrame({"mag": [17.1, 17.2, 17.3, 17.4, 18.5, 18.6], "found": [True, True, True, False, False, False]})
    result = measure_completeness(table, "found", "mag", bin_edges=[17, 18, 19])

    assert result["count"].tolist() == [4, 2]
    assert result["completeness"].tolist() == [0.75, 0.0]
    assert np.isclose(result["uncertainty"][0], np.sqrt(0.75 * 0.25 / 4))
