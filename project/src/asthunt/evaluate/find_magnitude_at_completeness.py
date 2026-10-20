"""Find the magnitude where completeness drops to a given level (e.g. 50%)."""

import numpy as np


def find_magnitude_at_completeness(completeness_table, level=0.5):
    """Return the magnitude where completeness first falls below `level`, interpolated.

    completeness_table: output of measure_completeness (bin_centre, completeness),
                        bins ordered from bright to faint.

    We walk from bright to faint and, at the first bin below `level`, draw a
    straight line between it and the previous bin to estimate the crossing.
    Example: 0.79 at mag 20.25 and 0.29 at 20.75 -> 50% at about 20.54.
    Returns NaN if completeness never drops below `level`.
    """
    magnitudes = completeness_table["bin_centre"].to_numpy(dtype=float)
    values = completeness_table["completeness"].to_numpy(dtype=float)

    for i in range(1, len(values)):
        if values[i] < level <= values[i - 1]:
            fraction = (values[i - 1] - level) / (values[i - 1] - values[i])
            return magnitudes[i - 1] + fraction * (magnitudes[i] - magnitudes[i - 1])
    return float("nan")
