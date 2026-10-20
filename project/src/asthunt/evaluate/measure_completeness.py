"""Fraction of injected objects recovered, per bin of some property (e.g. magnitude)."""

import numpy as np
import pandas as pd


def measure_completeness(injections, found_column, bin_column, bin_edges):
    """Return a table with one row per bin: count, completeness, and its uncertainty.

    injections   : one row per injected object (or per injected detection)
    found_column : name of a True/False column, True if it was recovered
    bin_column   : the property to bin by, e.g. "mag"
    bin_edges    : e.g. np.arange(17, 22.25, 0.25)

    Uncertainty of a fraction p from n objects: sqrt(p * (1 - p) / n).
    """
    bins = pd.cut(injections[bin_column], bin_edges)
    grouped = injections.groupby(bins, observed=True)[found_column]

    table = pd.DataFrame({
        "bin_centre": [interval.mid for interval in grouped.size().index],
        "count": grouped.size().to_numpy(),
        "completeness": grouped.mean().to_numpy(),
    })
    p, n = table["completeness"], table["count"]
    table["uncertainty"] = np.sqrt(p * (1 - p) / n)
    return table
