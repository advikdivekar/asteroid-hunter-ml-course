"""Collect stamps and labels from several nights into one training-ready set."""

import numpy as np
import pandas as pd

from asthunt.dataset.balance_classes import balance_classes
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.dataset.normalize_stamps import normalize_stamps


def assemble_stamp_dataset(units, processed_dir, rng, bogus_per_real=None):
    """Return (stamps, labels, detections) for the given units (CCD quadrants on nights).

    stamps     : float32 (n, 3, size, size), normalized
    labels     : int (n,) 1 = real, 0 = bogus
    detections : the matching rows of the detection tables

    bogus_per_real : None keeps every detection (use for validation/test,
                     so the class balance is realistic); a number balances
                     the classes (use for training).
    """
    all_stamps, all_rows = [], []
    for unit in units:
        tag = make_unit_tag(unit)
        rows = pd.read_parquet(processed_dir / f"detections__{tag}.parquet")
        stamps = np.load(processed_dir / f"stamps__{tag}.npy")

        if bogus_per_real is not None:
            keep = balance_classes(rows["label"], bogus_per_real, rng)
            rows, stamps = rows.iloc[keep], stamps[keep]

        all_rows.append(rows)
        all_stamps.append(normalize_stamps(stamps))

    detections = pd.concat(all_rows, ignore_index=True)
    return np.concatenate(all_stamps), detections["label"].to_numpy(), detections
