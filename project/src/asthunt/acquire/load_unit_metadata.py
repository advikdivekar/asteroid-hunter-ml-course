"""Exposure metadata for a unit's field/CCD/quadrant, cached under one consistent name."""

import pandas as pd

from asthunt.acquire.query_ztf_metadata import query_ztf_metadata


def load_unit_metadata(unit, raw_dir):
    """Return the archive's exposure table for unit = {field, ccd, quadrant, ...}.

    Cached as raw_dir/ztf/metadata__<field>_c<ccd>_q<quadrant>.csv, so every
    quadrant has its own file and re-runs never hit the network.
    Exposures listed in raw_dir/ztf/missing_exposures.csv (a product doesn't
    exist in the archive, see download_ztf_images) are left out.
    """
    field, ccd, quadrant = int(unit["field"]), int(unit["ccd"]), int(unit["quadrant"])
    cache_file = raw_dir / "ztf" / f"metadata__{field:06d}_c{ccd:02d}_q{quadrant}.csv"
    metadata = query_ztf_metadata(field, ccd, quadrant, cache_file=cache_file)

    missing_file = raw_dir / "ztf" / "missing_exposures.csv"
    if missing_file.exists():
        missing = pd.read_csv(missing_file)
        missing = missing[(missing["field"] == field) & (missing["ccdid"] == ccd) & (missing["qid"] == quadrant)]
        metadata = metadata[~metadata["filefracday"].isin(missing["filefracday"])].reset_index(drop=True)
    return metadata
