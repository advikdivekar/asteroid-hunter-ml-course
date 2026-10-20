"""Which nights on disk can the live viewer run? Those with all four products of enough frames and a cached archive table."""

import re
from collections import defaultdict
from pathlib import Path

FILE_PATTERN = re.compile(r"ztf_(\d+)_(\d{6})_(\w+)_c(\d{2})_o_q(\d)_sciimg\.fits$")
PRODUCTS = ["mskimg.fits", "scimrefdiffimg.fits.fz", "diffimgpsf.fits"]


def list_viewer_units(raw_dir, filter_code, min_frames):
    """Return [{field, ccd, quadrant, night, frames}] sorted by night; only units whose files are all on disk, no network needed."""
    counts = defaultdict(int)
    for science in sorted((raw_dir / "ztf").glob("*/ztf_*_sciimg.fits")):
        match = FILE_PATTERN.search(science.name)
        if not match or match.group(3) != filter_code:
            continue
        prefix = str(science).replace("sciimg.fits", "")
        if all(Path(prefix + product).exists() for product in PRODUCTS):
            counts[(int(match.group(2)), int(match.group(4)), int(match.group(5)), science.parent.name)] += 1
    units = []
    for (field, ccd, quadrant, night), frames in counts.items():
        cached_table = raw_dir / "ztf" / f"metadata__{field:06d}_c{ccd:02d}_q{quadrant}.csv"
        if frames >= min_frames and cached_table.exists():
            units.append({"field": field, "ccd": ccd, "quadrant": quadrant, "night": night, "frames": frames})
    return sorted(units, key=lambda u: (u["night"], u["field"], u["ccd"], u["quadrant"]))
