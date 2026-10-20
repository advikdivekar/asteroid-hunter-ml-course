"""Ask the IRSA archive for every exposure of one ZTF field on one night (all CCDs and quadrants)."""

import io
import time

import pandas as pd
import requests
from astropy.time import Time

from asthunt.acquire.query_ztf_metadata import IRSA_SEARCH_URL


def query_ztf_field_night(field, night, cache_file=None, attempts=4):
    """Return one row per (exposure, CCD, quadrant) of `field` on `night` (YYYY-MM-DD, UTC).

    Each row has the quadrant's four corners (ra1..ra4, dec1..dec4), so we
    can test which quadrant saw a given sky position (find_quadrants_covering_position).
    Querying by field + time is fast for the archive; position searches
    time out when it's busy. Cached like query_ztf_metadata.
    """
    if cache_file is not None and cache_file.exists():
        return pd.read_csv(cache_file)

    start = Time(f"{night}T00:00:00", scale="utc").jd
    where = f"field={field} AND obsjd>{start} AND obsjd<{start + 1}"
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(IRSA_SEARCH_URL, params={"WHERE": where, "ct": "csv"}, timeout=300)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == attempts:
                raise
            time.sleep(20 * attempt)

    metadata = pd.read_csv(io.StringIO(response.text))
    metadata["night"] = metadata["obsdate"].str[:10]
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(cache_file, index=False)
    return metadata
