"""Ask the IRSA archive which ZTF exposures exist for one field/CCD/quadrant."""

import io
import time

import pandas as pd
import requests

IRSA_SEARCH_URL = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"


def query_ztf_metadata(field, ccd, quadrant, cache_file=None, attempts=4):
    """Return a table with one row per exposure (time, filter, seeing, ...).

    The archive is queried only once: the answer is saved to `cache_file`
    and read from there next time. This keeps reruns fast and means the
    exact list we worked with is kept on disk.

    An extra column `night` (YYYY-MM-DD, UTC) is added for grouping.
    A busy archive sometimes drops the connection; we wait and ask again.
    """
    if cache_file is not None and cache_file.exists():
        return pd.read_csv(cache_file)

    where = f"field={field} AND ccdid={ccd} AND qid={quadrant}"
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
    metadata = metadata.sort_values("obsjd").reset_index(drop=True)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(cache_file, index=False)
    return metadata
