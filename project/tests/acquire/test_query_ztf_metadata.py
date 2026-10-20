import pandas as pd

from asthunt.acquire.query_ztf_metadata import query_ztf_metadata


def test_reads_from_cache_without_network(tmp_path):
    cache = tmp_path / "metadata.csv"
    pd.DataFrame({"obsjd": [1.0], "night": ["2019-02-08"]}).to_csv(cache, index=False)

    metadata = query_ztf_metadata(618, 3, 4, cache_file=cache)

    assert metadata["night"].tolist() == ["2019-02-08"]
