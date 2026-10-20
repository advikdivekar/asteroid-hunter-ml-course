import pandas as pd

from asthunt.acquire.download_ztf_images import _build_url, download_ztf_images

EXPOSURE = pd.Series({
    "filefracday": 20190208210347, "field": 618, "filtercode": "zr",
    "ccdid": 3, "imgtypecode": "o", "qid": 4, "night": "2019-02-08",
})


def test_url_follows_ztf_layout():
    assert _build_url(EXPOSURE, "sciimg.fits") == (
        "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2019/0208/210347/"
        "ztf_20190208210347_000618_zr_c03_o_q4_sciimg.fits"
    )


def test_existing_files_are_not_downloaded_again(tmp_path):
    existing = tmp_path / "2019-02-08" / "ztf_20190208210347_000618_zr_c03_o_q4_sciimg.fits"
    existing.parent.mkdir()
    existing.write_text("already here")

    # No network is needed: the file exists, so it must be skipped.
    paths = download_ztf_images(pd.DataFrame([EXPOSURE]), ["sciimg.fits"], tmp_path)

    assert paths == [existing]
    assert existing.read_text() == "already here"


class _NotFound:
    status_code = 404

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_a_product_missing_from_the_archive_skips_the_exposure_for_good(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("asthunt.acquire.download_ztf_images.requests.get", lambda url, **kw: calls.append(url) or _NotFound())

    first = download_ztf_images(pd.DataFrame([EXPOSURE]), ["diffimgpsf.fits", "sciimg.fits"], tmp_path)
    second = download_ztf_images(pd.DataFrame([EXPOSURE]), ["diffimgpsf.fits", "sciimg.fits"], tmp_path)

    assert first == [] and second == []
    assert len(calls) == 1                                    # no retries, no second product, nothing on the re-run
    missing = pd.read_csv(tmp_path / "missing_exposures.csv")
    assert missing["filefracday"].tolist() == [20190208210347]
