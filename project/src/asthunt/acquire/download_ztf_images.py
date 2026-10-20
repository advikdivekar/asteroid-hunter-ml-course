"""Download ZTF image products for a list of exposures."""

import time

import pandas as pd
import requests

IRSA_DATA_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"


def download_ztf_images(exposures, products, out_dir):
    """Download every product for every exposure; return the list of local paths.

    exposures : table rows from query_ztf_metadata (needs filefracday, field,
                filtercode, ccdid, imgtypecode, qid, night)
    products  : file endings, e.g. ["sciimg.fits", "mskimg.fits"]
    out_dir   : files are saved as out_dir/<night>/<ZTF file name>

    Files that already exist are skipped, so the function can be re-run
    after an interruption. Each file is first written as *.part and only
    renamed when complete, so a half-downloaded file never looks finished.
    If the archive is briefly unavailable, each file is retried a few times.

    Some exposures lack a product for good (e.g. ZTF's image subtraction
    failed, so there is no difference image): the archive answers 404. Such an
    exposure is written to out_dir/missing_exposures.csv and skipped, here and
    by load_unit_metadata, so no search ever tries to open it.
    """
    missing = _read_missing(out_dir)
    paths = []
    for _, exposure in exposures.iterrows():
        if _exposure_key(exposure) in missing:
            continue
        exposure_paths = []
        for product in products:
            url = _build_url(exposure, product)
            path = out_dir / exposure["night"] / url.split("/")[-1]
            if not path.exists() and not _download_file(url, path):
                _record_missing(out_dir, exposure, product)
                print(f"  {url.split('/')[-1]} does not exist in the archive: exposure skipped")
                break
            exposure_paths.append(path)
        else:
            paths += exposure_paths
    return paths


def _exposure_key(exposure):
    return (int(exposure["field"]), int(exposure["ccdid"]), int(exposure["qid"]), int(exposure["filefracday"]))


def _read_missing(out_dir):
    file = out_dir / "missing_exposures.csv"
    if not file.exists():
        return set()
    table = pd.read_csv(file)
    return {_exposure_key(row) for _, row in table.iterrows()}


def _record_missing(out_dir, exposure, product):
    file = out_dir / "missing_exposures.csv"
    row = pd.DataFrame([{"field": int(exposure["field"]), "ccdid": int(exposure["ccdid"]), "qid": int(exposure["qid"]),
                         "filefracday": int(exposure["filefracday"]), "product": product}])
    file.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(file, mode="a", header=not file.exists(), index=False)


def _build_url(exposure, product):
    """Build the archive URL, following ZTF's documented folder layout.

    Example: .../sci/2019/0208/210347/ztf_20190208210347_000618_zr_c03_o_q4_sciimg.fits
    """
    fracday = str(exposure["filefracday"])           # e.g. 20190208210347
    year, month_day, fraction = fracday[:4], fracday[4:8], fracday[8:]
    file_name = (
        f"ztf_{fracday}_{int(exposure['field']):06d}_{exposure['filtercode']}"
        f"_c{int(exposure['ccdid']):02d}_{exposure['imgtypecode']}_q{int(exposure['qid'])}_{product}"
    )
    return f"{IRSA_DATA_URL}/{year}/{month_day}/{fraction}/{file_name}"


def _download_file(url, path, attempts=5, wait_seconds=20):
    """Download url to path; return False if the archive says the file doesn't exist (404)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")

    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as response:
                if response.status_code == 404:              # not "busy": it will never exist
                    return False
                response.raise_for_status()
                with open(temporary, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            temporary.rename(path)
            return True
        except requests.RequestException as error:
            # Busy archives often answer "503: try again later". Wait longer each time.
            if attempt == attempts:
                raise
            print(f"  download failed ({error}); retry {attempt}/{attempts - 1} in {wait_seconds * attempt}s")
            time.sleep(wait_seconds * attempt)
