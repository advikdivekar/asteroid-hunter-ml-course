"""Download the deep reference image for one field/filter/CCD/quadrant."""

import requests

IRSA_REF_URL = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref"


def download_ztf_reference_image(field, filter_code, ccd, quadrant, out_dir):
    """Return the local path of the reference image, downloading it if needed.

    The reference image is a stack of many older exposures: deep and clean,
    with no moving objects. Difference images are science minus reference.
    """
    field_text = f"{int(field):06d}"
    file_name = f"ztf_{field_text}_{filter_code}_c{int(ccd):02d}_q{int(quadrant)}_refimg.fits"
    url = (
        f"{IRSA_REF_URL}/{field_text[:3]}/field{field_text}/{filter_code}"
        f"/ccd{int(ccd):02d}/q{int(quadrant)}/{file_name}"
    )

    path = out_dir / "reference" / file_name
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    response = requests.get(url, timeout=600)
    response.raise_for_status()
    temporary.write_bytes(response.content)
    temporary.rename(path)
    return path
