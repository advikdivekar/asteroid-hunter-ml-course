"""Check deep-search tracklets against known asteroids."""

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from asthunt.clean.read_exposure_header import read_exposure_header
from asthunt.crossmatch.crossmatch_known_asteroids import crossmatch_known_asteroids
from asthunt.label.load_known_asteroids import load_known_asteroids
from asthunt.label.query_known_asteroids import query_known_asteroids
from asthunt.label.refine_known_positions import refine_known_positions


def crossmatch_deep_tracklets(deep_result, unit, paths, max_v_mag=23.0):
    """Return the deep tracklets with known_name / known_matches columns added.

    Each stack detection is compared with the known objects of that stack's
    MIDDLE exposure (SkyBoT list, positions refined with JPL Horizons down to
    max_v_mag: the deep search reaches fainter than the regular pipeline).
    """
    stacks, detections = deep_result["stacks"], deep_result["stack_detections"].copy()
    folder = paths["raw"] / "ztf" / unit["night"]
    science = [folder / f"{exposure}_sciimg.fits" for exposure in stacks["middle_exposure"]]

    headers = {}
    for path in science:
        header, shape = read_exposure_header(path)          # works without the science image too (light mode)
        exposure = path.name.replace("_sciimg.fits", "")
        query_known_asteroids(header, shape,
                              cache_file=paths["interim"] / "known_asteroids" / f"known_asteroids__{exposure}.csv")
        headers[exposure] = header
    refine_known_positions(science, paths["interim"], max_v_mag=max_v_mag)

    # Stack positions (sky) -> pixels of each stack's middle exposure
    detections["exposure"] = detections["frame_index"].map(stacks.set_index("stack")["middle_exposure"])
    for exposure, rows in detections.groupby("exposure"):
        sky = SkyCoord(rows["ra"].to_numpy(), rows["dec"].to_numpy(), unit="deg")
        detections.loc[rows.index, "x"], detections.loc[rows.index, "y"] = WCS(headers[exposure]).world_to_pixel(sky)

    known = {exposure: load_known_asteroids(exposure, paths["interim"]) for exposure in headers}
    if len(deep_result["tracklets"]) == 0:
        return deep_result["tracklets"].assign(known_name=pd.Series(dtype=str), known_matches=pd.Series(dtype=int))
    return crossmatch_known_asteroids(deep_result["tracklets"], deep_result["members"], detections, known)
