"""Replace SkyBoT's known-asteroid positions with JPL Horizons positions for one night."""

import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from asthunt.clean.read_exposure_header import read_exposure_header
from asthunt.label.query_horizons_positions import query_horizons_positions

EXPOSURE_SECONDS = 30.0


def refine_known_positions(science_paths, interim_dir, max_v_mag=22.0):
    """Write known_asteroids_horizons__<exposure>.csv for every exposure; return how many objects were refined.

    science_paths : the science images of ONE night (their SkyBoT caches must exist)

    1. Collect every object SkyBoT lists in any frame, brighter than max_v_mag
       (fainter ones can't be detected, so asking about them wastes time).
    2. Ask Horizons once per object for all frames' mid-exposure times.
    3. Convert those sky positions to each frame's pixels with its own WCS.
    Objects Horizons can't identify keep no refined position (they are left out).
    """
    headers = {path.name.replace("_sciimg.fits", ""): read_exposure_header(path)[0] for path in science_paths}
    skybot = {exposure: pd.read_csv(interim_dir / "known_asteroids" / f"known_asteroids__{exposure}.csv", dtype={"name": str})
              for exposure in headers}

    names = sorted({name for table in skybot.values() for name in table.loc[table["v_mag"] <= max_v_mag, "name"]})
    mid_times = {exposure: header["OBSJD"] + EXPOSURE_SECONDS / 2 / 86400 for exposure, header in headers.items()}
    positions = pd.concat([query_horizons_positions(name, list(mid_times.values())) for name in names] or [pd.DataFrame()],
                          ignore_index=True)

    for exposure, header in headers.items():
        rows = positions[positions["jd"] == mid_times[exposure]] if len(positions) else positions
        table = pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"])
        if len(rows):
            x, y = WCS(header).world_to_pixel(SkyCoord(rows["ra"].to_numpy(), rows["dec"].to_numpy(), unit="deg"))
            table = pd.DataFrame({"name": rows["name"].to_numpy(), "ra": rows["ra"].to_numpy(), "dec": rows["dec"].to_numpy(),
                                  "v_mag": rows["v_mag"].to_numpy(), "x": x, "y": y})
            inside = table["x"].between(0, header["NAXIS1"] - 1) & table["y"].between(0, header["NAXIS2"] - 1)
            table = table[inside]
        table.to_csv(interim_dir / "known_asteroids" / f"known_asteroids_horizons__{exposure}.csv", index=False)

    return len(positions["name"].unique()) if len(positions) else 0
