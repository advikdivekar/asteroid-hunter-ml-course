"""Ask SkyBoT which known asteroids were inside an image when it was taken."""

import time

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.imcce import Skybot

EMPTY_TABLE_COLUMNS = ["name", "ra", "dec", "v_mag", "x", "y"]


def query_known_asteroids(header, image_shape, cache_file=None, search_radius_degrees=0.65, attempts=4):
    """Return a table of known solar-system objects inside the image.

    SkyBoT (IMCCE, Paris) computes the position of every known asteroid
    and comet at a given time. We ask for everything within
    `search_radius_degrees` of the image centre at the exposure time,
    convert each sky position to pixels with the image's WCS, and keep
    those that land on the image.

    Columns: name, ra, dec, v_mag (predicted brightness), x, y

    - The answer is saved to `cache_file` (CSV) and re-used next time.
    - SkyBoT sometimes fails for one exact time stamp (a server-side bug);
      asking again for a time 1 second later works. In 1 second an
      asteroid moves less than 0.03 arcsec, so the answer is unchanged.
    """
    if cache_file is not None and cache_file.exists():
        return pd.read_csv(cache_file, keep_default_na=False, na_values=[""],
                           dtype={"name": str})

    wcs = WCS(header)
    height, width = image_shape
    centre = wcs.pixel_to_world(width / 2, height / 2)

    result = None
    for attempt in range(attempts):
        observation_time = Time(header["OBSJD"] + attempt / 86400, format="jd")   # + 0, 1, 2 ... seconds
        try:
            # location "I41" = ZTF at Palomar. The default is Earth's centre, which shifts
            # a main-belt asteroid's predicted position by up to ~9 arcsec (parallax).
            result = Skybot.cone_search(centre, search_radius_degrees * u.deg, observation_time, location="I41")
            break
        except RuntimeError:
            # SkyBoT raises RuntimeError when no object is in the field
            result = None
            break
        except ValueError:
            # "No table found": the server failed for this time stamp
            if attempt == attempts - 1:
                raise
            time.sleep(5)

    if result is None:
        table = pd.DataFrame(columns=EMPTY_TABLE_COLUMNS)
    else:
        sky_positions = SkyCoord(result["RA"], result["DEC"])
        x, y = wcs.world_to_pixel(sky_positions)
        table = pd.DataFrame({
            "name": [str(name) for name in result["Name"]],
            "ra": sky_positions.ra.deg,
            "dec": sky_positions.dec.deg,
            "v_mag": _as_float_array(result["V"]),
            "x": x,
            "y": y,
        })
        table = _keep_inside_image(table, image_shape)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_file, index=False)
    return table


def _as_float_array(column):
    """Turn an astropy column (with or without units, maybe with gaps) into floats; gaps become NaN."""
    values = getattr(column, "value", column)
    return np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)


def _keep_inside_image(table, image_shape):
    height, width = image_shape
    inside = table["x"].between(0, width - 1) & table["y"].between(0, height - 1)
    return table[inside].reset_index(drop=True)
