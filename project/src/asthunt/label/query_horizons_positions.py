"""Ask JPL Horizons where a known object was at several times, as seen from ZTF."""

import time

import pandas as pd
from astroquery.jplhorizons import Horizons


def query_horizons_positions(name, times_jd, observatory_code="I41", attempts=3):
    """Return a table (name, jd, ra, dec, v_mag) with one row per requested time,
    or an empty table if Horizons can't identify the object.

    Why Horizons and not SkyBoT? We measured SkyBoT's predictions sitting
    ~2.9 arcsec away from where ZTF actually saw the asteroids, while JPL
    Horizons agreed with our detections to ~0.2 arcsec. Horizons computes one
    object at a time, so we only ask it about objects SkyBoT says are nearby.

    times_jd should be MID-exposure times (shutter open + half the exposure).
    """
    for attempt in range(1, attempts + 1):
        try:
            query = Horizons(id=name, location=observatory_code, epochs=list(times_jd), id_type="smallbody")
            table = query.ephemerides(quantities="1,9")
            return pd.DataFrame({
                "name": name,
                "jd": list(times_jd),
                "ra": [float(v) for v in table["RA"]],
                "dec": [float(v) for v in table["DEC"]],
                "v_mag": _brightness(table),
            })
        except ValueError:
            # Unknown or ambiguous name (Horizons asks "which one?"): we can't use it
            return pd.DataFrame(columns=["name", "jd", "ra", "dec", "v_mag"])
        except Exception:
            # Network hiccup or server busy: wait and try again
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def _brightness(table):
    """Predicted magnitude per row. Asteroids report V; comets report a total magnitude Tmag instead."""
    for column in ("V", "Tmag"):
        if column in table.colnames:
            return [float(v) if v is not None and str(v) != "--" else float("nan") for v in table[column]]
    return [float("nan")] * len(table)
