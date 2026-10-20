"""Read the known-asteroid table for one exposure (the single place that does this)."""

import pandas as pd


def load_known_asteroids(exposure, interim_dir):
    """Return name, ra, dec, v_mag, x, y, source of every known object in this exposure.

    - Objects refined with JPL Horizons (accurate to ~0.3 px for our data) use
      those positions: source = "horizons".
    - Every other object SkyBoT lists keeps SkyBoT's position (~2.7 px off):
      source = "skybot". We keep them so a detected faint known object is
      never mistaken for a new one.
    """
    folder = interim_dir / "known_asteroids"
    read = lambda path: pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"name": str})

    skybot = read(folder / f"known_asteroids__{exposure}.csv").assign(source="skybot")
    refined_file = folder / f"known_asteroids_horizons__{exposure}.csv"
    if not refined_file.exists():
        return skybot

    horizons = read(refined_file).assign(source="horizons")
    not_refined = skybot[~skybot["name"].isin(horizons["name"])]
    return pd.concat([horizons, not_refined], ignore_index=True)
