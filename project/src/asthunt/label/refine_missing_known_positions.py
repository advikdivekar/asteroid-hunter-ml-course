"""Make sure every exposure has Horizons-refined known-asteroid positions, asking Horizons only for the ones that don't."""

from asthunt.label.refine_known_positions import refine_known_positions


def refine_missing_known_positions(science_paths, interim_dir, max_v_mag=22.0):
    """Refine the exposures that have no known_asteroids_horizons__<exposure>.csv yet; return how many objects were refined.

    SkyBoT's raw positions sit 4-5 px from where ZTF sees the asteroids on some nights,
    farther than the 4 px cross-match radius: a known asteroid then looks like a discovery.
    Training nights were refined once in the label_quality notebook; a fresh night
    (hunt, live monitor) is refined here, before the cross-match. Nights that are
    already done cost nothing.
    """
    missing = [path for path in science_paths
               if not (interim_dir / "known_asteroids" / f"known_asteroids_horizons__{path.name.replace('_sciimg.fits', '')}.csv").exists()]
    if not missing:
        return 0
    return refine_known_positions(missing, interim_dir, max_v_mag)
