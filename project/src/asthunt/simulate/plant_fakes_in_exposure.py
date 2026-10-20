"""Plant this frame's fakes into one exposure's images (positions converted via the sky)."""

from asthunt.simulate.inject_point_source import inject_point_source


def plant_fakes_in_exposure(exposure, science, difference, fakes, reference_wcs, frame_wcs, rng):
    """Add fakes to `science` and `difference` IN PLACE; return the fakes table in THIS frame's pixels.

    fakes: rows of plan_injections for this frame, with x, y on the REFERENCE
    grid (frame 0's pixels). The telescope pointing shifts between exposures,
    so each position goes reference pixels -> sky (RA, Dec) -> this frame's
    pixels. That keeps fakes moving in straight lines on the sky, like real asteroids.
    """
    fakes = fakes.copy()
    if len(fakes) == 0:
        return fakes
    sky = reference_wcs.pixel_to_world(fakes["x"].to_numpy(), fakes["y"].to_numpy())
    fakes["x"], fakes["y"] = frame_wcs.world_to_pixel(sky)
    for _, fake in fakes.iterrows():
        inject_point_source(science, difference, exposure["psf"], fake["mag"], exposure["magzp"],
                            exposure["gain"], fake["x"], fake["y"], rng)
    return fakes
