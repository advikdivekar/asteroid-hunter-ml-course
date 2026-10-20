"""Pick the exposures of one night that we will link into tracklets."""


def select_night_exposures(metadata, night, filter_code, max_frames):
    """Return up to `max_frames` consecutive good exposures from one night.

    - Only the requested filter (asteroids look different in each colour,
      and difference images are made per filter).
    - Only exposures with infobits == 0: ZTF sets these flags when an
      image has a known quality problem.
    - The first `max_frames` in time order, so the frames are consecutive.
    """
    rows = metadata[
        (metadata["night"] == night)
        & (metadata["filtercode"] == filter_code)
        & (metadata["infobits"] == 0)
    ]
    rows = rows.sort_values("obsjd")
    return rows.head(max_frames).reset_index(drop=True)
