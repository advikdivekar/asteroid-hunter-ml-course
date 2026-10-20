"""Decide where fake asteroids go: random brightness, speed and direction."""

import numpy as np
import pandas as pd

COLUMNS = ["injection_id", "frame_index", "x", "y", "mag", "rate_arcsec_per_hour", "angle_degrees"]


def plan_injections(n_asteroids, image_shape, frame_times_jd, magnitude_range,
                    rate_range_arcsec_per_hour, pixel_scale, edge_margin, rng):
    """Return a table with one row per (fake asteroid, frame).

    Each fake asteroid gets:
      - a magnitude drawn uniformly from magnitude_range (uniform, so every
        brightness is equally represented when we measure completeness),
      - a speed drawn uniformly from rate_range ("/hr) and a random direction,
      - a random starting position at the time of the FIRST frame.
    Its position in later frames follows a straight line at that speed.

    Columns: injection_id, frame_index, x, y, mag, rate_arcsec_per_hour, angle_degrees
    """
    height, width = image_shape
    start_time = frame_times_jd[0]

    rows = []
    for injection_id in range(n_asteroids):
        mag = rng.uniform(*magnitude_range)
        rate = rng.uniform(*rate_range_arcsec_per_hour)
        angle = rng.uniform(0, 360)
        x_start = rng.uniform(edge_margin, width - edge_margin)
        y_start = rng.uniform(edge_margin, height - edge_margin)

        # Speed in pixels per hour, split into x and y parts
        pixels_per_hour = rate / pixel_scale
        vx = pixels_per_hour * np.cos(np.radians(angle))
        vy = pixels_per_hour * np.sin(np.radians(angle))

        for frame_index, time_jd in enumerate(frame_times_jd):
            hours = (time_jd - start_time) * 24
            rows.append({
                "injection_id": injection_id,
                "frame_index": frame_index,
                "x": x_start + vx * hours,
                "y": y_start + vy * hours,
                "mag": mag,
                "rate_arcsec_per_hour": rate,
                "angle_degrees": angle,
            })

    return pd.DataFrame(rows, columns=COLUMNS)      # columns are kept even when there are no fakes
