"""Which CCD quadrants saw a given sky position?"""

import numpy as np
from matplotlib.path import Path


def find_quadrants_covering_position(metadata, ra, dec, margin_degrees=0.0):
    """Return the rows of `metadata` whose quadrant contains (ra, dec), in degrees.

    metadata needs the corner columns ra1..ra4, dec1..dec4 and the centre ra, dec.
    Each quadrant is ~0.87 deg across, so a flat projection around its centre is
    accurate enough. margin_degrees > 0 also accepts positions just outside the
    edge (useful when the predicted position itself is uncertain).
    """
    keep = []
    for _, row in metadata.iterrows():
        cos_dec = np.cos(np.radians(row["dec"]))
        # Offsets from the quadrant centre; RA wraps at 360 so take the short way round
        corners = [(((row[f"ra{k}"] - row["ra"] + 180) % 360 - 180) * cos_dec, row[f"dec{k}"] - row["dec"]) for k in range(1, 5)]
        point = (((ra - row["ra"] + 180) % 360 - 180) * cos_dec, dec - row["dec"])
        corners = sorted(corners, key=lambda c: np.arctan2(c[1], c[0]))     # put the corners in order around the centre
        keep.append(Path(corners).contains_point(point, radius=margin_degrees) or
                    Path(corners).contains_point(point, radius=-margin_degrees))
    return metadata[np.array(keep, dtype=bool)]
