"""The random number generator for one unit: different per unit, identical on every re-run."""

import numpy as np

ORIGINAL_QUADRANT = (618, 3, 4)


def make_unit_rng(seed, unit):
    """Return np.random.Generator seeded from the project seed and the unit.

    - Every unit gets its own fakes (different positions, brightnesses, speeds).
    - Re-running gives exactly the same fakes, so builds are reproducible.
    The original quadrant (field 618, CCD 3, quadrant 4) keeps round 1's
    seed [seed, night date], so the files built back then stay valid.
    """
    night_number = int(unit["night"].replace("-", ""))
    if (int(unit["field"]), int(unit["ccd"]), int(unit["quadrant"])) == ORIGINAL_QUADRANT:
        return np.random.default_rng([seed, night_number])
    return np.random.default_rng([seed, night_number, int(unit["field"]), int(unit["ccd"]), int(unit["quadrant"])])
