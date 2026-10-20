"""Load the images of every frame of a night (with planted fakes, or straight from the archive files)."""

import numpy as np
from astropy.wcs import WCS

from asthunt.clean.load_exposure import load_exposure
from asthunt.dataset.make_unit_tag import make_unit_tag


def load_night_images(unit, exposure_names, paths, bad_mask_bits, with_injections):
    """Return {exposure name: {"science", "difference", "reference", "wcs"}} for the given frames of one unit.

    with_injections=True : the images WITH our planted fakes, saved by
                           build_night_detections (the evaluation/training run)
    with_injections=False: the original archive images (the real search)

    Everything a stage needs to cut stamps anywhere in the frame, including
    where nothing was detected (e.g. the predicted position of a faint asteroid).
    """
    images = {}
    planted_dir = paths["interim"] / "night_images" / make_unit_tag(unit)
    for name in exposure_names:
        exposure = load_exposure(paths["raw"] / "ztf" / unit["night"] / f"{name}_sciimg.fits", bad_mask_bits)
        science, difference = exposure["science"], exposure["difference"]
        if with_injections:
            science = np.load(planted_dir / f"{name}__science.npy").astype(np.float64)
            difference = np.load(planted_dir / f"{name}__difference.npy").astype(np.float64)
        images[name] = {
            "science": science,
            "difference": difference,
            "reference": exposure["reference"],          # fakes are never in the reference
            "wcs": WCS(exposure["header"]),
        }
    return images
