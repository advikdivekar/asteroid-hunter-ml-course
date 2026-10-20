import numpy as np
import pandas as pd
from astropy.wcs import WCS

from asthunt.detect.cut_tracklet_stamps import cut_tracklet_stamps


def simple_wcs(shift):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [135.6, 22.8]
    wcs.wcs.crpix = [50.0 + shift, 50.0 + shift]
    wcs.wcs.cd = [[-1 / 3600, 0], [0, 1 / 3600]]
    return wcs


def test_stamps_follow_the_predicted_path_through_pointing_shifts():
    times = [2459000.0, 2459000.0 + 1 / 24]                 # 0 and 1 hour
    shifts = [0.0, 7.0]                                     # the pointing moves 7 px for the second frame
    images = {}
    for name, shift in zip(["f0", "f1"], shifts):
        difference = np.zeros((100, 100))
        images[name] = {"science": difference.copy(), "reference": np.zeros((100, 100)), "difference": difference,
                        "wcs": simple_wcs(shift)}
    # Object at common (30, 40) at t0, moving +10 px/hour in x. In frame 1 it sits at common (40, 40) = own pixel (47, 47)
    images["f0"]["difference"][40, 30] = 5.0
    images["f1"]["difference"][47, 47] = 5.0
    tracklets = pd.DataFrame({"x0": [30.0], "y0": [40.0], "vx": [10.0], "vy": [0.0], "t0_jd": [times[0]]})

    stamps = cut_tracklet_stamps(tracklets, list(zip(["f0", "f1"], times)), images["f0"]["wcs"], images, size=11)

    assert stamps.shape == (1, 2, 3, 11, 11)
    assert stamps[0, 0, 2, 5, 5] == 5.0 and stamps[0, 1, 2, 5, 5] == 5.0   # centred in both frames
