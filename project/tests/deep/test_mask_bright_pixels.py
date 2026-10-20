import numpy as np

from asthunt.deep.mask_bright_pixels import mask_bright_pixels


def test_bright_spot_and_border_removed_faint_signal_kept():
    snr = np.zeros((1, 30, 30), dtype=np.float32)
    snr[0, 10, 10] = 40.0                       # bright
    snr[0, 20, 20] = 3.0                        # faint: must stay
    good = np.ones(snr.shape, dtype=bool)

    masked, good_after = mask_bright_pixels(snr, good, sigma=5.0, radius_pixels=2)

    assert masked[0, 10, 10] == 0 and not good_after[0, 12, 12] and good_after[0, 13, 13]
    assert masked[0, 20, 20] == 3.0 and good_after[0, 20, 20]
    assert snr[0, 10, 10] == 40.0               # the input is not modified
