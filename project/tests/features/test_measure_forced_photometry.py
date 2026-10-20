import numpy as np

from asthunt.features.measure_forced_photometry import measure_forced_photometry


def test_stack_snr_grows_with_frames_and_min_finds_the_empty_frame():
    stamps = np.zeros((2, 4, 3, 21, 21), dtype=np.float32)
    stamps[0, :, 2, 10, 10] = 3.0             # tracklet 0: the same faint signal in all 4 frames
    stamps[1, :3, 2, 10, 10] = 3.0            # tracklet 1: nothing in the last frame

    table = measure_forced_photometry(stamps)
    one_frame_snr = 3.0 / np.sqrt(13)         # a radius-2 circle covers 13 pixels

    assert np.isclose(table.loc[0, "stack_snr"], 4 * one_frame_snr / 2)     # sum of 4 / sqrt(4) = 2x one frame
    assert np.isclose(table.loc[0, "forced_snr_min"], one_frame_snr)
    assert table.loc[1, "forced_snr_min"] == 0.0
    assert table.loc[1, "stack_snr"] < table.loc[0, "stack_snr"]
