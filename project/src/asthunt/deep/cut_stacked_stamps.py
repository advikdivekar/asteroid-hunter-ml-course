"""Stamps of a deep tracklet: for every stack, the frames added up along the tracklet's motion."""

import numpy as np


def cut_stacked_stamps(snr_maps, good_maps, times_jd, offsets, stack_rows, tracklet, size=31):
    """Return an array (n_stacks, size, size): stacked SNR around the tracklet's predicted position.

    snr_maps, good_maps, times_jd, offsets : the maps used by the deep search (one per frame)
    stack_rows : list of lists, the frame rows of each stack
    tracklet   : a row with x0, y0, vx, vy, t0_jd (reference grid, pixels/hour)

    Each frame is cut around where the straight line puts the object at that
    frame's time (plus the frame's pointing offset), and the frames of a stack
    are averaged with the same sqrt(N) scaling as the search, so a stamp shows
    "how many sigma" a human would see in that stack.
    """
    half = size // 2
    stamps = np.zeros((len(stack_rows), size, size), dtype=np.float32)
    height, width = snr_maps.shape[1:]
    for s, rows in enumerate(stack_rows):
        total, count = np.zeros((size, size)), np.zeros((size, size))
        for k in rows:
            hours = (times_jd[k] - tracklet["t0_jd"]) * 24
            x = int(round(tracklet["x0"] + tracklet["vx"] * hours)) + int(offsets[k][0])
            y = int(round(tracklet["y0"] + tracklet["vy"] * hours)) + int(offsets[k][1])
            if half <= x < width - half and half <= y < height - half:
                window = (slice(y - half, y + half + 1), slice(x - half, x + half + 1))
                total += snr_maps[k][window]
                count += good_maps[k][window]
        stamps[s] = total / np.sqrt(np.maximum(count, 1))
    return stamps
