"""Cut stamps along each tracklet's predicted path, in EVERY frame of the night."""

import numpy as np

from asthunt.detect.cut_triplet_stamps import cut_triplet_stamps


def cut_tracklet_stamps(tracklets, frames, reference_wcs, images, size=21):
    """Return an array (n_tracklets, n_frames, 3, size, size): science / reference / difference.

    tracklets     : linker output with the fitted line x0, y0, vx, vy, t0_jd
                    (on the common grid = frame 0's pixels)
    frames        : list of (exposure name, obsjd) in time order
    reference_wcs : WCS of frame 0 (defines the common grid)
    images        : load_night_images output

    The stamp is centred where the straight line says the object should be at
    that frame's time, even in frames where it was NOT detected. A faint
    asteroid detected in only 3 frames still leaves a weak blob in the other
    2, and a model looking at all 5 stamps can use that signal.
    Positions go common grid -> sky -> this frame's pixels, because the
    telescope pointing shifts between exposures.
    """
    stamps = np.zeros((len(tracklets), len(frames), 3, size, size), dtype=np.float32)
    if len(tracklets) == 0:
        return stamps

    for index, (name, obsjd) in enumerate(frames):
        hours = (obsjd - tracklets["t0_jd"].to_numpy()) * 24
        x_common = tracklets["x0"].to_numpy() + tracklets["vx"].to_numpy() * hours
        y_common = tracklets["y0"].to_numpy() + tracklets["vy"].to_numpy() * hours

        sky = reference_wcs.pixel_to_world(x_common, y_common)
        x_frame, y_frame = images[name]["wcs"].world_to_pixel(sky)

        image = images[name]
        stamps[:, index] = cut_triplet_stamps(image["science"], image["reference"], image["difference"],
                                              x_frame, y_frame, size)
    return stamps
