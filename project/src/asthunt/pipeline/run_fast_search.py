"""Fast-mover search on one unit: find trails in every frame, link them, check they agree with the motion."""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.clean.load_exposure import load_exposure
from asthunt.deep.compute_frame_offsets import compute_frame_offsets
from asthunt.deep.standardize_over_night import standardize_over_night
from asthunt.fast.build_streak_kernels import build_streak_kernels
from asthunt.fast.check_streak_consistency import check_streak_consistency
from asthunt.fast.find_streaks import find_streaks
from asthunt.fast.flag_column_aligned import flag_column_aligned
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets
from asthunt.simulate.inject_trailed_source import inject_trailed_source


def run_fast_search(unit, metadata, acquire, fast, paths, fakes=None, rng=None, max_frames=5, progress=print):
    """Return {"streaks", "tracklets", "members"} for one unit.

    fakes: optional plan_injections table (x, y on the reference grid = first frame) of fast
           objects to plant as TRAILS before searching (the evaluation run); None = real search.

    1. Per frame: plant trailed fakes (if any); scale to units of the noise.
    2. Standardise every pixel over the frames (static and wobbling residuals fade).
    3. Find streaks with the trailed matched filters; put them on the common grid.
    4. Link streaks across frames (fast speeds only).
    5. Mark the tracklets whose motion agrees with their streaks' angle and length (and, if configured, drop column-aligned ones).
    """
    frames = select_night_exposures(metadata, unit["night"], acquire["filter"], max_frames)

    # 1. Load every frame, plant trailed fakes (if any), scale each difference image to units of its noise
    images, good_masks, wcs_list, exposures = [], [], [], []
    reference_wcs = None
    for frame_index, (_, frame) in enumerate(frames.iterrows()):
        name = (f"ztf_{frame['filefracday']}_{int(unit['field']):06d}_{acquire['filter']}"
                f"_c{int(unit['ccd']):02d}_o_q{int(unit['quadrant'])}_sciimg.fits")
        exposure = load_exposure(paths["raw"] / "ztf" / unit["night"] / name, acquire["bad_mask_bits"])
        frame_wcs = WCS(exposure["header"])
        reference_wcs = reference_wcs or frame_wcs
        difference = exposure["difference"].copy()
        if fakes is not None:
            these = fakes[fakes["frame_index"] == frame_index]
            sky = reference_wcs.pixel_to_world(these["x"].to_numpy(), these["y"].to_numpy())
            x_frame, y_frame = frame_wcs.world_to_pixel(sky)
            for (_, fake), x, y in zip(these.iterrows(), x_frame, y_frame):
                length = fake["rate_arcsec_per_hour"] * fast["exposure_seconds"] / 3600 / 1.012
                inject_trailed_source(exposure["science"].copy(), difference, exposure["psf"], fake["mag"],
                                      exposure["magzp"], exposure["gain"], x, y, length, fake["angle_degrees"], rng)
        good = ~exposure["bad"] & np.isfinite(difference)
        values = difference[good]
        scaled = np.where(good, difference, 0.0) / (1.4826 * np.median(np.abs(values - np.median(values))))
        images.append(scaled.astype(np.float32)); good_masks.append(good); wcs_list.append(frame_wcs)
        exposures.append({"name": exposure["name"], "obsjd": exposure["obsjd"], "psf": exposure["psf"]})

    # 2. Judge every pixel against its own behaviour over these frames. A fast object is never on
    #    the same pixel twice, so it is untouched; static and wobbling residuals are suppressed.
    if fast.get("standardize_over_night", True) and len(images) >= 3:
        offsets = compute_frame_offsets(wcs_list, images[0].shape)
        images = list(standardize_over_night(np.stack(images), np.stack(good_masks), offsets))

    # 3. Find streaks in each frame and put them on the common grid
    rows = []
    for frame_index, (image, good, frame_wcs, exposure) in enumerate(zip(images, good_masks, wcs_list, exposures)):
        kernels = build_streak_kernels(exposure["psf"], fast["streak_lengths_pixels"], fast["angle_step_degrees"])
        streaks = find_streaks(image, ~good, exposure["psf"], kernels, fast["detection_threshold_sigma"])
        if len(streaks):
            sky = frame_wcs.pixel_to_world(streaks["x"].to_numpy(), streaks["y"].to_numpy())
            streaks["x_common"], streaks["y_common"] = reference_wcs.world_to_pixel(sky)
            streaks["ra"], streaks["dec"] = sky.ra.deg, sky.dec.deg
            streaks["frame_index"] = frame_index
            streaks["obsjd"] = exposure["obsjd"] + fast["exposure_seconds"] / 2 / 86400     # middle of the trail
            streaks["exposure"] = exposure["name"]
            rows.append(streaks)
        progress(f"frame {frame_index + 1}/{len(images)}: {len(streaks)} streaks")

    streaks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["x", "y", "length", "angle", "snr", "point_snr", "x_common", "y_common", "frame_index", "obsjd"])
    link = fast["link"]
    linkable = streaks.assign(x=streaks["x_common"], y=streaks["y_common"])
    tracklets, members = link_detections_into_tracklets(
        linkable, pixel_scale=1.012, min_rate=link["min_rate_arcsec_per_hour"], max_rate=link["max_rate_arcsec_per_hour"],
        match_radius=link["match_radius_pixels"], min_detections=link["min_detections"],
        max_residual=link["max_residual_pixels"])
    if len(tracklets):
        tracklets = check_streak_consistency(tracklets, members, streaks, fast["exposure_seconds"], 1.012,
                                             fast["max_angle_difference_degrees"], fast["length_ratio_range"],
                                             fast.get("max_length_spread"))
        veto = fast.get("column_veto")                                   # off unless configured: see configs/fast.yaml
        if veto:
            tracklets = flag_column_aligned(tracklets, members, streaks, veto["max_angle_from_column_degrees"], veto["max_column_spread_pixels"])
            tracklets["consistent"] &= ~tracklets["column_aligned"]
    return {"streaks": streaks, "tracklets": tracklets, "members": members}
