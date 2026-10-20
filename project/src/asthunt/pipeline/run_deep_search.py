"""Deep search on one unit: stack groups of frames along trial motions, then link the stacks across the night."""

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.clean.load_exposure import load_exposure
from asthunt.deep.build_snr_image import build_snr_image
from asthunt.deep.compute_frame_offsets import compute_frame_offsets
from asthunt.deep.group_frames_into_stacks import group_frames_into_stacks
from asthunt.deep.make_velocity_grid import make_velocity_grid
from asthunt.deep.mask_bright_pixels import mask_bright_pixels
from asthunt.deep.measure_velocity_consistency import measure_velocity_consistency
from asthunt.deep.merge_peaks import merge_peaks
from asthunt.deep.stack_and_find_peaks import stack_and_find_peaks
from asthunt.deep.standardize_over_night import standardize_over_night
from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets
from asthunt.simulate.plant_fakes_in_exposure import plant_fakes_in_exposure


def run_deep_search(unit, metadata, acquire, deep, paths, device="cpu", fakes=None, rng=None, progress=print,
                    shuffle_times_seed=None, return_maps=False):
    """Return {"stack_detections", "tracklets", "members", "stacks", "reference_wcs"} for one unit.

    unit   : {field, ccd, quadrant, night}; EVERY good frame of the night is used
    deep   : configs/deep.yaml
    fakes  : optional plan_injections table (x, y on the reference grid = the
             night's first frame; frame_index = position in time order) to
             plant before stacking: the evaluation run. None = the real search.

    1. Group the night's frames into stacks of consecutive exposures.
    2. Matched-filter SNR map of every frame (fakes planted first, if any).
    3. Standardise every pixel over the night ((value - night median) / night
       scatter): removes static residuals, turns down ones that wobble.
       Then hide pixels single frames already see (> mask_bright_sigma): the deep
       search is for faint signals, and bright objects would create echo tracklets.
    4. Per stack: shift-and-stack along every trial motion (each frame clipped
       at clip_sigma), keep the peaks. Positions are on the reference grid, at
       the stack's middle time. Merge peaks of the same object.
    5. Link stack detections across the night: a real object must appear in
       >= `min_detections` independent stacks along one straight line.
    6. Mark tracklets `consistent` when every stack's own best velocity agrees
       with the night-long motion (velocity_mismatch <= the configured limit).

    progress: called with a short message after every stack (long runs should
              never be silent); pass `lambda message: None` to turn it off.
    shuffle_times_seed: the NULL TEST. Give every frame another frame's time
              (a random shuffle). Real moving objects then no longer line up along
              any straight motion, while noise and artifacts are unchanged: every
              tracklet that survives is a false alarm. Estimates the false-alarm rate.
    return_maps: also return the standardised maps ("maps": snr, good, times, offsets,
              stack_rows) for vetting stamps. Costs ~2-3 GB of memory for a 55-frame night.
    """
    frames = select_night_exposures(metadata, unit["night"], acquire["filter"], max_frames=10_000)
    stacks = group_frames_into_stacks(frames["obsjd"].tolist(), deep["frames_per_stack"], deep["max_gap_minutes"])
    names = [_science_path(frame, unit, acquire, paths) for _, frame in frames.iterrows()]
    reference_wcs = WCS(load_exposure(names[stacks[0][0]], acquire["bad_mask_bits"])["header"])
    velocities = make_velocity_grid(deep["max_rate_arcsec_per_hour"], deep["velocity_step_pixels_per_hour"], 1.012)

    # 1. Matched-filter SNR map of every frame of the night (fakes planted first, when evaluating)
    snr_maps, good_maps, wcs_list, times = [], [], [], []
    used = sorted({position for stack in stacks for position in stack})
    for frame_position in used:
        exposure = load_exposure(names[frame_position], acquire["bad_mask_bits"])
        difference = exposure["difference"]
        frame_wcs = WCS(exposure["header"])
        if fakes is not None:
            difference = difference.copy()
            plant_fakes_in_exposure(exposure, exposure["science"].copy(), difference,
                                    fakes[fakes["frame_index"] == frame_position], reference_wcs, frame_wcs, rng)
        snr_maps.append(build_snr_image(difference, exposure["bad"], exposure["psf"]))
        good_maps.append(~exposure["bad"])
        wcs_list.append(frame_wcs)
        times.append(exposure["obsjd"])
    snr_maps, good_maps, times = np.stack(snr_maps), np.stack(good_maps), np.array(times)
    if shuffle_times_seed is not None:
        times = np.random.default_rng(shuffle_times_seed).permutation(times)
    offsets = compute_frame_offsets([reference_wcs] + wcs_list, snr_maps[0].shape)[1:]
    progress(f"{len(used)} SNR maps built")

    # 2. Judge every pixel against its own night: removes static structure (star halos, galaxies)
    #    and turns down pixels that wobble from frame to frame (seeing-dependent star residuals)
    if deep.get("standardize_over_night", True):
        snr_maps = standardize_over_night(snr_maps, good_maps, offsets)
        progress("standardised over the night")

    #    Hide what single frames already see: bright objects otherwise create "echo" tracklets
    if deep.get("mask_bright_sigma") is not None:
        snr_maps, good_maps = mask_bright_pixels(snr_maps, good_maps, deep["mask_bright_sigma"], deep["mask_bright_radius_pixels"])

    # 3. Stack each group of frames along every trial motion
    row_of = {frame_position: row for row, frame_position in enumerate(used)}
    detections, stack_rows = [], []
    for stack_index, members in enumerate(stacks):
        rows = [row_of[frame_position] for frame_position in members]
        hours = (times[rows] - times[rows[0]]) * 24
        peaks = stack_and_find_peaks(snr_maps[rows], good_maps[rows], hours, offsets[rows], velocities,
                                     deep["peak_threshold_sigma"], device, clip_sigma=deep.get("clip_sigma"))

        # Position at the stack's middle time, on the reference grid
        middle_hours = hours.mean()
        peaks["x_common"] = peaks["x"] + peaks["vx"] * middle_hours
        peaks["y_common"] = peaks["y"] + peaks["vy"] * middle_hours
        peaks = merge_peaks(peaks.assign(x=peaks["x_common"], y=peaks["y_common"]), deep["merge_radius_pixels"])
        peaks["frame_index"] = stack_index
        peaks["obsjd"] = times[rows].mean()
        detections.append(peaks)
        stack_rows.append({"stack": stack_index, "frames": len(members), "start_jd": times[rows[0]], "middle_jd": times[rows].mean(),
                           "middle_exposure": names[members[len(members) // 2]].name.replace("_sciimg.fits", ""),
                           "peaks": len(peaks)})
        progress(f"stack {stack_index + 1}/{len(stacks)}: {len(members)} frames, {len(peaks)} peaks")

    stack_detections = pd.concat(detections, ignore_index=True)
    sky = reference_wcs.pixel_to_world(stack_detections["x_common"].to_numpy(), stack_detections["y_common"].to_numpy())
    stack_detections["ra"], stack_detections["dec"] = sky.ra.deg, sky.dec.deg

    link = deep["link"]
    linkable = stack_detections.assign(x=stack_detections["x_common"], y=stack_detections["y_common"])
    tracklets, members = link_detections_into_tracklets(
        linkable, pixel_scale=1.012, min_rate=link["min_rate_arcsec_per_hour"], max_rate=link["max_rate_arcsec_per_hour"],
        match_radius=link["match_radius_pixels"], min_detections=link["min_detections"],
        max_residual=link["max_residual_pixels"])
    # Every stack, on its own, must agree with the night-long motion (chance alignments don't)
    if len(tracklets):
        tracklets = measure_velocity_consistency(tracklets, members, stack_detections)
        tracklets["consistent"] = tracklets["velocity_mismatch"] <= deep["max_velocity_mismatch_pixels_per_hour"]

    result = {"stack_detections": stack_detections, "tracklets": tracklets, "members": members,
              "stacks": pd.DataFrame(stack_rows), "reference_wcs": reference_wcs}
    if return_maps:
        result["maps"] = {"snr": snr_maps, "good": good_maps, "times": times, "offsets": offsets,
                          "stack_rows": [[row_of[p] for p in members_] for members_ in stacks]}
    return result


def _science_path(frame, unit, acquire, paths):
    name = (f"ztf_{frame['filefracday']}_{int(unit['field']):06d}_{acquire['filter']}"
            f"_c{int(unit['ccd']):02d}_o_q{int(unit['quadrant'])}_sciimg.fits")
    return paths["raw"] / "ztf" / unit["night"] / name
