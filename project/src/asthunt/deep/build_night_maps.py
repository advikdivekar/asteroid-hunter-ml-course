"""Standardised matched-filter SNR maps for every frame of one unit's night."""

import numpy as np
from astropy.wcs import WCS

from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.clean.load_exposure import load_exposure
from asthunt.deep.build_snr_image import build_snr_image
from asthunt.deep.compute_frame_offsets import compute_frame_offsets
from asthunt.deep.standardize_over_night import standardize_over_night
from asthunt.simulate.plant_fakes_in_exposure import plant_fakes_in_exposure

EXPOSURE_SECONDS = 30.0


def build_night_maps(unit, acquire, paths, min_frames=4, fakes=None, rng=None):
    """Return a dict of the night's maps, or None if the unit has fewer than `min_frames` usable frames.

    unit : {field, ccd, quadrant, night}; every good frame of the night is used (the same frames the deep search stacks)

    Keys: snr (frames, h, w) standardised matched-filter maps; good (frames, h, w) usable-pixel masks;
          times_jd start of each exposure; mid_times_jd the middle of each (what a position belongs to);
          offsets whole-pixel pointing shifts vs frame 0 (for stacking on one grid);
          wcs_list every frame's WCS; names; headers.
    These are exactly the maps the deep search builds, so a track measured on them means what a deep-search peak means.

    fakes : optional plan_injections-style table (injection_id, frame_index, x, y on the REFERENCE grid = frame 0's pixels, mag):
            planted into every frame before the maps are built (the evaluation run: what does this search recover?). None = real data.
    """
    metadata = load_unit_metadata(unit, paths["raw"])
    frames = select_night_exposures(metadata, unit["night"], acquire["filter"], max_frames=10_000)
    if len(frames) < min_frames:
        return None
    snr_maps, good_maps, wcs_list, times, names, headers = [], [], [], [], [], []
    for frame_index, (_, frame) in enumerate(frames.iterrows()):
        name = (f"ztf_{frame['filefracday']}_{unit['field']:06d}_{acquire['filter']}"
                f"_c{unit['ccd']:02d}_o_q{unit['quadrant']}")
        exposure = load_exposure(paths["raw"] / "ztf" / unit["night"] / f"{name}_sciimg.fits", acquire["bad_mask_bits"])
        frame_wcs = WCS(exposure["header"])
        difference = exposure["difference"]
        if fakes is not None:
            difference = difference.copy()
            plant_fakes_in_exposure(exposure, exposure["science"].copy(), difference, fakes[fakes["frame_index"] == frame_index],
                                    wcs_list[0] if wcs_list else frame_wcs, frame_wcs, rng)
        snr_maps.append(build_snr_image(difference, exposure["bad"], exposure["psf"]))
        good_maps.append(~exposure["bad"])
        wcs_list.append(frame_wcs)
        times.append(exposure["obsjd"])
        names.append(name)
        headers.append(exposure["header"])
    snr_maps, good_maps, times = np.stack(snr_maps), np.stack(good_maps), np.array(times)
    offsets = compute_frame_offsets(wcs_list, snr_maps[0].shape)
    snr_maps = standardize_over_night(snr_maps, good_maps, offsets)
    return {"snr": snr_maps, "good": good_maps, "times_jd": times, "mid_times_jd": times + EXPOSURE_SECONDS / 2 / 86400,
              "offsets": offsets, "wcs_list": wcs_list, "wcs": wcs_list[0], "names": names, "headers": headers}
