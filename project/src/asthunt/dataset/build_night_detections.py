"""Turn one night of images into a labelled table of detections plus their stamps."""

import numpy as np
import pandas as pd
from astropy.wcs import WCS

from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.clean.load_exposure import load_exposure
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.detect.cut_triplet_stamps import cut_triplet_stamps
from asthunt.detect.extract_sources import extract_sources
from asthunt.label.label_detections import label_detections
from asthunt.label.load_known_asteroids import load_known_asteroids
from asthunt.label.query_known_asteroids import query_known_asteroids
from asthunt.simulate.plant_fakes_in_exposure import plant_fakes_in_exposure
from asthunt.simulate.plan_injections import plan_injections


def build_night_detections(unit, metadata, acquire, inject, detect, paths, rng, with_injections=True):
    """Process one unit (one CCD quadrant on one night) and save three files in data/processed/:

        detections__<tag>.parquet   one row per detection, with labels
        stamps__<tag>.npy           (n_detections, 3, size, size) float16 stamps
        injections__<tag>.parquet   where every fake asteroid was planted

    unit     : {field, ccd, quadrant, night}; tag = make_unit_tag(unit), e.g. 000618_c03_q4_2019-02-08
    metadata : the archive's exposure table for this unit's field/CCD/quadrant

    with_injections=False is the real search: no fakes are planted, and the
    files get the prefix "search_" (search_detections__..., search_stamps__...)
    so they never overwrite the training/evaluation files.

    With fakes planted, the planted science and difference images of every
    frame are also saved (data/interim/night_images/<tag>/), because later
    stages cut stamps at positions where nothing was detected (see
    clean/load_night_images.py).

    If the files already exist they are loaded instead of rebuilt.
    Returns (detections, stamps, injections).

    Steps for each frame of the night:
      load images -> plant fakes -> detect -> label -> cut stamps

    The telescope pointing shifts by several pixels between exposures, so the
    same sky position lands on different pixels in each frame. Asteroids move
    in straight lines ON THE SKY, so:
      - fakes are planned on frame 0's pixel grid and converted, via the sky
        (RA, Dec), to each frame's own pixels before planting;
      - every detection also gets x_common, y_common: its position on frame
        0's pixel grid. Linking uses these common coordinates.
    """
    night = unit["night"]
    tag = make_unit_tag(unit)
    prefix = "" if with_injections else "search_"
    out = paths["processed"]
    detections_file = out / f"{prefix}detections__{tag}.parquet"
    stamps_file = out / f"{prefix}stamps__{tag}.npy"
    injections_file = out / f"{prefix}injections__{tag}.parquet"
    if detections_file.exists() and stamps_file.exists() and injections_file.exists():
        return pd.read_parquet(detections_file), np.load(stamps_file), pd.read_parquet(injections_file)

    # 1. The frames of this night, in time order
    frames = select_night_exposures(metadata, night, acquire["filter"], acquire["frames_per_night"])
    exposures = []
    for _, frame in frames.iterrows():
        science_name = (f"ztf_{frame['filefracday']}_{int(unit['field']):06d}_{acquire['filter']}"
                        f"_c{int(unit['ccd']):02d}_o_q{int(unit['quadrant'])}_sciimg.fits")
        exposures.append(load_exposure(paths["raw"] / "ztf" / night / science_name, acquire["bad_mask_bits"]))

    # 2. Plan the fakes once for the whole night, so each one moves through all frames
    n_fakes = inject["asteroids_per_night"] if with_injections else 0
    injections = plan_injections(
        n_fakes, exposures[0]["science"].shape, [e["obsjd"] for e in exposures],
        inject["magnitude_range"], inject["rate_range_arcsec_per_hour"],
        exposures[0]["header"]["PIXSCALE"], inject["edge_margin_pixels"], rng,
    )
    reference_wcs = WCS(exposures[0]["header"])       # frame 0's grid = the common grid
    planted = []

    all_detections, all_stamps = [], []
    for frame_index, exposure in enumerate(exposures):
        # 3. Plant this frame's fakes into copies of the images
        science, difference = exposure["science"].copy(), exposure["difference"].copy()
        frame_wcs = WCS(exposure["header"])
        fakes = plant_fakes_in_exposure(exposure, science, difference, injections[injections["frame_index"] == frame_index],
                                        reference_wcs, frame_wcs, rng)
        planted.append(fakes)

        # 4. Detect candidates in the difference image
        detections = extract_sources(difference, exposure["bad"], detect["threshold_sigma"], detect["min_area_pixels"])
        if n_fakes:
            _save_planted_images(paths, tag, exposure["name"], science, difference)

        # 5. Label: matched fake, bright-enough known asteroid, or bogus
        query_known_asteroids(                     # makes sure the SkyBoT answer is cached
            exposure["header"], science.shape,
            cache_file=paths["interim"] / "known_asteroids" / f"known_asteroids__{exposure['name']}.csv")
        known = load_known_asteroids(exposure["name"], paths["interim"])   # Horizons positions if refined
        known = known[known["v_mag"] <= inject["known_asteroid_max_v_mag"]]
        detections = label_detections(detections, fakes, known, inject["match_radius_pixels"],
                                      inject["known_match_radius_pixels"])

        # 6. Describe each detection: when, where on the sky, how bright
        sky = frame_wcs.pixel_to_world(detections["x"].to_numpy(), detections["y"].to_numpy())
        detections["ra"], detections["dec"] = sky.ra.deg, sky.dec.deg
        detections["x_common"], detections["y_common"] = reference_wcs.world_to_pixel(sky)
        detections["mag"] = exposure["magzp"] - 2.5 * np.log10(detections["flux"].clip(lower=1e-3))
        detections["night"] = night
        detections["unit"] = tag
        detections["frame_index"] = frame_index
        detections["exposure"] = exposure["name"]
        detections["obsjd"] = exposure["obsjd"]
        detections["seeing"] = exposure["header"]["SEEING"]
        detections["maglim"] = exposure["header"]["MAGLIM"]

        # 7. Cut stamps. float16 halves the disk space and has plenty of precision for counts,
        #    but its largest value is 65504: clip first, or saturated stars become "infinity".
        stamps = cut_triplet_stamps(science, science - difference, difference,
                                    detections["x"].to_numpy(), detections["y"].to_numpy(), detect["stamp_size"])
        all_detections.append(detections)
        all_stamps.append(np.clip(stamps, -65000, 65000).astype(np.float16))

    detections = pd.concat(all_detections, ignore_index=True)
    stamps = np.concatenate(all_stamps)
    injections = pd.concat(planted, ignore_index=True)    # positions in each frame's own pixels
    injections["night"] = night
    injections["unit"] = tag

    out.mkdir(parents=True, exist_ok=True)
    detections.to_parquet(detections_file)
    np.save(stamps_file, stamps)
    injections.to_parquet(injections_file)
    return detections, stamps, injections


def _save_planted_images(paths, tag, exposure_name, science, difference):
    """Keep the images WITH fakes, as float16 (clipped: float16 can't hold values above 65504)."""
    folder = paths["interim"] / "night_images" / tag
    folder.mkdir(parents=True, exist_ok=True)
    for kind, image in [("science", science), ("difference", difference)]:
        np.save(folder / f"{exposure_name}__{kind}.npy", np.clip(image, -65000, 65000).astype(np.float16))
