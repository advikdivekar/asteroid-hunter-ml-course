"""Search one unit on night 1, follow every object to wherever it is on night 2, and link the two nights."""

import numpy as np
import pandas as pd
from astropy.time import Time

from asthunt.acquire.download_ztf_images import download_ztf_images
from asthunt.acquire.download_ztf_reference_image import download_ztf_reference_image
from asthunt.acquire.find_quadrants_covering_position import find_quadrants_covering_position
from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.acquire.query_ztf_field_night import query_ztf_field_night
from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.crossmatch.crossmatch_deep_tracklets import crossmatch_deep_tracklets
from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.link.link_tracklets_across_nights import link_tracklets_across_nights
from asthunt.link.measure_sky_motion import measure_sky_motion
from asthunt.pipeline.run_full_search import run_full_search


def run_two_night_hunt(unit, second_night, configs, bundle, paths, seed=42, device="cpu", progress=print, light=False):
    """Return {"first", "second", "pairs", "second_units"}.

    first / second : one row per accepted object (point search tracklets above the
                     bundle threshold, velocity-consistent deep tracklets): position,
                     velocity, search, known_name, unit
    pairs          : the cross-night links, with both nights' known names
    second_units   : the night-2 quadrants that were searched

    Steps:
      1. night 1: download, run all searches (run_full_search), measure every object's sky motion;
      2. predict each object's night-2 position and ask the archive which quadrants
         of the same field saw it (objects move ~0.75 deg in 3 days: often a neighbouring quadrant);
      3. download and search those quadrants on night 2;
      4. link the nights (link_tracklets_across_nights).
    Known asteroids are kept on purpose: when they link to themselves, the hunt has proved
    it works on this data, which is what makes an unknown link believable.

    light=True: download only configs["acquire"]["light_products"] (no science image, 41% of the data) and skip the point
    search; the fast and deep searches (and everything after them) are unchanged.
    """
    first = _search_unit(unit, configs, bundle, paths, seed, device, progress, light)
    progress(f"night 1: {len(first)} objects ({int((first['known_name'] == '').sum())} unknown)")

    # Where will each object be on night 2? Search every quadrant that saw one of those positions
    field_night = query_ztf_field_night(unit["field"], second_night,
                                        cache_file=paths["raw"] / "ztf" / f"field_night__{unit['field']:06d}_{second_night}.csv")
    field_night = field_night[(field_night["filtercode"] == configs["acquire"]["filter"]) & (field_night["infobits"] == 0)]
    quadrants = field_night.drop_duplicates(["ccdid", "qid"])
    middle_jd = field_night["obsjd"].median()
    needed = set()
    for _, row in first.iterrows():
        hours = (middle_jd - row["jd"]) * 24
        ra = row["ra"] + row["v_ra"] * hours / 3600 / np.cos(np.radians(row["dec"]))
        dec = row["dec"] + row["v_dec"] * hours / 3600
        margin = 0.15 * np.hypot(row["v_ra"], row["v_dec"]) * abs(hours) / 3600      # the linker's own tolerance
        for _, quadrant in find_quadrants_covering_position(quadrants, ra, dec, margin).iterrows():
            needed.add((int(quadrant["ccdid"]), int(quadrant["qid"])))
    progress(f"night 2: the objects should be in {len(needed)} quadrant(s): {sorted(needed)}")

    second_units, tables = [], []
    for ccd, quadrant in sorted(needed):
        night_unit = {"field": unit["field"], "ccd": ccd, "quadrant": quadrant, "night": second_night}
        frames = field_night[(field_night["ccdid"] == ccd) & (field_night["qid"] == quadrant)]
        if len(frames) < configs["live"]["min_frames_for_point_search"]:
            progress(f"  {make_unit_tag(night_unit)}: only {len(frames)} frames, skipped")
            continue
        tables.append(_search_unit(night_unit, configs, bundle, paths, seed, device, progress, light))
        second_units.append(make_unit_tag(night_unit))
    second = pd.concat(tables, ignore_index=True) if tables else first.iloc[0:0]

    pairs = link_tracklets_across_nights(first, second)
    names = lambda table, suffix: table[["tracklet_id", "known_name", "search", "unit"]].add_suffix(suffix)
    pairs = pairs.merge(names(first, "_1"), left_on="first_id", right_on="tracklet_id_1") \
                 .merge(names(second, "_2"), left_on="second_id", right_on="tracklet_id_2") \
                 .drop(columns=["tracklet_id_1", "tracklet_id_2"])
    return {"first": first, "second": second, "pairs": pairs, "second_units": second_units}


def _search_unit(unit, configs, bundle, paths, seed, device, progress, light=False):
    """Download one unit, run all searches, return one sky-motion row per accepted object."""
    raw = paths["raw"] / "ztf"
    metadata = load_unit_metadata(unit, paths["raw"])
    frames = select_night_exposures(metadata, unit["night"], configs["acquire"]["filter"], max_frames=10_000)
    if not light:                                       # the reference image is only for the point search's stamps
        download_ztf_reference_image(unit["field"], configs["acquire"]["filter"], unit["ccd"], unit["quadrant"], raw)
    download_ztf_images(frames, configs["acquire"]["light_products" if light else "products"], raw)
    progress(f"  {make_unit_tag(unit)}: {len(frames)} frames, searching" + (" (light mode)" if light else ""))

    result = run_full_search(unit, metadata, configs, bundle, paths, make_unit_rng(seed, unit), device, progress, light)
    tables = []
    point = result["point"]
    if point is not None and len(point["tracklets"]):
        accepted = point["tracklets"][point["tracklets"]["tracklet_score"] >= bundle["thresholds"]["tracklet"]]
        members = point["members"][point["members"]["tracklet_id"].isin(accepted["tracklet_id"])]
        tables.append(measure_sky_motion(members, point["detections"])
                      .merge(accepted[["tracklet_id", "known_name"]], on="tracklet_id").assign(search="point"))
    if result["deep"] is not None and len(result["deep"]["tracklets"]):
        matched = crossmatch_deep_tracklets(result["deep"], unit, paths)
        consistent = matched[matched["consistent"]]
        members = result["deep"]["members"][result["deep"]["members"]["tracklet_id"].isin(consistent["tracklet_id"])]
        tables.append(measure_sky_motion(members, result["deep"]["stack_detections"])
                      .merge(consistent[["tracklet_id", "known_name"]], on="tracklet_id").assign(search="deep"))
    if not tables:
        return pd.DataFrame(columns=["tracklet_id", "jd", "ra", "dec", "v_ra", "v_dec", "known_name", "search", "unit"])
    table = pd.concat(tables, ignore_index=True)
    table["unit"] = make_unit_tag(unit)
    # Tracklet ids are only unique within one search on one unit; night 2 combines several units
    table["tracklet_id"] = table["unit"] + "/" + table["search"] + "-" + table["tracklet_id"].astype(str)
    return table
