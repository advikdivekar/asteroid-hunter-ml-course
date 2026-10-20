"""All three searches on one unit, one combined list of candidates."""

import pandas as pd

from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.crossmatch.crossmatch_deep_tracklets import crossmatch_deep_tracklets
from asthunt.pipeline.run_deep_search import run_deep_search
from asthunt.pipeline.run_fast_search import run_fast_search
from asthunt.pipeline.run_night_pipeline import run_night_pipeline

CANDIDATE_COLUMNS = ["search", "tracklet_id", "n_detections", "rate_arcsec_per_hour", "ra", "dec", "score"]


def run_full_search(unit, metadata, configs, bundle, paths, rng, device="cpu", progress=print, light=False):
    """Return {"candidates", "point", "fast", "deep"} for one unit (the real search, no fakes).

    configs: {"acquire", "inject", "detect", "link", "fast", "deep", "live"} (the YAML files)

    1. point search (run_night_pipeline): objects visible in single frames, up to 120 "/hr
    2. fast search: trails of fast movers (~480-3,800 "/hr), only with >= fast["min_frames"] frames
    3. deep search: shift-and-stack, only if the night has enough frames for the 3 stacks the linker needs (20)
    Every search cross-matches with known objects; `candidates` lists what is left:
    unknown objects that passed each search's own quality checks.

    light=True: for exposures downloaded without their science image (configs/acquire.yaml light_products). The point
    search cuts science stamps, so it is skipped (result["point"] is None); the fast and deep searches run unchanged.
    """
    frames = select_night_exposures(metadata, unit["night"], configs["acquire"]["filter"], max_frames=10_000)
    live = configs["live"]
    rows = []

    point = None
    if light:
        progress("point search skipped (light mode: no science images)")
    else:
        point = run_night_pipeline(unit, metadata, configs, bundle, paths, rng, device)
        for _, c in point["candidates"].iterrows():
            first = point["members"].loc[point["members"]["tracklet_id"] == c["tracklet_id"], "detection_id"].iloc[0]
            rows.append({"search": "point", "tracklet_id": c["tracklet_id"], "n_detections": c["n_detections"],
                         "rate_arcsec_per_hour": c["rate_arcsec_per_hour"], "ra": point["detections"].loc[first, "ra"],
                         "dec": point["detections"].loc[first, "dec"], "score": c["tracklet_score"]})
        progress(f"point search: {len(point['candidates'])} unknown candidates")

    fast = None
    if len(frames) >= configs["fast"]["min_frames"]:
        fast = run_fast_search(unit, metadata, configs["acquire"], configs["fast"], paths,
                               max_frames=configs["fast"]["max_frames"], progress=lambda m: None)
    else:
        progress(f"fast search skipped: {len(frames)} frames, needs {configs['fast']['min_frames']} for a clean night standardisation")
    if fast is not None and len(fast["tracklets"]):
        for _, c in fast["tracklets"][fast["tracklets"]["consistent"]].iterrows():
            first = fast["members"].loc[fast["members"]["tracklet_id"] == c["tracklet_id"], "detection_id"].iloc[0]
            rows.append({"search": "fast", "tracklet_id": c["tracklet_id"], "n_detections": c["n_detections"],
                         "rate_arcsec_per_hour": c["rate_arcsec_per_hour"], "ra": fast["streaks"].loc[first, "ra"],
                         "dec": fast["streaks"].loc[first, "dec"], "score": 1.0 - c["angle_difference"] / 90})
    progress(f"fast search: {sum(r['search'] == 'fast' for r in rows)} consistent fast tracks")

    deep = None
    if len(frames) >= live["min_frames_for_deep_search"]:
        deep = run_deep_search(unit, metadata, configs["acquire"], configs["deep"], paths, device, progress=lambda m: None)
        matched = crossmatch_deep_tracklets(deep, unit, paths)
        unknown = matched[matched["consistent"] & (matched["known_name"] == "")] if len(matched) else matched
        for _, c in unknown.iterrows():
            first = deep["members"].loc[deep["members"]["tracklet_id"] == c["tracklet_id"], "detection_id"].iloc[0]
            rows.append({"search": "deep", "tracklet_id": c["tracklet_id"], "n_detections": c["n_detections"],
                         "rate_arcsec_per_hour": c["rate_arcsec_per_hour"], "ra": deep["stack_detections"].loc[first, "ra"],
                         "dec": deep["stack_detections"].loc[first, "dec"], "score": 1.0 / (1.0 + c["velocity_mismatch"])})
        progress(f"deep search: {len(unknown)} unknown consistent tracklets")
    else:
        progress(f"deep search skipped: {len(frames)} frames, needs {live['min_frames_for_deep_search']} for the 3 stacks the linker requires")

    candidates = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    return {"candidates": candidates, "point": point, "fast": fast, "deep": deep}
