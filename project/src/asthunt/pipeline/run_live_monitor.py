"""One pass of the live monitor: find newly published nights, search them, raise alerts."""

import json

from astropy.time import Time

from asthunt.acquire.download_ztf_images import download_ztf_images
from asthunt.acquire.query_ztf_metadata import query_ztf_metadata
from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.pipeline.run_full_search import run_full_search


def run_live_monitor(configs, paths, bundle, now_jd=None, device="cpu", progress=print,
                     query_metadata=query_ztf_metadata, download=download_ztf_images, search=run_full_search):
    """Check every watched unit once; return a list of {unit, frames, candidates} for the nights searched.

    For each watched (field, CCD, quadrant) in configs["live"]["watch"]:
      1. ask the archive for its exposures (fresh, not cached: we want what's new);
      2. pick nights after `start_after_night` (the deployment date; before
         `stop_before_night`) that are not processed yet, have enough good
         frames, and are complete (no new frame for `night_is_complete_after_hours`);
         Setting both dates to a past period REPLAYS it through the live system:
         the standard way to test a live pipeline before trusting it.
      3. download them, run all three searches, and append one ALERT line per
         unknown candidate to reports/alerts.jsonl (one JSON object per line,
         easy to forward to e-mail, Slack, ...);
      4. remember the night in reports/live_state.json so it's never searched twice.

    query_metadata / download / search can be replaced (tests use fakes for them).
    Call this on a schedule (e.g. every hour) to run it as a service.
    """
    live, acquire = configs["live"], configs["acquire"]
    now_jd = now_jd or Time.now().jd
    state_file = paths["reports"] / "live_state.json"
    alerts_file = paths["reports"] / "alerts.jsonl"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    searched = []

    for watched in live["watch"]:
        field, ccd, quadrant = int(watched["field"]), int(watched["ccd"]), int(watched["quadrant"])
        key = f"{field:06d}_c{ccd:02d}_q{quadrant}"
        metadata = query_metadata(field, ccd, quadrant)
        good = metadata[(metadata["filtercode"] == acquire["filter"]) & (metadata["infobits"] == 0)]

        for night, frames in good.groupby("night"):
            complete = (now_jd - frames["obsjd"].max()) * 24 >= live["night_is_complete_after_hours"]
            in_window = live.get("start_after_night", "0000-00-00") < night < live.get("stop_before_night", "9999-12-31")
            if not in_window or night in state.get(key, []) or not complete or len(frames) < live["min_frames_for_point_search"]:
                continue
            unit = {"field": field, "ccd": ccd, "quadrant": quadrant, "night": night}
            progress(f"{make_unit_tag(unit)}: {len(frames)} new frames, searching")
            download(frames, acquire["products"], paths["raw"] / "ztf")
            result = search(unit, metadata, configs, bundle, paths, make_unit_rng(42, unit), device, progress)

            alerts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alerts_file, "a") as f:
                for _, candidate in result["candidates"].iterrows():
                    f.write(json.dumps({"unit": make_unit_tag(unit), **{k: (v.item() if hasattr(v, "item") else v)
                                                                        for k, v in candidate.items()}}) + "\n")
            state.setdefault(key, []).append(night)
            state_file.write_text(json.dumps(state, indent=1))
            searched.append({"unit": make_unit_tag(unit), "frames": len(frames), "candidates": len(result["candidates"])})

    return searched
