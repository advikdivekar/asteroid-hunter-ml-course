import json

import pandas as pd

from asthunt.pipeline.run_live_monitor import run_live_monitor

CONFIGS = {"acquire": {"filter": "zr", "products": ["sciimg.fits"]},
           "live": {"watch": [{"field": 618, "ccd": 3, "quadrant": 4}], "min_frames_for_point_search": 4,
                    "night_is_complete_after_hours": 12}}


def fake_metadata(field, ccd, quadrant):
    rows = []
    for night, jd, n in [("2026-10-01", 2461314.7, 5),     # complete, enough frames -> search
                         ("2026-10-02", 2461315.7, 2),     # too few frames -> skip
                         ("2026-10-13", 2461326.7, 6)]:    # observed hours ago -> not complete yet
        rows += [{"night": night, "filtercode": "zr", "infobits": 0, "obsjd": jd + i / 288} for i in range(n)]
    return pd.DataFrame(rows)


def fake_search(unit, metadata, configs, bundle, paths, rng, device, progress):
    return {"candidates": pd.DataFrame([{"search": "deep", "tracklet_id": 7, "n_detections": 7,
                                         "rate_arcsec_per_hour": 40.4, "ra": 135.52, "dec": 22.50, "score": 0.9}])}


def test_searches_only_complete_new_nights_and_writes_alerts_once(tmp_path):
    paths = {"raw": tmp_path / "raw", "reports": tmp_path / "reports"}
    downloads = []
    options = dict(now_jd=2461326.75, progress=lambda m: None, query_metadata=fake_metadata,
                   download=lambda frames, products, folder: downloads.append(len(frames)), search=fake_search)

    first = run_live_monitor(CONFIGS, paths, bundle=None, **options)
    second = run_live_monitor(CONFIGS, paths, bundle=None, **options)            # same moment: nothing new

    assert [s["unit"] for s in first] == ["000618_c03_q4_2026-10-01"] and second == []
    assert downloads == [5]
    alerts = [json.loads(line) for line in (paths["reports"] / "alerts.jsonl").read_text().splitlines()]
    assert len(alerts) == 1 and alerts[0]["search"] == "deep" and alerts[0]["unit"] == "000618_c03_q4_2026-10-01"


def test_nights_outside_the_window_are_ignored(tmp_path):
    configs = {**CONFIGS, "live": {**CONFIGS["live"], "start_after_night": "2026-10-05"}}
    searched = run_live_monitor(configs, {"raw": tmp_path, "reports": tmp_path}, bundle=None, now_jd=2461326.75,
                                progress=lambda m: None, query_metadata=fake_metadata,
                                download=lambda *a: None, search=fake_search)
    assert searched == []                        # 2026-10-01 is before the deployment date
