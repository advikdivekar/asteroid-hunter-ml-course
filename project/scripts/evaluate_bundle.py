"""Put one bundle on the scoreboard. Run: python scripts/evaluate_bundle.py v6 [validation test holdout]

The same evaluation as notebooks/scoreboard.ipynb, for one version: the deployed pipeline (run_night_pipeline)
with fakes planted, on every unit of each split; one row per split in reports/scoreboard.csv.
"""
import subprocess
import sys
import time
from datetime import date

from asthunt.acquire.list_units import list_units
from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.config.load_config import load_config
from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.evaluate.build_scoreboard import build_scoreboard
from asthunt.evaluate.save_scoreboard_rows import save_scoreboard_rows
from asthunt.models.load_model_bundle import load_model_bundle
from asthunt.pipeline.run_night_pipeline import run_night_pipeline

version = sys.argv[1]
splits = sys.argv[2:] or ["validation", "test", "holdout"]
project = load_config("project"); paths = project["paths"]
configs = {name: load_config(name) for name in ["acquire", "inject", "detect", "link"]}
bundle = load_model_bundle(paths["models"], version)
git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=paths["models"].parent).stdout.strip()

rows = []
for split in splits:
    results = []
    for unit in list_units(configs["acquire"], split):
        results.append(run_night_pipeline(unit, load_unit_metadata(unit, paths["raw"]), configs, bundle, paths,
                                          make_unit_rng(project["seed"], unit), "cpu", with_injections=True, write_reports=False))
        print(time.strftime("%H:%M:%S"), version, split, unit, "done", flush=True)
    row = build_scoreboard(results, bundle["thresholds"], configs["link"]["min_detections"], configs["inject"]["known_asteroid_max_v_mag"])
    rows.append({"version": version, "split": split, "evaluated": str(date.today()), "code": git_sha, **row})
    print(rows[-1], flush=True)

print(save_scoreboard_rows(rows, paths["reports"] / "scoreboard.csv").round(3).to_string())
