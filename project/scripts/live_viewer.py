"""Watch the pipeline work, stage by stage, in your browser.

    python scripts/live_viewer.py            # then open http://127.0.0.1:8765  (it opens by itself)

Pick a night, press Start. The page shows each stage as it finishes: the four files of an exposure, the noise, the blobs found, the
real/bogus model's verdicts, the tracks, the scores and the cross-match. "Step by step" pauses after every stage until you press Next.
"fakes" paints that many fake asteroids into the images first, so you can watch them being found (or missed, when too faint).
Runs the point-source search on nights whose images are on disk; the models are the live bundle in configs/live.yaml.
"""

import argparse
import threading
import webbrowser

import torch

from asthunt.config.load_config import load_config
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.models.load_model_bundle import load_model_bundle
from asthunt.viewer.list_viewer_units import list_viewer_units
from asthunt.viewer.run_traced_night import run_traced_night
from asthunt.viewer.serve_live_viewer import make_server

CONFIG_NAMES = ["acquire", "inject", "detect", "link", "live"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    project = load_config("project")
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    paths = project["paths"]
    bundle = load_model_bundle(paths["models"], configs["live"]["bundle"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    units = list_viewer_units(paths["raw"], configs["acquire"]["filter"], configs["acquire"]["frames_per_night"])
    for unit in units:
        unit["tag"] = make_unit_tag(unit)
        unit["label"] = f"{unit['night']}  field {unit['field']}  CCD {unit['ccd']}  quadrant {unit['quadrant']}  ({unit['frames']} frames)"
        if unit["night"] in configs["acquire"]["holdout_nights"] + configs["acquire"]["nights"]["test"]:
            unit["label"] += "  [held-out night: watching it changes nothing]"
    print(f"{len(units)} nights on disk can be shown; models {configs['live']['bundle']}, device {device}")

    def run_unit(unit, state, n_fakes):
        run_traced_night({k: unit[k] for k in ["field", "ccd", "quadrant", "night"]}, state, configs, bundle, paths, n_fakes, project["seed"], device)

    server, _ = make_server("127.0.0.1", args.port, units, run_unit)
    url = f"http://127.0.0.1:{args.port}"
    print(f"live view at {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
