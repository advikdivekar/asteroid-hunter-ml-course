"""Command line: search one unit, or run the live monitor.

    python -m asthunt.cli search --field 618 --ccd 3 --quadrant 4 --night 2019-02-08
    python -m asthunt.cli watch --once          # one monitoring pass
    python -m asthunt.cli watch                 # keep watching, every `poll_minutes` (configs/live.yaml)
"""

import argparse
import time

import torch

from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.config.load_config import load_config
from asthunt.dataset.make_unit_rng import make_unit_rng
from asthunt.models.load_model_bundle import load_model_bundle
from asthunt.pipeline.run_full_search import run_full_search
from asthunt.pipeline.run_live_monitor import run_live_monitor

CONFIG_NAMES = ["acquire", "inject", "detect", "link", "fast", "deep", "live"]


def main(arguments=None):
    """Parse the command line and run the chosen command."""
    parser = argparse.ArgumentParser(prog="asthunt", description="Find asteroids in ZTF images.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="run all searches on one CCD quadrant on one night")
    for name in ["field", "ccd", "quadrant"]:
        search.add_argument(f"--{name}", type=int, required=True)
    search.add_argument("--night", required=True, help="YYYY-MM-DD")
    watch = commands.add_parser("watch", help="look for newly published nights and search them")
    watch.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args(arguments)

    project = load_config("project")
    configs = {name: load_config(name) for name in CONFIG_NAMES}
    bundle = load_model_bundle(project["paths"]["models"], configs["live"]["bundle"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if args.command == "search":
        unit = {"field": args.field, "ccd": args.ccd, "quadrant": args.quadrant, "night": args.night}
        result = run_full_search(unit, load_unit_metadata(unit, project["paths"]["raw"]), configs, bundle,
                                 project["paths"], make_unit_rng(project["seed"], unit), device)
        print(result["candidates"].to_string(index=False) if len(result["candidates"]) else "no unknown candidates")
        return

    while True:
        searched = run_live_monitor(configs, project["paths"], bundle, device=device)
        print(f"{time.strftime('%Y-%m-%d %H:%M')}: searched {len(searched)} new night(s)")
        if args.once:
            return
        time.sleep(configs["live"]["poll_minutes"] * 60)


if __name__ == "__main__":
    main()
