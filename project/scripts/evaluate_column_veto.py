"""What would vetoing column-aligned fast tracks remove, and what would it cost? Two measurements.

    python scripts/evaluate_column_veto.py

  1. REAL TRACKS: the fast search on the field 567 hunt units on disk. Every consistent track, with its motion angle
     from the column direction and the spread of its column, and whether the veto would remove it.
  2. PLANTED TRAILS: the injection evaluation of docs/fast_movers.md (618 c03 q4, 2019-02-08, 10 frames, magnitude 15-20,
     400-5,000 arcsec/hr, random directions). How many are recovered with the current cuts, and how many would still
     be if column-aligned tracks were removed?
Nothing is changed in the search: this reads the tables and applies flag_column_aligned afterwards.
"""

import numpy as np
import pandas as pd

from asthunt.acquire.load_unit_metadata import load_unit_metadata
from asthunt.acquire.select_night_exposures import select_night_exposures
from asthunt.config.load_config import load_config
from asthunt.fast.flag_column_aligned import flag_column_aligned
from asthunt.pipeline.run_fast_search import run_fast_search
from asthunt.simulate.plan_injections import plan_injections

HUNT_UNITS = ["000567_c10_q1_2019-01-28", "000567_c10_q2_2019-01-31", "000567_c14_q3_2019-01-31"]
VETOS = [(5, 3.0), (10, 3.0), (15, 3.0)]                  # (max degrees from the column, max column spread in pixels)
SEEDS = [11, 12, 13]
FAKES_PER_RUN = 150
MATCH_RADIUS = 6


def parse(tag):
    field, ccd, quadrant, night = tag.split("_")
    return {"field": int(field), "ccd": int(ccd[1:]), "quadrant": int(quadrant[1:]), "night": night}


def tracks_following_fakes(result, fakes):
    """tracklet_id -> the fake it follows in at least 3 frames (the notebook's matching rule)."""
    streaks, members = result["streaks"], result["members"]
    answer = {}
    for tracklet_id, group in members.groupby("tracklet_id"):
        rows = streaks.loc[group["detection_id"]]
        for fake_id, track in fakes.groupby("injection_id"):
            hits = sum(((track["frame_index"] == r.frame_index) & (np.hypot(track["x"] - r.x_common, track["y"] - r.y_common) < MATCH_RADIUS)).any()
                       for r in rows.itertuples())
            if hits >= 3:
                answer[tracklet_id] = fake_id
                break
    return answer


def main():
    project, acquire, fast = load_config("project"), load_config("acquire"), load_config("fast")
    paths = project["paths"]

    print("1. REAL TRACKS (consistent fast tracks on the field 567 hunt units)")
    for tag in HUNT_UNITS:
        unit = parse(tag)
        result = run_fast_search(unit, load_unit_metadata(unit, paths["raw"]), acquire, fast, paths, max_frames=fast["max_frames"], progress=lambda m: None)
        tracklets = result["tracklets"]
        consistent = tracklets[tracklets["consistent"]] if len(tracklets) else tracklets
        print(f"\n{tag}: {len(consistent)} consistent tracks")
        if not len(consistent):
            continue
        flagged = flag_column_aligned(consistent, result["members"], result["streaks"], 90, 1e9)         # no limits here: just the numbers
        print(flagged[["tracklet_id", "rate_arcsec_per_hour", "column_angle_difference", "column_spread"]].round(1).to_string(index=False))
        for angle, spread in VETOS:
            removed = int(((flagged["column_angle_difference"] <= angle) & (flagged["column_spread"] <= spread)).sum())
            print(f"  veto within {angle} degrees and column spread <= {spread:.0f} px: removes {removed} of {len(flagged)}")

    print("\n2. PLANTED TRAILS (618 c03 q4, 2019-02-08)")
    unit = {"field": 618, "ccd": 3, "quadrant": 4, "night": "2019-02-08"}
    metadata = load_unit_metadata(unit, paths["raw"])
    frames = select_night_exposures(metadata, unit["night"], acquire["filter"], fast["max_frames"])
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        fakes = plan_injections(FAKES_PER_RUN, (3080, 3072), frames["obsjd"].tolist(), (15.0, 20.0), (400.0, 5000.0), 1.012, 100, rng)
        result = run_fast_search(unit, metadata, acquire, fast, paths, fakes=fakes, rng=rng, max_frames=fast["max_frames"], progress=lambda m: None)
        on_chip = fakes[fakes["x"].between(30, 3040) & fakes["y"].between(30, 3050)].groupby("injection_id").size()
        per_fake = fakes.groupby("injection_id").agg(mag=("mag", "first"), angle=("angle_degrees", "first"))
        per_fake = per_fake[per_fake.index.isin(on_chip[on_chip >= 3].index)]
        tracklets = result["tracklets"]
        hits = tracks_following_fakes(result, fakes)
        flagged = flag_column_aligned(tracklets, result["members"], result["streaks"], 90, 1e9) if len(tracklets) else tracklets
        for fake_id, row in per_fake.iterrows():
            ids = [t for t, f in hits.items() if f == fake_id and tracklets.set_index("tracklet_id").loc[t, "consistent"]]
            entry = {"seed": seed, "fake": fake_id, "mag": row["mag"], "angle": row["angle"], "found": bool(ids)}
            if ids:
                best = flagged.set_index("tracklet_id").loc[ids[0]]
                entry["column_angle_difference"], entry["column_spread"] = best["column_angle_difference"], best["column_spread"]
            rows.append(entry)
        print(f"seed {seed}: {len(per_fake)} fakes on the chip in >= 3 frames, {sum(r['found'] for r in rows if r['seed'] == seed)} recovered")
    table = pd.DataFrame(rows)
    table.to_csv(paths["processed"] / "column_veto_evaluation.csv", index=False)
    found = table[table["found"]]
    print(f"\nall seeds: {len(table)} fakes, {len(found)} recovered ({len(found) / len(table):.0%})")
    # fakes whose own direction runs along the column, found or not: what share of the injected population a veto could touch
    along = (np.abs((table["angle"] % 180) - 90) <= 10)
    print(f"planted trails within 10 degrees of the column direction: {int(along.sum())} of {len(table)} ({along.mean():.0%}); "
          f"recovered among them: {int(table.loc[along, 'found'].sum())}")
    for angle, spread in VETOS:
        lost = (found["column_angle_difference"] <= angle) & (found["column_spread"] <= spread)
        print(f"veto within {angle} degrees, column spread <= {spread:.0f} px: would remove {int(lost.sum())} of the {len(found)} recovered fakes "
              f"-> recovery {(len(found) - int(lost.sum())) / len(table):.0%} instead of {len(found) / len(table):.0%}")


if __name__ == "__main__":
    main()
