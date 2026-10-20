# Phase: Live monitor (round 2)

## Goal
Turn the project into a system that runs on its own. It notices new nights, searches each one with **all three searches**, and raises an alert for every unknown object.

**Files built:**
- `pipeline/run_full_search.py`: runs the point search (v4 bundle, ≤ 120 "/hr), the fast search (trails, ~480–3,800 "/hr) and the deep search (shift-and-stack, if ≥ 16 frames), and returns one combined candidate table
- `pipeline/run_live_monitor.py`: one monitoring pass, in four steps: query the archive, pick complete nights not yet searched (inside a date window), download, search, append alerts, remember
- `cli.py`: `python -m asthunt.cli search …` / `watch [--once]`
- `configs/live.yaml`: watched units, bundle, frame minimums, "night complete" delay, poll interval, date window
- tests: an offline test with fake archive and search functions (searches only complete new nights, never twice, honours the window); CLI argument checks
- `notebooks/live_monitor.ipynb`: **replay** of a 2019 night through the live system

## What "end to end" covers now
| kind of object | search | depth / range |
|---|---|---|
| slow to normal, bright (main belt, farther) | point search + v4 | 50% at mag ~21.4 (test night), 1–120 "/hr |
| faint, too faint for one frame | deep search | 50% at mag **22.1**, ≤ 120 "/hr |
| fast, trailed (close NEOs) | fast search | 97% of injected trails (mag 15–20), ~480–3,800 "/hr |
| the same object on another night | cross-night linker | 5/5 known asteroids over 3 days, 0 wrong |

## How I thought about it
1. **Separate "when" from "what".** `run_full_search` knows how to search a night; `run_live_monitor` knows when to search one. Each can be tested alone.
2. **Inject dependencies for testing.** The archive query, the download and the search are arguments, so the test runs in 4 seconds with fakes, without a network or GPU.
3. **State makes it idempotent.** `reports/live_state.json` remembers searched nights, so running twice never alerts twice.
4. **Don't search half a night.** A night counts as complete only after 12 hours with no new frames; otherwise the deep search would get half its stacks.
5. **Replay before going live.** Pointing the date window at the past replays old nights through the live code path. Replaying 2019-01-23 produced 1 alert, and a second pass produced none.

## Honest limits
- **"Live" means as soon as the data is public.** ZTF images reach the public IRSA archive after a delay, so a true real-time system would read ZTF's alert stream or a telescope's own feed into `run_full_search`.
- Speeds of 120–480 "/hr fall between the point and fast searches.
- One quadrant takes ~15–20 minutes (most of it the deep search on the GPU). A whole field (64 quadrants) per night needs a bigger GPU or several machines.
- Alerts are lines in a file. Sending them by e-mail or Slack would mean changing one place in `run_live_monitor`.
- Never forward an alert to the MPC without vetting it and confirming it on a second night.

## Quiz
1. Why is the archive query passed as an argument?
2. What would go wrong if `live_state.json` were deleted?
3. Why wait 12 hours before searching a night?
4. What's the difference between replay mode and the real thing?
