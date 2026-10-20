# Phase: Scoreboard (round 2)

## Goal
Before improving anything, decide **how we'll know it's better**. One function computes every headline number, on fixed nights, from the same pipeline we deploy. Every version of the system is saved as a bundle and gets one row per split.

**Files built:**
- `models/save_model_bundle.py`, `models/load_model_bundle.py`: one folder per system version (`models/v1/`: CNN + tracklet classifier + thresholds + `bundle.json` with commit and training nights); bundles are never overwritten
- `pipeline/run_night_pipeline.py` (changed): takes a bundle; can plant fakes and label (`with_injections=True`) or not; can skip writing reports
- `evaluate/find_magnitude_at_completeness.py`: interpolates the 50%-completeness magnitude
- `evaluate/build_scoreboard.py`: the headline numbers, defined once
- `evaluate/save_scoreboard_rows.py`: `reports/scoreboard.csv`, one row per (version, split); re-runs replace, never duplicate
- tests for all of them, including the **first end-to-end test of the deployed pipeline**; `notebooks/scoreboard.ipynb`

---

## Before (v1) → After (round 2)
| | Before | After | Why it matters |
|---|---|---|---|
| Where numbers came from | each notebook computed its own, slightly differently (the baseline showed 813 vs 804 false positives per frame in two lessons) | one function, `build_scoreboard` | differences between versions are real, not bookkeeping |
| What got evaluated | saved intermediate files (tracklets from notebooks) | **the deployed `run_night_pipeline`**, with fakes planted | you measure exactly what you'd run in production |
| Models | `models/<name>__<date>__<sha>` files + a shared `thresholds.json` that notebooks overwrote | `models/v1/` bundles: models + thresholds + provenance together, never overwritten | roll back or compare by loading a different folder |
| Unseen data | the test night was opened while debugging | a **holdout night (2022-06-17)**, opened only by the scoreboard | an honest final number |
| Pipeline tests | none for the full pipeline | `test_run_night_pipeline.py`: 3 tiny fake frames with pointing shifts → fakes linked, never reported as candidates | the most important code path is protected |

---

## v1's scoreboard (the baseline every later version must beat)
| split | nights | detection AP | detection recall | bogus kept / frame | tracklet AP | **depth (50%)** | **known recovered** | **unexplained / night** |
|---|---|---|---|---|---|---|---|---|
| validation | 2 | 0.899 | 0.950 | 1,250 | 0.9997 | 20.22 | 8 / 12 | 0 |
| test | 1 | 0.938 | 0.975 | 1,194 | 0.9986 | 20.51 | 17 / 21 | 0 |
| **holdout** | 1 | **0.906** | 0.964 | 1,483 | 0.9997 | **20.41** | **8 / 8** | 0 |

Reading it:
- **Holdout ≈ validation ≈ test.** v1 wasn't secretly tuned to the test night. That's good news, and now it's measured, not assumed.
- **Depth ~20.2–20.5 mag** is the number to push. It's set by single-frame detection: an asteroid must be visible in ≥ 4 of 5 frames.
- **Tracklet AP ≈ 0.999 is saturated.** Further gains there are invisible. The improvements that matter will show up in **depth** and in **known recovered** (validation misses 4 of 12).
- **0 unexplained alarms per night** is good for a human's workload. It also means v1 never proposes anything new, consistent with "cleaner, not deeper".

---

## Concepts
- **Evaluation protocol before optimisation.** Fix the nights, metrics and definitions first, then change the system. Otherwise you end up redefining success to match whatever you built.
- **Holdout discipline.** A holdout is valuable only while nothing has been tuned on it. We open it in exactly one place, write the number down, and never use it to choose settings.
- **Model registry (by folder).** A version = all the parts that must travel together (both models *and* their thresholds). Mixing a new CNN with an old threshold is a classic production bug.
- **Evaluate the deployed path.** Notebooks that re-implement pipeline steps drift from the real pipeline. Our scoreboard calls the same function the live system will call.
- **Saturated metrics.** When a metric is near perfect, it can't show improvement. Pick metrics with headroom (depth, known recovery).

## Steps, in order
1. Started round-2 downloads in the background (holdout night first).
2. Wrote bundle save/load + test (save → load gives identical models; refuses to overwrite).
3. Rewrote `run_night_pipeline` to take a bundle and return fakes, known asteroids and tracklet labels; added an end-to-end test.
4. Wrote `find_magnitude_at_completeness`, `build_scoreboard`, `save_scoreboard_rows` + tests.
5. Froze the existing models as `models/v1/`; switched the two notebooks to the bundle (identical results: 10 and 31 known asteroids, 0 candidates).
6. Ran the scoreboard on validation, test and holdout → `reports/scoreboard.csv`.

## Commands
```bash
pytest tests/evaluate tests/models tests/pipeline
jupyter lab notebooks/scoreboard.ipynb
cat reports/scoreboard.csv
```

## Quiz
1. Why must thresholds live in the same bundle as the models?
2. Why does the scoreboard call `run_night_pipeline` instead of reading saved tracklet files?
3. Tracklet AP is 0.999 for v1. Why is that a poor metric for judging v2?
4. What would make the holdout night stop being a holdout?
5. v1 has 0 unexplained alarms per night. Is that the ideal number for a discovery system? Why or why not?
