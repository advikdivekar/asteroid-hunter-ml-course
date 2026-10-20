# Phase: More data (round 2)

## Goal
Find out whether training on **more sky** makes the model better. Before this phase, everything came from one CCD quadrant (field 618, CCD 3, quadrant 4) on 8 training nights. Now there are 26 training units: 3 quadrants of field 618 × 8 nights, plus 2 nights of field 567 near opposition (crowded with asteroids).

**Files built or changed:**
- `acquire/list_units.py`: the (field, CCD, quadrant, night) units for a split, including `extra_quadrants` and `extra_sources` from `configs/acquire.yaml`
- `acquire/load_unit_metadata.py`: per-quadrant cached archive table
- `dataset/make_unit_tag.py`: one name per unit (`000618_c03_q4_2019-02-08`). Old one-quadrant file names collided once there were several quadrants, so 120 files were renamed (logged in `data/migration_unit_tags.log`)
- `dataset/make_unit_rng.py`: reproducible random numbers per unit (the original quadrant keeps its old seeds, so old results still reproduce)
- `models/cross_fit_realbogus_scores.py`: folds by **night**, so stage 2 never sees stage-1 scores from a model that trained on that night
- `evaluate/measure_depth_at_thresholds.py`: depth vs alarms at several thresholds (and its test)
- `scripts/train_more_data_bundle.py` → `models/v5`
- `notebooks/scoreboard.ipynb` now evaluates every bundle on every unit of each split. `notebooks/more_data.ipynb` holds the comparison.

## Before (v4) → After (v5)
**Each model at its own chosen threshold (scoreboard, 2026-10-14):**
| split (units) | v4 depth | v5 depth | v4 known | v5 known | v4 alarms/night | v5 alarms/night |
|---|---|---|---|---|---|---|
| validation (6) | 20.49 | 20.42 | 24/28 | 24/28 | 0.0 | 0.0 |
| test (3) | 21.34 | **20.56** | 40/44 | 38/44 | 5.0 | **0.3** |
| holdout (3) | 20.47 | 20.46 | 17/18 | 17/18 | 0.3 | 0.3 |

**The same test units at matched false-alarm levels:**
| alarms per night | v4 depth | v5 depth |
|---|---|---|
| ~10 | 21.38 | 21.28 |
| ~5 | 21.34 | 21.25 |
| ~3 | 21.33 | 21.09 |

**Verdict: more data did not help.** At equal alarms v4 is equal or slightly better. The big "drop" on the test row came mostly from v5's threshold landing at a much stricter point on the same trade-off curve. **v4 stays the live bundle** (`configs/live.yaml`).

## What it taught
1. **Compare models at the same operating point.** A rule like "95% recall on validation" picks different points for different models. One table row hid a tie; the depth-vs-alarms curve showed it.
2. **A near-perfect validation AP (0.997) predicts nothing here.** Most validation positives are planted fakes, many of them bright and easy. The scoreboard measures what matters: depth, known asteroids and alarms on nights nobody tuned on.
3. **Data helps when the model is the bottleneck. It wasn't.** The pipeline is capped by what single 30-second frames can see (~21.4 mag on a good night). More examples can't reveal photons that aren't there. That's why the deep search (+0.7 mag) gained more than any model change.
4. **The plumbing was the real work.** Multi-unit naming, per-unit seeds, cross-fitting by night, metadata per quadrant: none of it raised a score, but without it a mixed dataset leaks or collides silently.

## Quiz
1. Why can a model with a lower depth on the scoreboard still be as good as another?
2. Why must cross-fitting fold by night, not by detection?
3. What evidence says the single-frame pipeline is limited by the data, not by the model?
4. When would more training data be the right fix?
