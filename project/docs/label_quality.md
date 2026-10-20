# Phase: Label quality (round 2)

## Goal
Two questions every ML project should ask about its own labels:
1. **Are the "real" labels in the right place?** Known-asteroid labels came from SkyBoT, whose positions we measured to be off.
2. **Do the synthetic positives look like the real thing?** Our fakes were assumed realistic, never tested.

**Files built:**
- `label/query_horizons_positions.py`: one object, many times, from JPL Horizons (observatory I41, mid-exposure times)
- `label/refine_known_positions.py`: for one night, refine every object SkyBoT lists (V ≤ 22) and save `known_asteroids_horizons__<exposure>.csv`
- `label/load_known_asteroids.py`: **the single place** that reads known asteroids: Horizons positions where refined, SkyBoT for the rest (each row tagged with its `source`)
- `dataset/relabel_night_detections.py`: re-applies labelling to a saved night when label sources improve (row order preserved, so stamps stay aligned)
- `evaluate/measure_injection_realism.py`: adversarial validation of fakes vs real asteroids
- tests for each, `notebooks/label_quality.ipynb`

---

## Before (v1) → After
| | Before | After | Measured effect |
|---|---|---|---|
| Known-asteroid positions | SkyBoT: median **2.72 px** from where ZTF saw the asteroid (90% within 3.42) | JPL Horizons: median **0.26 px** (90% within 0.79) | 10× more accurate (314 matches, 12 nights) |
| Match radius for known asteroids | 4 px (to cover SkyBoT's error) | 2 px | 4× less area → 4× fewer chance coincidences |
| Faint known objects not refined | n/a | kept with SkyBoT positions, tagged `skybot` | a detected faint known object is never mistaken for a discovery |
| Fixing labels | a one-off script in round 1 | `relabel_night_detections`, tested and idempotent | re-runnable whenever labels improve |
| Labels changed | — | 12 of ~13,000 real labels (5 doubtful removed, 7 real added) | small, so no retraining needed |
| Fake realism | assumed | **AUC 0.46–0.48** (0.5 = indistinguishable) | evidence the fakes look like real asteroids |
| v1 re-measured | test AP 0.938 | test AP 0.944; depth, known recovery unchanged | fair baseline for every later version |

---

## How I thought about it
1. **Measure the error before changing anything.** Horizons vs SkyBoT across all 12 nights: a clear 10× difference, not an anecdote.
2. **Don't create a new failure while fixing the old one.** Refining only V ≤ 22 objects could make a detected faint known asteroid look "unknown", which is a fake discovery. So the loader merges both sources.
3. **Is the improvement big enough to matter for the model?** 12 labels out of 13,000 won't move a retrained CNN. I didn't spend a retraining run on it; I re-measured v1 instead.
4. **Test assumptions with the right comparison.** Raw fake SNR (13) ≠ raw real SNR (7), but only because fakes cover all magnitudes evenly. Magnitude-matched, the difference vanishes.

---

## Concepts

### Systematic vs random error
SkyBoT's offsets were large *and* pointed the same way on every night: a **systematic** error. A wider match radius hides it but adds chance matches. The fix is a better source, not a bigger tolerance.

### Adversarial validation
Train a classifier to separate two datasets that *should* look the same (here: fake vs real asteroid detections).
- **AUC ≈ 0.5:** it can't, so the datasets are alike (on these features).
- **AUC ≫ 0.5:** they differ, and the feature importances show how.

Two rules make it meaningful:
- **Match the confounders:** we sample fakes with the same magnitude distribution as the real asteroids.
- **Cross-validate by night:** otherwise the classifier could learn night-specific quirks.

The same trick is used across ML to detect train/test distribution shift.

### Uncertainty from small samples
With 178 real detections, two random seeds gave 0.481 and 0.463. Report "≈ 0.47 ± 0.02", not three decimals of false precision.

### Re-measure the baseline when the ruler changes
Labels are part of the ruler. After relabelling, v1's scoreboard rows were recomputed (the scoreboard replaces rows, never duplicates), so v1 and later versions are compared on identical labels.

---

## Steps, in order
1. Wrote `query_horizons_positions` (tested live: 1.3 s per object; unknown names → empty table).
2. Wrote `refine_known_positions` and ran it on all 12 nights (19–52 objects per night; 319 of 322 identified).
3. Measured offsets: SkyBoT 2.72 px vs Horizons 0.26 px.
4. Wrote `load_known_asteroids`, first Horizons-only, then **merged with SkyBoT for unrefined objects** (to avoid fake discoveries). Wired it into `build_night_detections` and `run_night_pipeline`. Fixed a test that hadn't written the SkyBoT file real runs always have.
5. Set `known_match_radius_pixels: 2.0`; wrote and ran `relabel_night_detections` (12 labels changed).
6. Wrote `measure_injection_realism` (+ tests: identical populations → chance; a planted giveaway → detected and named) → AUC 0.46–0.48.
7. Re-ran the scoreboard for v1.

## Commands
```bash
pytest tests/label tests/dataset tests/evaluate
jupyter lab notebooks/label_quality.ipynb
jupyter lab notebooks/scoreboard.ipynb     # re-measure after label changes
```

## Pitfalls
- Widening a tolerance to hide a systematic error.
- Fixing a list of objects but silently dropping the ones you didn't fix.
- Comparing fake vs real without matching magnitudes.
- Quoting a small-sample AUC to three decimals.
- Changing labels without re-measuring the baseline.

## Quiz
1. Why is a 4-px radius worse than a 2-px radius, if both find the same asteroids?
2. Why does `load_known_asteroids` keep SkyBoT rows for unrefined objects?
3. What does a realism AUC of 0.85 tell you, and what would you look at next?
4. Why must fakes be magnitude-matched in the realism test?
5. After relabelling, why re-run v1's scoreboard rather than just compare v2 with the old v1 numbers?
