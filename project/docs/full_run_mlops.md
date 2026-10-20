# Phase: Full run and MLOps

## Goal
1. **One** honest evaluation on the untouched test night (2019-02-08), using the planted fakes.
2. The real **search** on that night (no fakes).
3. The engineering that makes the result trustworthy: provenance, reproducibility, a model card.

**Files built:** `notebooks/full_run_mlops.ipynb` (it uses `pipeline/run_night_pipeline.py`), this document, and `reports/2019-02-08/` (HTML report + draft ADES file).

---

## How I thought about it
1. **Everything was frozen before this phase:** models, thresholds (`thresholds.json`), configs. The test night is evaluated, not tuned on.
2. **Then run the real search and compare against known reality:** how many SkyBoT asteroids did we find by ourselves?
3. **When the search disagreed with expectations, investigate, don't tune.** The first test-night search recovered only 1 known asteroid. That exposed a real bug (pointing shifts) in data preparation, not a threshold problem. After fixing the bug, *everything* was rebuilt and retrained from scratch with the same, still-untouched protocol.

**Honesty note:** we looked at test-night results, found bugs, and fixed them. Strictly, the test night is no longer perfectly "unseen". The fixes were physics corrections (coordinates, units, speed noise), not tuning to raise a score. To be fully clean, evaluate on another night that has never been opened (see the exercises).

---

## Results on the test night (2019-02-08, opposition, 1.6 px seeing)

### With planted fakes (evaluation)
| stage | result |
|---|---|
| real/bogus CNN | AP **0.938**; 97.5% of real detections kept at the linking threshold |
| tracklet classifier | AP 0.988, recall 0.887; 9 "bogus" tracklets kept, **all 9 real known asteroids** our labels missed → effective false positives: **0** |

End-to-end completeness (planted asteroid → final candidate):

| magnitude | 17–18 | 18–19 | 19–19.5 | 19.5–20 | 20–20.5 | 20.5–21 | 21–21.5 | 21.5–22 |
|---|---|---|---|---|---|---|---|---|
| fraction recovered | 0.93–0.96 | 0.97–1.00 | 0.91 | 0.89 | 0.79 | 0.29 | 0.03 | 0.00 |

**50% completeness at about mag 20.5.** The ~5–10% loss even for bright fakes comes from masked pixels and fakes slower than 8 "/hr (excluded by design).

### The real search (no fakes)
- 42,149 detections → **50 tracklets → 31 known asteroids recovered**, with measured r magnitudes from 16.2 down to 21.8.
- Known asteroids present in ≥ 4 frames, recovered by brightness:

| V mag | ≤ 19.5 | 19.5–20 | 20–20.5 | 20.5–21 | 21–22 | > 22 |
|---|---|---|---|---|---|---|
| recovered | 8 / 8 | 4 / 5 | 6 / 8 | 6 / 6 | 7 / 24 | 0 / 20 |

Real asteroids and planted fakes agree: both fade out between mag 20.5 and 21.5. That's the strongest evidence our simulation is now realistic.

- **Unknown candidates above the threshold: 0.** The 19 unknown tracklets below it were mostly stationary artifacts. The three most interesting were vetted by eye: a galaxy residual, satellite streaks, a saturated-star bleed (see `docs/crossmatch_vetting.md`).

**Conclusion:** no new asteroid on this night. The pipeline independently finds the known population down to ~mag 21 with essentially no false alarms. That's exactly what you need before a real discovery claim could ever be trusted.

---

## MLOps: what makes this reproducible

### Provenance chain
| question | answered by |
|---|---|
| Which code? | git commit in every model file name: `__fef32b9` (CNN), `__70280ed` (tracklet classifier, retrained after the linker fix so its name points at the code that made it) |
| Which settings? | `configs/*.yaml` in that commit |
| Which software? | `environment.yml` (PyTorch from conda-forge) |
| Which exposures? | `data/raw/ztf/metadata__000618.csv` + `configs/acquire.yaml` |
| Which random fakes? | seed = [project seed 42, night date] |
| Which known asteroids? | SkyBoT answers cached in `data/interim/known_asteroids/` |
| Which model, trained how? | `models/<name>.json` (nights, settings, training history) |
| Which thresholds? | `data/processed/thresholds.json` |

Superseded outputs (before the sky-coordinate fix) were **moved, not deleted**, to `data/superseded/before_sky_fix/`.

### Reproduce everything
```bash
conda env create -f environment.yml && conda activate asthunt
python -m ipykernel install --user --name asthunt
pytest                                                   # 62 tests
# then run the notebooks in PLAN.md order; each one re-uses what earlier ones saved:
#   data_acquisition → dataset_building → realbogus_cnn → linking → tracklet_classifier → full_run_mlops
```
On an M3 Pro: download ~1 h (archive speed), dataset build ~1.5 min (about 8 min the first time, while SkyBoT is queried), CNN + cross-fitting ~7 min, everything else seconds.

### Tests as a safety net
62 tests, each on tiny synthetic data. Several exist *because* a bug was found:
- `test_jittery_stationary_source_is_not_linked`: speed checked on the fitted track
- `test_one_fake_night_end_to_end` with a pointing shift: fakes planted via the sky
- `test_works_with_no_fakes_and_no_known_asteroids`: rewritten until it actually failed without the fix
- `test_known_asteroids_can_use_a_wider_radius`: the SkyBoT offset

---

## Model card

**Models:** `realbogus_cnn__20261013__fef32b9` (CNN, 72,145 weights) + `tracklet_classifier__20261013__70280ed` (XGBoost, 300 depth-3 trees).

**Intended use:** ranking candidate moving objects in ZTF r-band difference images of field 618, CCD 3, quadrant 4, for **human vetting**. Not for automatic reporting to the MPC.

**Training data:** 8 nights (Sep 2020–2023), 5 exposures each. Positives: planted fakes (mag 17–22, 5–100 "/hr, ZTF PSF, Poisson noise, planted via sky coordinates) plus SkyBoT known asteroids (V ≤ 20.5, 4-px match). Negatives: all other detections (~33 per real one), balanced to 5:1 for the CNN. Validation: 2 other nights. Test: 2019-02-08.

**Performance:** see results above (CNN validation AP 0.899, test AP 0.938; 50% end-to-end completeness at mag ~20.5 on the test night; 0 confirmed false alarms on the test night).

**Known limitations:**
- One field, one CCD quadrant, one filter. Other fields (crowded Milky Way, different backgrounds) are untested.
- Fakes are point sources: fast near-Earth objects that trail (> ~150 "/hr) and objects slower than 8 "/hr are out of scope.
- Satellite streaks and galaxy residuals can still form tracklets; the classifier scores them low, but a feature for streaks would help.
- Labels: "bogus" contains real but unlabelled objects (faint or unmatched known asteroids, unknown asteroids, variable stars).
- Known-asteroid labels rely on SkyBoT, whose predictions were offset by ~2.9" for these data.
- The test night was opened while debugging (see the honesty note).

**Ethics/process:** never submit to the MPC without human vetting and a second-night confirmation. Archival ZTF observations are ZTF's (observatory I41).

---

## Steps, in order
1. Loaded the newest models and frozen thresholds.
2. Evaluated once on the test night with fakes.
3. Ran the search without fakes → found the pointing-shift bug → fixed it in data preparation → rebuilt and retrained everything (outputs archived, not deleted).
4. Found stationary "slow movers" → fitted-speed check + 8 "/hr minimum → re-linked and retrained the tracklet classifier.
5. Final run: 31 known asteroids recovered, 0 candidates above threshold; vetted the top unknowns by eye.
6. Wrote the provenance table and model card.

## Pitfalls
- Tuning on the test night (we documented exactly what changed and why).
- Deleting superseded results (move them instead).
- Choosing "the newest model" by sorting file names that contain git hashes (we use file time).
- Two notebooks writing the same config file (one now updates instead of overwriting).
- Saving a model while code changes are uncommitted: its name then points at the wrong commit. We retrained the tracklet classifier after committing (identical results).

## Quiz
1. Why is recovering 31 known asteroids a stronger result than any score on fakes?
2. Fakes and real asteroids both fade between mag 20.5 and 21.5. Why does that matter?
3. List four things needed to reproduce a model exactly.
4. We fixed bugs after opening the test night. What would make the evaluation fully clean again?
5. Name two limitations from the model card you would fix first, and how.

## Exercises
1. Pick a night that's in no split (e.g. 2022-06-17), add it to `configs/acquire.yaml` under a new `holdout` key, download it, and run the search. That's a truly unseen evaluation.
2. Add a "streakiness" feature (elongation of the difference stamp along the motion direction) and see whether #44-type tracklets drop.
3. Run the search on the remaining 50 exposures of 2019-02-08 in groups of 5. Do any unknown tracklets repeat at consistent positions? That's how a second sighting would look.
