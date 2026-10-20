# Phase: Multi-frame model (round 2)

## Goal
Make the system **deeper**: find asteroids v1 misses. The plan was a model that looks at a whole tracklet (all frames' stamps along the predicted path), like ATLAS's two-stage classifier. The measurements took us somewhere more interesting.

**Files built:**
- `clean/load_night_images.py`: a night's images, with planted fakes (saved by `build_night_detections`) or straight from the archive
- `link/link_detections_into_tracklets.py` (changed): also returns each tracklet's fitted line (`x0, y0, t0_jd`)
- `detect/cut_tracklet_stamps.py`: stamps at the **predicted** position in every frame, detected or not (common grid → sky → frame pixels)
- `dataset/build_tracklet_stamp_dataset.py`, `dataset/augment_stamps.py` (now rotates whole tracklets consistently)
- `models/define_multiframe_model.py`, `train_multiframe_model.py`, `predict_multiframe.py`: shared per-frame CNN + stack + mean/max pooling + features; class weighting via `pos_weight`
- `features/measure_forced_photometry.py`: SNR at the predicted position in every frame; `stack_snr`
- `features/count_static_neighbours.py`: detections left behind at a tracklet's own positions in other frames
- `features/add_path_features.py`: both of the above, **shared by training and the pipeline**
- `pipeline/run_night_pipeline.py` (changed): computes the new features; each bundle names its `feature_columns` and can set `min_detections`
- tests for all of it, `notebooks/multiframe_model.ipynb`; bundles `models/v2`, `models/v4` (v3 retired)

---

## Before (v1) → After (v4): the scoreboard
| split | metric | v1 | v2 | v4 |
|---|---|---|---|---|
| validation | depth (50% completeness) | 20.22 | 20.47 | **20.48** |
| | known asteroids recovered | 8/12 | 10/12 | **10/12** |
| test (opposition, dense) | depth | 20.51 | 21.41 | **21.41** |
| | known recovered | 17/21 | 18/21 | **18/21** |
| | unexplained alarms / night | 0 | 15 | **7** |
| **holdout (never tuned on)** | depth | 20.41 | 20.46 | **20.46** |
| | known recovered | 8/8 | 8/8 | 8/8 |
| | unexplained / night | 0 | 0 | 0 |

**Net:** deeper everywhere (most on the sharp, dark test night: +0.9 mag), more real asteroids found, and on the hardest night a vetting load of ~7 artifacts, down from 15.

---

## How I thought about it (in the order things happened)
1. **Find where the asteroids are lost, before building a model.** On one training night:

   | stage-1 recall | min detections | fakes linked | bogus tracklets |
   |---|---|---|---|
   | 95% | 4 of 5 (v1) | 184 / 294 | 9 |
   | 95% | 3 of 5 | 200 / 294 | 74 |
   | 99% | 3 of 5 | **205 / 294** | 1,499 |

   Relaxed linking finds 11% more asteroids. The price is ~1,500 chance alignments per night, which are also the realistic hard negatives v1 never had.
2. **Build the ambitious model.** A multi-frame CNN sees stamps at the predicted position in *all* frames, plus their average (a learned mini shift-and-stack). In a synthetic test it learns a blob of only **1σ per pixel per frame**.
3. **Benchmark honestly, on identical tracklets:**

   | model | AP all | AP 3-det | AP 4-det | AP 5-det |
   |---|---|---|---|---|
   | XGBoost, v1 features | 0.9923 | 0.769 | 0.964 | 0.9989 |
   | XGBoost + forced photometry | 0.9924 | 0.761 | 0.960 | 0.9990 |
   | multi-frame CNN | 0.9921 | 0.708 | 0.973 | 0.9992 |
   | average of the two | 0.9933 | 0.762 | 0.970 | 0.9992 |

   A tie. The 3-detection column has only 33 real tracklets, so its differences are noise. **The classifier was not the bottleneck; linking was.** Rule: *use the simplest model that ties*. → **v2** = XGBoost retrained on the harder tracklets + relaxed linking.
4. **The scoreboard exposed v2's weakness:** 15 unexplained alarms on the dense test night. **Vetting by eye:** all 15 bogus: *artifact walkers* (slow tracks along a faint line present in every frame) and *noise alignments* (three ~2σ blips, nothing at the predicted spot elsewhere).
5. **Physics features for each failure mode:** static neighbours (artifact walkers) and forced photometry (noise alignments). On the test night: known asteroids 0.0 / +6.4, bogus 1.0 / −0.4.
6. **A bug in my own feature:** it counted a slow asteroid's *own* other detections as "static neighbours", flagging **22% of real tracklets**. Caught by checking the feature on real examples before trusting a rule; fixed (0.4%); test added that reproduces it.
7. **Versioning consequence:** v3 had been trained with the buggy feature. Re-running it with fixed code would feed it inputs it never saw, so **v3 was retired** (moved, with a note; its scoreboard rows kept as history) and **v4** trained with the correct feature.
8. **What's left:** 7 alarms, all slow artifact walkers. The model learned the static feature only weakly because the September training nights contain few such artifacts: **distribution shift**. That's the next phase's job (more, and more varied, training nights).

---

## Concepts
- **Find the bottleneck before optimising.** A better classifier can't recover asteroids the linker never produces.
- **Benchmark on identical data, read the error bars.** 33 positives can't separate 0.71 from 0.77.
- **Simplest model that ties.** Fewer parts, faster, easier to explain and maintain. The CNN stays in the repo as a documented result.
- **Failure-mode-driven feature engineering.** Look at the errors, name their physical cause, encode that cause as a number.
- **Check a feature on the positive class.** If a "bogus indicator" fires on 22% of real objects, the feature is broken, not the data.
- **Code and model are versioned together.** Feature code is part of the model. Change it → new version.
- **Honesty about test data.** The test night was used to find failure modes, so the **holdout** is now the clean judge.
- **Hardware matters:** Apple's GPU (MPS) has no float64; a test on CPU didn't catch it.

## Steps, in order
1. Saved planted images during the build; rebuilt 12 nights (bit-for-bit identical, verified).
2. Linker returns fitted lines; wrote and tested `cut_tracklet_stamps` (incl. pointing shifts).
3. Wrote the multi-frame model, trainer (fixed an MPS float64 error) and predictor; tests.
4. Built tracklet datasets (3 of 5, 99% stage-1): 7,580 training / 877 validation tracklets.
5. Benchmarked four scorers → tie → v2 = XGBoost; pipeline reads `min_detections` from the bundle.
6. Scoreboard: v2 deeper, 15 alarms on the test night → vetted → two physics features → shared helper, pipeline integration.
7. Found and fixed the own-detection bug; retired v3; trained v4; scoreboard.

## Commands
```bash
pytest tests/features tests/models tests/detect tests/pipeline
jupyter lab notebooks/multiframe_model.ipynb
jupyter lab notebooks/scoreboard.ipynb
```

## Quiz
1. Why did relaxing linking help more than a better classifier?
2. The multi-frame CNN is "more powerful". Why didn't we deploy it?
3. What physical situation creates an "artifact walker", and which feature catches it?
4. How did checking the feature on *real* tracklets reveal the bug?
5. Why can't v3 simply be re-evaluated with the fixed code?
6. v4 still has 7 alarms on the test night. Why is more training data, rather than a hand-written rule, the better fix?
