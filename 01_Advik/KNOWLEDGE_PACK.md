# Advik: knowledge pack

## 0. Read first (in this order)
`00_Start_Here/01_Project_Overview` -> `02_Domain_Primer` -> `03_ML_Primer` (sections D and E especially) -> then the lessons `project/docs/scoreboard.md`, `multiframe_model.md`, `more_data.md`, `deep_search.md`, `fast_movers.md`, `live_monitor.md`, `full_run_mlops.md`.

---

## 1. The system as a whole (architecture and engineering)
**The pipeline** (see `diagrams/architecture_full.svg`): data -> difference images -> detect blobs -> stamps -> CNN -> link -> features -> XGBoost -> cross-match -> candidates.
`run_night_pipeline(unit, metadata, configs, bundle, paths, rng, ...)` runs those stages in order and returns detections, members, tracklets, candidates, injections, known objects.

**One deployed path.** The same function is used (a) with fakes planted, for evaluation, and (b) without, for the real search. *Why it matters:* notebooks that re-implement pipeline steps drift from the real pipeline, so their numbers stop describing what you would run. The scoreboard calls the same function the live system calls.

**Bundles (a mini model registry).** `models/v4/` contains the CNN, the tracklet model, `thresholds.json` (CNN cut 0.0081, tracklet cut 0.601, minimum 3 detections) and `bundle.json` (git commit, training nights, the 17 feature names). *Why together:* mixing a new CNN with an old threshold is a classic production bug. Bundles are never overwritten, so you can roll back.

**Configs, not constants.** Every setting lives in `configs/*.yaml` (night split, thresholds, sizes). *Why:* a result must be tied to its settings; the split is written before modelling so it cannot quietly change after disappointing results.

**Tests.** About 150, all offline (they build tiny fake images). Important ones: the end-to-end pipeline test (3 tiny frames with pointing shifts; fakes linked, never reported as discoveries), the leakage test, the viewer test (the traced run equals the pipeline).

**Provenance.** Which code (git sha in model names), settings (configs), software (`environment.yml`), data (cached archive tables), random fakes (seed = [project seed, night]), thresholds (bundle).

---

## 2. The evaluation framework (your most important topic)
### 2.1 Injection-recovery and completeness
We cannot know the true asteroids in an image, so we **plant** 300 fakes per night (brightness 17-22, speed 5-100 arcsec/hour, any direction), run the *normal* pipeline, and count how many come out. The recovered fraction versus brightness is the **completeness curve**; the magnitude where it crosses 50% is the **depth**.
Fakes are made realistic on purpose: the real PSF of each frame, Poisson noise, added to both science and difference images, real frame times, and, after a bug fix, moving straight **on the sky** (not in each frame's pixels).
**Realism check (adversarial validation):** a classifier asked to tell fakes from real known asteroids (magnitude-matched, cross-validated by night) gets AUC 0.46-0.48; 0.5 means indistinguishable.

### 2.2 The scoreboard
One function, `build_scoreboard`, computes: detection AP and recall, bogus kept per frame, tracklet AP, **depth at 50% completeness**, **known asteroids recovered**, **unexplained alarms per night**. Results are saved to `reports/scoreboard.csv`, one row per (version, split); re-running replaces rows.
*Before the scoreboard:* every notebook computed numbers slightly differently (the same baseline showed 813 and 804 false positives per frame in two lessons). *After:* one definition.

| split | v1 depth | v4 depth | v1 known recovered | v4 known | v4 unexplained/night |
|---|---|---|---|---|---|
| validation (6 units) | 20.18 | 20.49 | 21/28 | 24/28 | 0.0 |
| test (3 units) | 20.53 | 21.34 | 39/44 | 40/44 | 5.0 |
| holdout (3 units) | 20.14 | 20.47 | 16/18 | 17/18 | 0.3 |
(from `reports/scoreboard.csv`; the single-quadrant test-night numbers quoted elsewhere are 20.51 and 21.41.)

### 2.3 Holdout and honesty
Nights are split train / validation / test **by night** in `configs/acquire.yaml`. A **holdout** night (2022-06-17) is opened only by the scoreboard. *Honesty note:* the test night was looked at while fixing real bugs (pointing shift, units, speed noise). Those were physics corrections, not score tuning, but strictly the test night is no longer untouched; that is why the holdout exists and why the holdout numbers match the validation numbers (a good sign nothing was secretly tuned).

### 2.4 Comparing models fairly
- **Same operating point.** At each model's own threshold, v5 looked 0.8 mag shallower on test than v4 (20.56 vs 21.34). At matched false-alarm levels (about 10, 5, 3 alarms per night) v4 was equal or slightly better (21.38 vs 21.28, 21.34 vs 21.25, 21.33 vs 21.09). The "drop" was mostly v5's threshold landing at a stricter point on the same trade-off curve.
- **Metrics with headroom.** Tracklet AP is about 0.999 for every version: saturated, cannot show progress. Depth and known-asteroid recovery have headroom.
- **Measure where the loss is before building a model.** On one training night: 95% stage-1 recall with 4-of-5 linking gave 184/294 fakes linked and 9 bogus tracklets; 3-of-5 gave 200/294 and 74 bogus; 99% recall gave 205/294 and 1,499 bogus. The loss was in linking, not in the classifier.
- **A tie means the simpler model.** Multi-frame CNN vs XGBoost (AP all: 0.9921 vs 0.9924; averaged 0.9933): a tie, so keep XGBoost.

### 2.5 Negative results (say them with pride)
- A fancy model did not beat a simple one.
- More data did not help.
- The v2 system raised 15 unexplained alarms on the dense test night; looking at all 15 by eye showed artifact walkers and noise alignments, which led to two physics features (static neighbours, forced photometry) that cut alarms from 15 to 7 on the single-quadrant test night (13.3 to 5.0 on the three-quadrant scoreboard).
- A bug in your own feature: it counted a slow asteroid's own detections as "static neighbours", flagging 22% of real tracklets; found by checking the feature on real examples; fixed (0.4%); a test was written; v3, trained with the buggy feature, was retired and v4 trained correctly.

---

## 3. The deep search (shift-and-stack)
### 3.1 The idea and the maths
A faint asteroid is below the noise in any single 30-second frame. If you know its motion you can shift each frame so it sits on the same pixel and add them. Signal adds N times; independent noise adds sqrt(N) times. 8 frames: gain 8/2.83 = 2.83x in SNR = 2.5 x log10(2.83) = **+1.13 magnitudes**. (`diagrams/shift_and_stack.svg`)

### 3.2 We do not know the motion, so we try 4,888 of them
Every direction, up to 120 arcsec/hour, in steps of 3 pixels/hour (a half-step over a 0.5-hour stack is a worst-case smear of 0.75 px). On the Mac's GPU. Then we **link** stack detections across the night (a real object must appear in several independent stacks along one straight path): 55 frames over 4.4 hours -> 7 stacks of about 8 frames.

### 3.3 The statistics (the part examiners love)
- **Look-elsewhere effect.** Picking the best of 4,888 motions at every pixel makes chance peaks common. On pure-noise stacks: 6,179 false peaks at 5 sigma but only 44 at 6 sigma, so the cut is **6 sigma** (set from data, not assumed).
- **Real data are not Gaussian.** The first version at 6 sigma still gave about 47,000 peaks per stack. Each fix and its effect: clip each frame at 5 sigma (one bright frame cannot dominate); subtract the night's median per pixel (static residuals no longer stack at any motion) -> 5,765; **per-pixel standardisation** by median *and scatter* (residuals that wobble with the seeing are turned down) -> 391; **mask pixels single frames already see** (otherwise bright asteroids create echo tracklets) -> **94**.
- **Echo test and null test.** Before masking, 151 of 242 unknown tracklets lay within 30 px of a known asteroid (random would be about 190 px away): ghosts. A **null test** (frame times shuffled, so real movers no longer line up) gave 16 false tracklets per night; a **velocity-consistency cut** (each stack's own best motion must match the night-long motion within 6 px/h: known asteroids 2.4 px/h, null tracklets 61 px/h) took the null to **0** while keeping **20 of 20** known asteroids.
- **Calibration:** mag = 26.25 - 5.23 log10(mean stack SNR), scatter 0.17 mag, fitted on 100 found fakes (a straight-line regression).

### 3.4 Result (300 injected fakes, magnitude 20-23, test night)
50% completeness at **22.12** vs 21.41 for the best single-frame system on the same night (+0.7 mag). Recovery: 94-100% at 21.4-21.9, 55% at 21.1 (masked on purpose below ~20.9 because single frames already see them), 48% at 22.1, 0% at >= 22.4.
*Limits:* needs at least 20 frames in a night (8 + 8 + 4 = three stacks); a night with 18 frames silently found nothing until we added the check. It was developed on one night/quadrant.

---

## 4. Fast movers and the live monitor (brief)
- **Fast movers:** near-Earth objects at 300-15,000 arcsec/hour trail during a 30 s exposure (length in px about rate / 120). A matched filter for trails (4-32 px, every 15 degrees); a track is kept only if the trail points along the motion (within 15 degrees), its length equals the speed x exposure (ratio 0.625-1.6) and the length is steady. 97% of planted trails recovered on the nights used to choose the cuts (cuts chosen on the same nights: a stated limit); 93% on a later 10-frame night; 0 false tracks. Real-data lesson: six of seven fast "tracks" we inspected ran along CCD columns (bleed-trail residuals); a veto exists as an option in `configs/fast.yaml`, off by default, costing about 1 point of recovery.
- **Live monitor:** watches the archive, runs point + fast + deep searches on a complete night, remembers what it has done (idempotent), alerts on unknown objects; replay of a past night gave 1 alert and no duplicates on re-run.

---

## 5. The live viewer (your demo)
`python scripts/live_viewer.py` starts a small web server and opens a page. Choose a night, set "fakes", press Start (tick "step by step" to pause after every stage). It runs the **real** functions in the real order and draws each stage: the four files, the noise histogram, planted fakes, blobs (with a zoom), stamps, CNN verdicts, tracks, features, classifier scores, known-asteroid matches, the result. A test proves that on a real night the traced run equals `run_night_pipeline` exactly (max difference 0.0). Design in `diagrams/live_viewer_flow.svg`; code in `src/asthunt/viewer/`.
*Demo tip:* use 8 fakes; the bright ones are found and the faint ones are missed, which shows the depth limit live.

---

## 6. Cross-part cheat sheet (you must be able to say these)
- **Aaryan (data):** ZTF difference images; unit = field+CCD+quadrant+night; downloads are resumable and atomic; split by night; mask bad pixels; noise via MAD; leakage audit (accuracy 100% -> 50% -> 99%).
- **Paras (labels + CNN):** labels by injection, known asteroids (Horizons 0.26 px vs SkyBoT 2.7 px) and "the rest"; 3x31x31 stamps normalised by MAD; CNN with 5 conv layers, 72,145 weights, dropout, early stopping; AP 0.899 vs baseline 0.780; cross-fitting by night.
- **Satvik (linking + trees + metrics):** linking by seed-and-predict with KD-trees on the common sky grid; 17 features; XGBoost (300 trees, depth 3, lr 0.05); models tie in cross-validation; accuracy paradox, PR vs ROC, thresholds chosen on validation, calibration.

## 7. Self-quiz (answers in `VIVA_QA.md`)
1. Why does the scoreboard call `run_night_pipeline` instead of reading saved tracklet files? 2. Why must thresholds live in the same bundle as the models? 3. Why is tracklet AP 0.999 a poor metric for v2? 4. What would stop the holdout night being a holdout? 5. Explain the +1.1 magnitude of an 8-frame stack.
6. Why 6 sigma and not 5? 7. What does the null test prove, and what does it not? 8. Why did v5 look worse and why was that misleading? 9. What is the difference between a null test and an injection test? 10. How does the viewer prove it shows the truth?
