# Plan: "Asteroid Hunter": ML asteroid discovery in archival ZTF data (taught step by step)

## Context
The post describes someone finding a candidate planet by re-mining old public telescope data (Kepler) with AI-assisted code. They pulled apart blended pixels, searched for a repeating signal, and vetted it. You chose to apply the same idea to **your existing ZTF asteroid project**: dig through archival Zwicky Transient Facility images and find moving objects that are not in the known-asteroid catalogues. You also want to learn ML fundamentals and ML engineering while we build it.

**What already exists in `~/Desktop/asteroid`:**
- `ztf_test/sci/2018/...` (now `data/raw/ztf_test/`): **10** ZTF science images of the same field, CCD and quadrant (`000618_zg_c03_q4`), spread over Mar to Nov 2018. Only one pair shares a night (2018-04-28, about 1.3 h apart), which is too few frames to link a tracklet.
- `asteroid.py`: queries SkyBoT for known asteroids in an image and overlays them. This is the coordinate sanity check, and we keep it.
- `inject.py`, `detect.py`, `link.py`: inject a fake asteroid with a GalSim Moffat PSF, detect sources with SEP, and link 4 detections into a straight-line tracklet.
- `generate_dataset.py` → `tracklets.csv`: 3,530 rows, 50/50 real and bogus, with features `measured_rate`, `linearity_residual` and x/y positions. **No model exists yet.**
- Environment: base miniconda (Py 3.14) has numpy, pandas, astropy, sep, galsim, astroquery, sklearn and xgboost, but no torch. Machine: M3 Pro, 109 GB free.

**Problems in the current setup.** These are the first teaching moments:
1. **The dataset leaks the answer.** Bogus tracklets come from 4 *random* sources anywhere on the chip, so their `linearity_residual` is at least 222 px while real ones stay at or below 2.6 px, and the rates are about 2000 vs about 30. Any model will score close to 100%, and that score means nothing. Real false tracklets are *near-misses*, so we need hard negatives.
2. **The time baseline is fake.** The 4 frames come from different months, but the code treats them as 15 minutes apart. Real asteroids move too far between nights. Discovery needs **several exposures of the same field on the same night**, which ZTF does routinely.
3. **There's no difference imaging.** Detecting on raw science images means every star is a candidate. Real surveys subtract a reference image first, so only things that changed are left.
4. Paths are hardcoded, `sys.path` hacks are used, and functions are copy-pasted across 4 files.

**A note on the post, and a lesson in skepticism:** KIC 5689121 is not in the literature, but its catalogue neighbour **KIC 5689351 (KOI-505)** already has a catalogued **13.77-day** candidate ([arXiv:1205.6492](https://arxiv.org/pdf/1205.6492)). Neighbouring KIC numbers are often close together on the sky, and the post itself mentions blended light from nearby stars. So the "new planet" may be leaked light from a known one. This *contamination / blend false positive* is exactly the trap our crossmatch and vetting stages must catch. One short side-notebook will check it.

*Update after checking (see `docs/kepler_post_check.md`):* KOI-505.01 is the **confirmed** planet Kepler-169 e (P = 13.7671 d), 263" away. The period match has odds of about 1 in 350 by chance. But simple blending can't produce the post's 0.24% dip from a 0.068% transit. So the verdict is "suspicious, not settled": the decisive test is comparing transit times.

## Architecture

```
            ┌──────────── DATA LAYER ────────────┐
 IRSA ZTF   │ archive.py: query metadata (field, │
 archive ──▶│ ccd, quad, night) → download sci,  │──▶ data/raw/  (gitignored)
            │ ref and diff images + masks        │
            └────────────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────┐
   │ 1. PREPROCESS  calibrate.py: mask bad pixels, use   │
   │    ZTF diff images (sci − ref), read PSF/zeropoint  │
   ├──────────────────────────────────────────────────────┤
   │ 2. INJECT (truth)  inject.py: fake asteroids with    │
   │    real motion into same-night frames → labels       │
   ├──────────────────────────────────────────────────────┤
   │ 3. DETECT  detect.py: SEP on diff images → sources   │
   │    + 63×63 cutout triplets (sci / ref / diff)        │
   ├──────────────────────────────────────────────────────┤
   │ 4. REAL/BOGUS ML #1  CNN on triplets → p(real)       │  ← deep learning
   ├──────────────────────────────────────────────────────┤
   │ 5. LINK  linking.py: KD-tree + linear-motion fit     │
   │    across N same-night frames → candidate tracklets  │
   ├──────────────────────────────────────────────────────┤
   │ 6. TRACKLET ML #2  gradient boosting on features     │  ← classic ML
   │    (rate, residual, mag scatter, mean p(real), PA)   │
   ├──────────────────────────────────────────────────────┤
   │ 7. CROSSMATCH  SkyBoT/MPC known objects (reuse       │
   │    asteroid.py logic) → known vs UNKNOWN             │
   ├──────────────────────────────────────────────────────┤
   │ 8. VET & REPORT  stamps, light/astrometry check,     │
   │    ADES-format report (manual MPC submission only)   │
   └──────────────────────────────────────────────────────┘
 Evaluation spans all stages: injection-recovery completeness vs magnitude and rate,
 false-positive rate per field, and sanity checks against known SkyBoT asteroids.
```

**File rule: one purpose per file, named for that purpose.**
- Code file = `verb_object.py`. Each file has **one public function with the same name as the file** (`extract_sources.py` → `extract_sources()`). Private helpers start with `_`. If a file needs a second public function, split it into a new file.
- Folders are named after the **pipeline stage** (a noun). Tests mirror the files: `tests/<stage>/test_<file>.py`.
- Notebooks and lesson docs: `<phase_name>.ipynb` / `<phase_name>.md`, with no numbers anywhere. Configs: `<stage>.yaml`.
- Data artifacts: `<stage>__<field>_<night>.<ext>` (e.g. `detections__000618_20180916.parquet`). Models: `<model>__<YYYYMMDD>__<git-sha>.pt`.

**Repo layout (as built).** Old scripts live in `legacy/`, untouched, for reference.
```
asteroid/
  environment.yml  pyproject.toml  README.md  PLAN.md  .gitignore
  configs/{project,acquire,inject,detect,dataset,train,link}.yaml
  src/asthunt/
    config/     load_config.py
    acquire/    query_ztf_metadata.py  select_night_exposures.py  download_ztf_images.py  download_ztf_reference_image.py
    clean/      read_fits_image.py  build_bad_pixel_mask.py  estimate_background.py  load_exposure.py
    detect/     extract_sources.py  cut_triplet_stamps.py
    simulate/   render_point_source.py  plan_injections.py  inject_point_source.py
    label/      match_detections_to_positions.py  query_known_asteroids.py  label_detections.py  label_tracklets.py
    dataset/    build_night_detections.py  normalize_stamps.py  augment_stamps.py  balance_classes.py
                assemble_stamp_dataset.py  build_tracklet_dataset.py
    evaluate/   check_split_leakage.py  compute_classification_metrics.py  choose_threshold.py
                calibrate_probabilities.py  adjust_for_base_rate.py  measure_completeness.py
    models/     train_baseline_classifier.py  define_realbogus_cnn.py  train_realbogus_cnn.py  predict_realbogus.py
                cross_fit_realbogus_scores.py  train_tracklet_classifier.py
    link/       fit_linear_motion.py  link_detections_into_tracklets.py
    features/   compute_tracklet_features.py
    crossmatch/ crossmatch_known_asteroids.py
    report/     write_candidate_report.py  write_ades_report.py
    pipeline/   run_night_pipeline.py
  notebooks/<phase_name>.ipynb      docs/<phase_name>.md
  tests/<stage>/test_<file>.py
  data/{raw,interim,processed}/   models/   reports/   (gitignored)
  legacy/  (original scripts, tracklets.csv, pngs)
```
Changes from the first draft, and why:
- ZTF's own PSF model replaced the Moffat profile → `render_point_source.py`, `inject_point_source.py`.
- Hard negatives come from the real linker output (mixed tracklets), so `mine_hard_negatives.py` became `label_tracklets.py`.
- The split lives in `configs/acquire.yaml`, so `split_by_night.py` wasn't needed (`check_split_leakage.py` verifies it).
- `cross_fit_realbogus_scores.py` was added so the tracklet classifier never trains on over-confident CNN scores.
- Completeness is returned as a table (`measure_completeness.py`) and plotted in notebooks.

**Data processes taught as their own stage.** Each one is a file above:
acquire → QA/clean → label (injection, SkyBoT, hard negatives) → explore (EDA) → feature engineering → normalize → split by night/field (leakage-safe) → balance → augment → version (artifact naming + config hash).
Code we reuse and clean up into the package: `inject_asteroid` and `load_image` (inject.py / generate_dataset.py), `detect_sources`, `linearity_residual` and `angular_rate` (generate_dataset.py), and the SkyBoT query plus WCS→pixel logic (asteroid.py).

## How we work: one phase at a time
- **No numbers in any file name.** The order lives in `PLAN.md` only. Files are named by purpose: notebooks `<topic>.ipynb`, docs `<phase_name>.md`.
- **Only one phase is active at a time.** We finish it completely before opening the next one.
- Every phase ends with three things:
  1. **Code:** the one-purpose files for that stage, plus their tests.
  2. **A notebook** `notebooks/<phase_name>.ipynb`: runnable, with plots, where you experiment.
  3. **A lesson doc** `docs/<phase_name>.md` covering: the goal, how I reasoned about it, the concepts (with a small example), the exact steps in order, the commands to reproduce it, the pitfalls, and a short quiz.
- You confirm a phase is done (quiz answered, tests green) before we move on.

## Phases (in order)
The order changed during the build: linking needs CNN scores, and the tracklet classifier needs linked tracklets.

| Phase | Notebook / lesson doc | What you learn | What gets built |
|---|---|---|---|
| Project setup | `project_setup` | envs, git, packaging, configs, tests | env `asthunt`, git repo, skeleton, `legacy/` |
| ML basics on old data | `ml_basics` | features, labels, models, train/test, logistic regression | baseline on `legacy/tracklets.csv` |
| Leakage audit | `leakage_audit` | leakage, shortcut learning, distribution shift | `check_split_leakage.py`, hard-negative spec |
| Astronomy data primer | `astronomy_data_primer` | FITS, WCS, PSF, zeropoints, noise | `clean/` files |
| Data acquisition | `data_acquisition` | metadata queries, caching, provenance | `acquire/` files, one real night |
| Difference imaging and detection | `difference_imaging` | subtraction, artifacts as the bogus class | `detect/` files |
| Injection and labelling | `injection_labelling` | synthetic labels, label noise, hard negatives | `simulate/` + `label/` files |
| Dataset building | `dataset_building` | normalize, split by night, balance, augment, versioning | `dataset/` files |
| Metrics and evaluation | `metrics_evaluation` | precision/recall, PR curves, base rates, completeness, calibration | `evaluate/` files |
| Neural networks | `neural_networks` | loss, gradient descent, backprop (numpy → PyTorch), overfitting | from-scratch network (notebook) |
| Real/bogus CNN | `realbogus_cnn` | convolutions, augmentation, early stopping, cross-fitting | CNN on stamp triplets + per-night scores |
| Linking | `linking` | search vs learning, KD-trees, seed-and-predict | `link/` files, tracklets for every night |
| Tracklet classifier | `tracklet_classifier` | trees, forests, boosting, cross-validation by night | `features/` + XGBoost model + thresholds |
| Crossmatch and vetting | `crossmatch_vetting` | known-object matching, look-elsewhere effect, human review | `crossmatch/` + `report/` files |
| Full run and MLOps | `full_run_mlops` | one-time test evaluation, end-to-end search, provenance, model card | `run_night_pipeline.py` + final report |
| Side check of the Kepler post | `kepler_post_check` | skepticism: blends and contamination | short notebook |

**Field nitpicks woven into the lessons:** injected fakes vs real distribution shift; label noise from unknown real asteroids hiding in "bogus"; stars, cosmic rays, satellites and ghost artifacts; trailing for fast NEOs; astrometric precision and why MPC requires ≥2 nights; ZTF's own pipeline (ZMODE) has already mined this data, so a genuinely new object will most likely be **faint (near the limit) or fast (streaked)**, which is where we focus; never submit to MPC without manual vetting.

## Honest expectations
- The pipeline and the learning are guaranteed. A real discovery is not: ZTF data is heavily mined. Our realistic edge is faint, low-SNR tracklets that ML can rescue below the classical threshold.
- "Unknown" first means "unmatched in SkyBoT". The usual explanations are artifacts or catalogue gaps, and the vetting notebook teaches how to tell these apart.

## Verification
- `pytest tests/`: injection flux matches the requested magnitude within ~0.1 mag; the linker recovers synthetic straight-line tracks; there's no train/test overlap across fields/nights (a leakage test).
- **Known-asteroid check:** run on a real night; ≥90% of SkyBoT asteroids brighter than about mag 19.5 inside the frames get recovered as tracklets.
- **Injection-recovery:** a completeness vs magnitude curve; the ML pipeline must beat the classical cut-based pipeline at the faint end at an equal false-positive rate.
- Held-out evaluation always on a **different night/field** than training.
- Final: `asthunt run` on a fresh night produces `reports/<night>/candidates.html` with stamps for each unknown candidate.

## First execution step after approval
**Project setup phase only:** copy this plan to `PLAN.md`, create the `asthunt` conda env (Python 3.12 + the astro/ML stack), `git init`, move the old scripts into `legacy/`, scaffold the folders, then write `docs/project_setup.md` and `notebooks/project_setup.ipynb`. Then stop, teach that phase, and wait for your go-ahead.

---

# Round 2: make the system better, then hunt (plan, 2026-10-13)

**Decided:** no app yet; IASC data is optional (applied, not received); priority = a better system and model aimed at a real discovery on public ZTF data; publication later. Every phase's lesson has a **"Before (v1) → After (v2)"** section: what we did first, why it fell short, what changed, and the measured difference on the same scoreboard.

## Why v1 can't discover anything (the measured facts)
- 50% completeness at mag ~20.5 ≈ the single-frame limit: the ML makes v1 **cleaner, not deeper**. ZTF's own system has already mined everything at that depth.
- The single-detection CNN never sees the frames together; the tracklet classifier scores look perfect only because fakes dominate.
- One field, one quadrant, 11 nights; known-asteroid labels from SkyBoT (~2.9" off); the test night was opened while debugging.

## Phases (one at a time; each ends with code + tests, `docs/<phase>.md`, `notebooks/<phase>.ipynb`, commit)
| Phase | What | Why / how we'll know it's better |
|---|---|---|
| `scoreboard` | One notebook + `evaluate/build_scoreboard.py` that computes every headline number from saved files: CNN AP, end-to-end 50%-completeness magnitude, known-asteroid recovery by magnitude, bogus candidates per night. Freeze v1's row. Add a **never-opened holdout night**. | Every later change is measured the same way, on data nobody tuned on |
| `label_quality` | Known-asteroid positions from **JPL Horizons** (agrees with our detections to 0.2") instead of SkyBoT's offset ones; **injected-vs-real test** (a classifier that can't tell our fakes from real known asteroids = realistic fakes) | Cleaner labels → honest metrics; proof the fakes look real |
| `multiframe_model` | ATLAS-style **tracklet model**: the stamps from all frames of a tracklet → small per-frame CNN → combine → one score. Harder negatives: link with a looser CNN cut, so there are many more realistic false tracklets. | Compare against v1 (per-detection CNN + XGBoost) on the scoreboard |
| `more_data` | More CCD quadrants of field 618 and more nights; split by quadrant and night | Shows the model generalises beyond one patch of sky |
| `deep_search` | **Stack-then-link** on many-exposure nights (2019-02-08 has 55 frames over 4.4 h): stack ~12 consecutive difference images along plausible main-belt motions (about 1,200 velocities, feasible on the M3 GPU), detect peaks the single frames can't see, then link stack detections across the night as an independent confirmation | Expected depth gain ~+1.3 mag per 12-frame stack, beyond ZTF's per-frame limit, which is where an unmined asteroid could hide. Measured with injections. |
| `hunt` | Find fields with many exposures on **≥2 nights close together** (the MPC needs two nights), run the deep search, cross-match with Horizons/SkyBoT, vet with the checklist | The discovery attempt |

**Later (publication path, not now):** non-differenced four-frame mode for IASC packets; 2D completeness map (magnitude × speed); the classical-vs-CNN benchmark write-up.

## Added 2026-10-13: "detect live, end to end; fast, slow, close, far"
| Phase | What |
|---|---|
| `fast_movers` | trailed matched filters for NEOs (~480–3,800 "/hr) + physical consistency cuts |
| `multi_night_linking` | link tracklets across nights days apart (tolerance grows with distance travelled) |
| `live_monitor` | `run_full_search` (point + fast + deep) + `run_live_monitor` (archive watch, alerts, state, replay) + CLI |

## Status (2026-10-14)
| Phase | Status | Headline |
|---|---|---|
| scoreboard | ✓ | all numbers from saved files; holdout 2022-06-17 |
| label_quality | ✓ | Horizons labels (0.26 px), fakes indistinguishable from real (AUC 0.47) |
| multiframe_model | ✓ | ties v4: kept the simpler model |
| more_data | ✓ | 26 training units: v5 ties v4 at equal alarms; v4 stays live |
| deep_search | ✓ | 50% at mag **22.1** (single-frame 21.4); null test 0; **6 single-night candidates** r ≈ 21.4–21.9 |
| fast_movers | ✓ | 97% of injected trails, 0 false tracks on 2 nights (cuts chosen there) |
| multi_night_linking | ✓ | 5/5 known asteroids linked over 3 days, 0 wrong |
| live_monitor | ✓ | replay of 2019-01-23: 1 alert, idempotent |
| hunt | ✓ | a two-night search on a nearby field: most cross-night links were known asteroids linked to themselves; an early set of false "candidates" turned out to be known asteroids missed by the cross-match (catalogue positions 4-5 px off), which was fixed. One faint object survives every check so far. **This is not a discovery** and nothing has been reported. (The detailed record is kept in the original project, not in this course copy.) |
