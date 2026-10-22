# Advik: system architecture, evaluation, deep search, live viewer (LEAD)

> **Update: classical ML only (bundle v6).** Our faculty asked for classical ML. The deployed system is now **v6**: the v4 design with
> XGBoost on 29 features + 20 PCA numbers instead of the CNN in stage 1 (`project/docs/classical_realbogus.md`). You built and scored it:
> `scripts/train_classical_bundle.py`, `scripts/evaluate_bundle.py`. On the scoreboard v6 ties v4 (test: 40/44 known asteroids for both,
> depth 21.28 vs 21.34; validation and holdout equal), and re-running v4 the same day reproduced its stored row exactly. The live viewer
> now uses v6 (`configs/live.yaml`). Wherever this folder says "CNN" for stage 1, read "the stage-1 model"; the CNN is the comparison.

**Marks this part carries:** Model Evaluation and Performance (3), Documentation and Presentation (2), and the integration that makes the other parts one system.
**Why it is the most important part:** every number the team quotes comes from the evaluation you own; it contains the main new method (the deep search); and it holds the pieces together. You also open and close the presentation.

## Your four jobs
1. **The system as a whole**: the architecture, how the stages connect, the deployed path `run_night_pipeline`, model bundles (versions), configs, tests, reproducibility.
2. **The evaluation framework**: planting fakes, the scoreboard, held-out nights, fair model comparison, what the numbers mean, and the negative results (a multi-frame CNN tied XGBoost; 3x more data did not help).
3. **The deep search** (shift-and-stack): the main method that goes fainter than a single frame, with its statistics (look-elsewhere effect, null test) and its measured depth.
4. **The live viewer and the demo**: the interface that runs the real pipeline stage by stage; plus a short account of fast movers and the live monitor.

## What is NOT yours (hand over politely)
- How the data is downloaded and preprocessed: Aaryan.
- Labels, stamps and the CNN: Paras.
- Linking, tracklet features, XGBoost, metric definitions: Satvik.
You must understand each in outline (see the cross-part cheat sheet in your `02_Knowledge_Pack.md`), because you integrate them.

## Where your code is (in `project/`)
| Area | Files | What they do |
|---|---|---|
| Evaluation | `src/asthunt/evaluate/build_scoreboard.py`, `save_scoreboard_rows.py`, `measure_completeness.py`, `find_magnitude_at_completeness.py`, `measure_depth_at_thresholds.py`, `measure_injection_realism.py` | the headline numbers from one function; completeness curves; depth vs false alarms; "are our fakes realistic?" |
| Deployed path | `src/asthunt/pipeline/run_night_pipeline.py`, `run_full_search.py`, `run_deep_search.py`, `run_fast_search.py`, `run_live_monitor.py` | the single code path used for evaluation *and* the real search |
| Model versions | `src/asthunt/models/save_model_bundle.py`, `load_model_bundle.py`; `models/v1 ... v5` | a version = CNN + tracklet model + thresholds + provenance |
| Multi-frame experiment | `models/define_multiframe_model.py`, `train_multiframe_model.py`, `predict_multiframe.py`; `features/add_path_features.py` | the model that tied XGBoost; the two physics features that were kept |
| Deep search | `src/asthunt/deep/*.py` (11 files): `build_snr_image`, `standardize_over_night`, `group_frames_into_stacks`, `make_velocity_grid`, `stack_and_find_peaks`, `mask_bright_pixels`, `merge_peaks`, `measure_velocity_consistency`, ... | matched filter, per-pixel standardisation, GPU stacking over 4,888 motions, peak merging, consistency cut |
| Fast movers | `src/asthunt/fast/*.py` | trailed-streak matched filters and the trail-vs-motion consistency check |
| Live viewer | `src/asthunt/viewer/*.py`, `scripts/live_viewer.py` | web page + server + a traced run of the real functions, one stage at a time |
| Tests | `tests/evaluate`, `tests/pipeline`, `tests/deep`, `tests/fast`, `tests/viewer` | |
| Lessons and notebooks | `docs/scoreboard.md`, `multiframe_model.md`, `more_data.md`, `deep_search.md`, `fast_movers.md`, `live_monitor.md`, `full_run_mlops.md`, `label_quality.md` (evaluation half), `docs/architecture/`; the notebooks with the same names | |

## What you hand to the team
- One list of numbers with their source files (start from `05_Team_Plan.md`) so slides and report agree.
- The demo: a laptop where `python scripts/live_viewer.py` works on a night with fakes.
- The report chapters on evaluation methodology, the deep search and limitations.

## Your seven key messages (say these even if you forget everything else)
1. **We do not trust a number unless it comes from the deployed pipeline on nights nobody tuned on.**
2. **v1 looked perfect (tracklet AP 0.999) but the scoreboard said depth only ~20.5.** The right metric was depth and recovery of known asteroids.
3. **We found where asteroids were lost (linking), fixed that, and a bigger model only tied a simple one.** Use the simplest model that ties.
4. **More data did not help, because one 30 s frame holds only so many photons.** Adding frames (the deep search) did: +0.7 magnitude.
5. **The deep search needed statistics, not just code:** 4,888 trials means noise reaches 5 sigma, so we need 6 sigma, and a null test with shuffled times gave 0 false tracks.
6. **Compare models at the same false-alarm level.**
7. **No discovery is claimed.**

## Numbers you must know by heart (with where they come from)
| Number | Meaning | Source |
|---|---|---|
| 20.5 -> 21.4 -> 22.1 | 50%-completeness magnitude: v1, v4, deep search (test night 2019-02-08) | `reports/scoreboard.csv`, `docs/deep_search.md` |
| 4,888 | trial motions in the deep search | `docs/deep_search.md` |
| 6,179 -> 44 | false peaks per stack at 5 sigma vs 6 sigma on pure noise | `docs/deep_search.md`, `configs/deep.yaml` |
| 47,550 -> 94 | peaks per stack from first version to final | `docs/deep_search.md` |
| 16 -> 0 | null-test false tracklets per night before/after the velocity-consistency cut | `configs/deep.yaml` |
| 20 of 20 | known asteroids kept by that cut | same |
| 0.7 mag | gain of deep search over best single-frame system on the same night | `docs/deep_search.md` |
| 1,499 | chance tracks per night with relaxed linking | `docs/multiframe_model.md` |
| 0.9933 / 0.9924 | multi-frame CNN vs XGBoost average precision (a tie) | `docs/multiframe_model.md` |
| 150+ | offline tests | `pytest` |

## Hands-on tasks (do these yourself; they make the explanation yours)
1. `cd project && pytest -q` (all pass).
2. `python scripts/live_viewer.py`: run a night with 8 fakes, step by step; note which fakes are missed and why.
3. Re-run `notebooks/scoreboard.ipynb` and `deep_search.ipynb`; write one paragraph each in your own words.
4. Change `configs/deep.yaml: peak_threshold_sigma` from 6.0 to 5.0 in a copy and predict (then check) what happens to the false-peak count.
5. Open `reports/scoreboard.csv`; for each of v1, v2, v4, v5 find the test-split depth and unexplained alarms per night and explain the pattern in two sentences.

## Pitfalls
- Quoting a depth number without saying *which night and quadrant*.
- Saying "the model is 99.9% accurate": it is a saturated metric on a planted-fake-heavy set.
- Claiming the deep search "finds new asteroids": it reaches fainter *known* and unknown objects; no discovery is claimed.
- Forgetting that the test night was looked at while fixing real bugs (the holdout night exists because of that).
