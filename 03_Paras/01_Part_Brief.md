# Paras: labels, dataset building and the classical ML model

**Marks this part carries:** Model Development (4, shared with Satvik) and the model-ready data part of Data Preprocessing (3, shared with Aaryan).
**Why it matters:** this is the stage-1 machine-learning model, the one that decides for every blob whether it is real. It is also where
"garbage labels in, garbage model out" is most visible, and where the course's classical topics meet: feature engineering, SVM,
trees and boosting, PCA and k-means.

> **Classical ML only.** Our faculty asked for classical machine learning. The deployed model (bundle **v6**) is XGBoost on 29 measured
> features plus 20 PCA numbers. The CNN we built first is kept **only as a comparison**: it tied (AP 0.898 vs 0.902), so the simpler,
> explainable model is the one we use. Lesson: `project/docs/classical_realbogus.md`.

## Your four jobs
1. **Where labels come from:** injection-recovery (planted fakes), known asteroids from catalogues, and "everything else is bogus", with the label-noise caveats and the quality checks.
2. **Building the dataset:** cutting 3 x 31 x 31 stamps, normalising, balancing classes, assembling train/validation sets without leakage.
3. **The classical model:** 29 hand-made features with a physical meaning, 20 PCA numbers, then a fair comparison of logistic regression, an SVM, a random forest and XGBoost (and the CNN as a reference); XGBoost + PCA is deployed.
4. **Unsupervised learning and error analysis:** k-means clusters of the junk, the label bug the first model exposed, and cross-fitting so the next stage gets honest scores.

## What is NOT yours
Downloads and the first detection step: Aaryan. Linking, tracklet features, the tracklet trees and the metric definitions: Satvik. Scoreboard, deep search and viewer: Advik. You hand Satvik a `detections` table with a `score` column and a threshold.

## Where your code is (in `project/`)
| Area | Files | What they do |
|---|---|---|
| Simulate | `src/asthunt/simulate/render_point_source.py`, `plan_injections.py`, `inject_point_source.py`, `plant_fakes_in_exposure.py` | draw a fake from ZTF's own PSF, choose brightness/speed/position, add it to science and difference |
| Label | `src/asthunt/label/match_detections_to_positions.py` (KD-tree), `query_known_asteroids.py`, `refine_known_positions.py`, `label_detections.py` | decide real or bogus for each blob |
| Dataset | `src/asthunt/dataset/build_night_detections.py`, `normalize_stamps.py`, `balance_classes.py`, `assemble_stamp_dataset.py` | one night -> detections + stamps; model-ready arrays |
| Stamps | `src/asthunt/detect/cut_triplet_stamps.py` | 31x31 science / reference / difference cut-outs |
| **Features** | `src/asthunt/features/compute_stamp_features.py` | the 29 numbers per stamp |
| **Model** | `src/asthunt/models/train_realbogus_trees.py`, `predict_realbogus_trees.py`, `predict_realbogus.py` | PCA + XGBoost: train, score |
| **Comparison and clustering** | `scripts/compare_realbogus_models.py` -> `reports/realbogus_comparison.csv`, `junk_clusters.png` | LR, SVM, forest, XGBoost, CNN; k-means on junk |
| **Bundle** | `scripts/train_classical_bundle.py` -> `models/v6/` | cross-fitting by night, the stage-1 cut, the deployed bundle |
| CNN (comparison only) | `src/asthunt/models/define_realbogus_cnn.py`, `train_realbogus_cnn.py`; `models/v4/` | the deep-learning reference |
| Realism check | `src/asthunt/evaluate/measure_injection_realism.py` | adversarial validation of fakes vs real asteroids |
| Tests | `tests/features/test_compute_stamp_features.py`, `tests/models/test_realbogus_trees.py`, `tests/simulate`, `tests/label`, `tests/dataset` | the features tell a new object, a dipole, a cosmic ray and noise apart; the model learns "star in the centre = real" |
| Lessons | `docs/injection_labelling.md`, `label_quality.md`, `dataset_building.md`, **`classical_realbogus.md`**; `realbogus_cnn.md` for the comparison | |

## What you hand over
- The model and its numbers for the report: features, PCA, the model comparison, the deployed settings, error analysis, cross-fitting.
- Figures: the triplet picture, `00_Start_Here/00_How_Our_Model_Works.svg`, the comparison table, the junk clusters.
- A clear statement of label sources and their weaknesses (this is critical for the evaluation chapter).

## Your seven key messages
1. **The sky is not labelled, so we make labels** (planted fakes + known asteroids + "the rest"), and we say honestly where each one is weak.
2. **A real asteroid shows in science and difference but not in the reference**, so our features ask exactly that ("already in the old picture?", "looks like a star in the difference?").
3. **Feature engineering does most of the work:** 29 numbers lifted logistic regression from 0.783 to 0.887 AP; trees and 20 PCA numbers take it to 0.902.
4. **We compared five classical models fairly** (same nights, cut chosen on validation, test scored once): XGBoost + PCA won, 158 false alarms per photo at 90% recall vs 806 for the old baseline.
5. **Deep learning did not beat it:** the CNN scored 0.898; the whole pipeline with either model finds the same known asteroids. Simpler and explainable wins a tie.
6. **k-means shows the kinds of junk without labels:** 93% faint noise, star leftovers with a dark hole, bright stars, old-picture-only.
7. **The first model found a labelling bug:** its most confident "bogus" were known asteroids; the catalogue was 2.9 arcsec off. Fixed with JPL Horizons.

## Numbers you must know by heart
| Number | Meaning | Source |
|---|---|---|
| 31 x 31 x 3 | stamp size and channels | `configs/detect.yaml` |
| 29 + 20 = 49 | hand-made features + PCA numbers per blob | `compute_stamp_features.py`, `train_realbogus_trees.py` |
| 78% | pixel variance kept by 20 PCA components (of 675) | `classical_realbogus.md` |
| 600 trees, depth 4, lr 0.05, 80% / 80% | XGBoost settings, chosen on validation | `train_realbogus_trees.py` |
| 0.783 / 0.887 / 0.886 / 0.895 / **0.902** | validation AP: baseline, LR 29 features, SVM, forest, **XGBoost + PCA** | `reports/realbogus_comparison.csv` |
| 0.938 | test-night AP of the deployed model | same |
| 806 -> 158 | false alarms per photo at 90% recall, baseline -> ours | same |
| 0.898 / 0.944 | the CNN (comparison): validation / test AP | same |
| 1,059 : 36,161 | real : bogus detections on one training night (1 : 34) | `docs/injection_labelling.md` |
| 8,863 real + 5 bogus per real = 53,178; 16.7% real | the balanced training set | `docs/dataset_building.md` |
| 67,041; 3.2% real | the realistic validation set | same |
| 2.72 px -> 0.26 px | SkyBoT vs Horizons median position error | `docs/label_quality.md` |
| 0.46-0.48 | AUC of "fake vs real" classifier (0.5 = indistinguishable) | same |

## Hands-on tasks (do these yourself)
1. `cd project && pytest tests/features tests/models -q`.
2. `python scripts/compare_realbogus_models.py` (about 2 minutes): reproduce the comparison table and the junk clusters.
3. Remove `diff_psf_match` from the feature list in a copy and re-run: how much does XGBoost lose? Why is that feature so useful?
4. Change k-means from 6 to 4 clusters: which kinds of junk merge?
5. Open the web app's "The model" page, pick a junk blob that scored high, and say from its numbers why the trees were fooled.

## Pitfalls
- Saying "the model is 99% accurate": accuracy is meaningless at 3% positives; use AP and false alarms at fixed recall.
- Forgetting that training is balanced (16.7% real) while the real sky is about 3%: raw probabilities are inflated.
- Calling labels "ground truth": they are "best available", with known noise.
- Using a feature that only exists for planted fakes (true magnitude, true speed): that is target leakage. Every feature here is measured on the stamp.
- Choosing settings on the test night. Everything was chosen on validation.
