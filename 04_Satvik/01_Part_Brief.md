# Satvik: linking, the tracklet classifier (trees) and metrics

> **Update: classical ML only (bundle v6).** Stage 1 is now XGBoost on 29 stamp features + PCA instead of the CNN
> (`project/docs/classical_realbogus.md`). Your part is unchanged: you consume its `score` column and threshold (v6: 0.0052, 99% recall
> on validation); the tracklet model was retrained on those scores (v6 tracklet cut 0.533). Where this folder says "CNN score", read
> "stage-1 score".

**Marks this part carries:** Model Development (4, shared with Paras) and the evaluation-metrics foundation of Model Evaluation (3, shared with Advik).
**Why it matters:** this is the classical-ML half of the project (logistic regression, trees, forests, boosting, cross-validation) and the metric thinking that every model in the project is judged by.

## Your four jobs
1. **Linking:** how blobs from different frames become straight-line tracks (a search, not learning), why it is fast, and the biggest bug of the project (pointing shifts).
2. **The tracklet classifier:** the 17 features and the physical idea behind each, the logistic regression baseline, decision tree -> random forest -> gradient boosting, cross-validation by night, the operating point.
3. **Metrics and evaluation basics:** confusion matrix, accuracy paradox, precision/recall/FPR, average precision, PR vs ROC, choosing a threshold on validation, calibration and base-rate adjustment.
4. **Cross-match and vetting:** separating known from unknown asteroids, the expected number of false survivors, and the checklist for a human reviewing a candidate.

## What is NOT yours
Downloads and detection: Aaryan. Labels, stamps, the CNN: Paras (you consume its `score` column and threshold). Scoreboard, deep search, viewer: Advik.

## Where your code is (in `project/`)
| Area | Files | What they do |
|---|---|---|
| Linking | `src/asthunt/link/fit_linear_motion.py`, `link_detections_into_tracklets.py`, `measure_sky_motion.py`, `link_tracklets_across_nights.py` | straight-line fit; the seed-and-predict linker with KD-trees; speed on the sky; (two-night linker, brief) |
| Features | `src/asthunt/features/compute_tracklet_features.py`, `count_static_neighbours.py`, `measure_forced_photometry.py`, `add_path_features.py` | the 11 + 2 + 4 = 17 numbers per tracklet |
| Models | `src/asthunt/models/train_tracklet_classifier.py`, `train_baseline_classifier.py` | XGBoost; logistic-regression baseline |
| Metrics | `src/asthunt/evaluate/compute_classification_metrics.py`, `choose_threshold.py`, `calibrate_probabilities.py`, `adjust_for_base_rate.py` | the numbers and the thresholds |
| Labels and cross-match | `src/asthunt/label/label_tracklets.py`, `src/asthunt/crossmatch/crossmatch_known_asteroids.py`, `src/asthunt/report/write_candidate_report.py`, `write_ades_report.py` | tracklet labels; known vs unknown; the HTML report and draft ADES file |
| Config | `configs/link.yaml` | linking limits |
| Tests | `tests/link`, `tests/features`, `tests/evaluate`, `tests/crossmatch`, `tests/report` | |
| Lessons and notebooks | `docs/ml_basics.md`, `linking.md`, `tracklet_classifier.md`, `metrics_evaluation.md`, `crossmatch_vetting.md`, `multi_night_linking.md`; notebooks with the same names | |

## What you hand over
- The classical-ML section and the metrics definitions for the report.
- Figures: the linking picture, the model comparison table, the confusion matrix and the PR curve (from the metrics notebook).
- The "how a human vets a candidate" checklist.

## Your seven key messages
1. **Linking is a search, not learning:** physics says an asteroid moves in a straight line at constant speed; a KD-tree makes the search fast.
2. **We never try all combinations** (about 2 x 10^19); a pair of blobs defines a line and we look up the rest, after the CNN has removed about 76% of blobs.
3. **The telescope pointing shifts between exposures**, so linking must use a common sky grid; ignoring that was the biggest bug of the project.
4. **Every feature has a physical reason** (steady speed, steady brightness, shape, what the CNN thought, what the sky shows at the predicted spot).
5. **Trees fit tables of numbers best;** four models tied in cross-validation by night, so the choice is practical, and that honesty is a result.
6. **Accuracy is the wrong metric:** "always bogus" scores 96.8%; use recall, precision, average precision, and fix recall first.
7. **A human makes the final call,** with a written checklist; and we always look at the pictures, not only numbers.

## Numbers you must know by heart
| Number | Meaning | Source |
|---|---|---|
| about 2 x 10^19 | combinations if every blob were tried in every frame | `docs/linking.md` |
| 52,000 -> 3,500 | seed pairs before/after the CNN filter (about 15x cheaper); 0.2 s per night | same |
| 8 arcsec/hour | minimum fitted speed (centroid noise fakes slower speeds) | `configs/link.yaml` |
| 3 of 5 frames; 1.5 px; 2 px | minimum detections (v4; v1 used 4 of 5), max residual, match radius | `configs/link.yaml`, bundle |
| 2.5 px -> 0.05-0.47 px | real asteroids' straight-line residual before/after the common sky grid | `docs/linking.md` |
| 11 + 2 + 4 = 17 | features per tracklet | `features/` |
| 300 trees, depth 3, lr 0.05, 80% / 80% | XGBoost settings | `train_tracklet_classifier.py` |
| 0.999 / 0.983 / 0.999 / 0.999 | cross-validated AP: logistic regression, tree, forest, XGBoost | `docs/tracklet_classifier.md` |
| 96% real tracklets | why tracklet AP is saturated (we plant 300 fakes per night) | same |
| 96.8% vs 98.4% | accuracy of "always bogus" vs the baseline | `docs/metrics_evaluation.md` |
| 0.95 vs 0.78 | ROC AUC vs average precision of the baseline | same |
| 1,547 / 582 / 493 / 64,419 | TP / FN / FP / TN of the baseline at threshold 0.5 | same |
| 0.082; 19.3%; 804 | threshold, precision and false positives per frame at 90% recall (baseline) | same |
| 9 of 9 | "false positives" on the test night that were real known asteroids | `docs/tracklet_classifier.md` |

## Hands-on tasks (do these yourself)
1. `cd project && pytest tests/link tests/features tests/evaluate tests/crossmatch -q`.
2. Re-run `notebooks/ml_basics.ipynb`, `metrics_evaluation.ipynb` and `tracklet_classifier.ipynb`. Reproduce the confusion matrix and the four-model comparison.
3. Compute by hand: with 2,099 real and 64,974 bogus, FPR 2% and recall 95%, what is the precision? (answer in `04_Viva_QA.md`).
4. In `configs/link.yaml` set `min_rate_arcsec_per_hour` to 3 in a copy and predict how many more "tracklets" appear and why most would be junk.
5. Train a depth-3 decision tree on the tracklet features and read out its first three questions in words.

## Pitfalls
- Reporting accuracy or ROC AUC alone for a rare-class problem.
- Using random `KFold` on grouped data (inflated scores): always group by night.
- Trusting AP from a handful of negatives (38 bogus tracklets in training, 9 in validation).
- Calling a tracklet "bogus" in a label table without checking: all 9 test-night "false positives" were real known asteroids.
- Quoting balanced-training probabilities as real-world probabilities.
