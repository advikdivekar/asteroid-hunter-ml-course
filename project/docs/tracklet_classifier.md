# Phase: Tracklet classifier

## Goal
Score every linked tracklet: real moving object, or near miss / chance alignment? We learn the tree family (decision tree → random forest → gradient boosting), cross-validate by night, and choose an operating threshold.

**Files built:**
- `features/compute_tracklet_features.py`: 11 numbers per tracklet (motion, brightness, strength, CNN opinion, shape)
- `models/train_tracklet_classifier.py`: XGBoost with modest, explained settings
- tests for both, `notebooks/tracklet_classifier.ipynb`
- saved: `models/tracklet_classifier__<date>__<sha>.json`, and `tracklet` in `data/processed/thresholds.json`

---

## How I thought about it
1. **Tabular data → trees.** Tracklet features are a small table of numbers on different scales. Gradient-boosted trees are the strongest standard tool for that, and need no scaling.
2. **Start readable.** A depth-3 decision tree shows *which* questions separate real from bogus. Then check whether ensembles add anything.
3. **Cross-validate by night** (`GroupKFold`), never by row, for the same reason as every split in this project.
4. **Count the negatives before trusting the score.** There are only 38 bogus tracklets in training and 9 in validation. Any estimate built on 9 examples is rough.
5. **Inspect the test night's "false positives".** All 9 turned out to be real known asteroids.

---

## Concepts

### Features (what a person would check)
| feature | idea |
|---|---|
| `n_detections`, `max_residual`, `rate_arcsec_per_hour` | steady straight motion |
| `mag_mean`, `mag_std` | brightness stays constant over 40 min |
| `snr_mean`, `snr_min` | one weak link makes a weak tracklet |
| `score_mean`, `score_min` | what the CNN thought of each detection (cross-fitted for training nights!) |
| `npix_std`, `elongation_mean` | same size and roundness in every frame |

### Decision tree → random forest → gradient boosting
- **Decision tree:** repeated yes/no splits that best separate the classes. Readable, but one tree overfits or underfits easily.
- **Random forest:** hundreds of trees, each trained on a random sample of rows and features; average their votes. Lower variance.
- **Gradient boosting (XGBoost):** trees built one after another, each fitting the *errors* of the trees so far, with a small learning rate. Usually the most accurate on tables.

Our settings: 300 trees, depth 3, learning rate 0.05, 80% row and column subsampling. These are deliberately modest for ~1,600 examples.

### Cross-validation by night (average precision per fold)
| model | AP per fold | mean |
|---|---|---|
| logistic regression | 0.999 / 0.999 / 0.999 / 0.999 | 0.999 |
| decision tree (depth 3) | 0.987 / 0.979 / 0.992 / 0.974 | 0.983 |
| random forest | 0.999 / 0.999 / 1.000 / 0.999 | 0.999 |
| XGBoost | 1.000 / 0.999 / 1.000 / 0.999 | 0.999 |

Everything except the single tree is near-perfect, and the folds overlap, so **the models are statistically tied**. We use XGBoost because it handles missing values and feature interactions without extra work, but logistic regression would be a defensible, simpler choice. That's an honest lesson: fancier isn't automatically better.

### Why it looks so easy, and what that means
In this dataset, **96% of tracklets are real**, because we plant 300 fakes per night while the CNN filter plus the linker's physics already removes most junk. That's the *opposite* of the real sky, where almost every tracklet with no known match is bogus. So:
- AP ≈ 0.999 says the features separate *these* examples well.
- The number that matters for a search is **bogus tracklets per night above the threshold**. Validation estimate: **2 per night**, from just 9 bogus examples, so treat it as "a few", not "exactly 2".

### Operating point
Keep 95% of real tracklets on validation → threshold **0.980** → precision 0.989, ~2 bogus per night.

Recall by brightness (validation): ≥ 98% brighter than mag 19.5, 93% at 19.5–20, **79% at 20–20.5**. The faint end is where it loses.

### The test night (looked at once)
- AP 0.988, recall 0.887, and 9 tracklets labelled "bogus" were kept.
- **All 9 are real known asteroids** (V 17.5–21.2) that our labels missed: fainter than the V ≤ 20.5 labelling cut, or with a frame where the match failed.
- So the classifier's *true* false-positive count on the test night was **0**, and the labels were wrong.

---

## Steps, in order
1. Wrote `compute_tracklet_features` and `train_tracklet_classifier` + tests.
2. Notebook: feature histograms; depth-3 tree drawn; 4 model families cross-validated by night; final XGBoost; feature importance; threshold at 95% recall on validation; recall by magnitude; model + threshold saved.
3. After the linking fixes (common grid, 8 "/hr minimum), rebuilt tracklets and retrained: fewer, cleaner bogus tracklets (38 in train instead of 115).
4. Checked the test night's kept "bogus" tracklets against all known asteroids → all 9 were real.

## Commands
```bash
pytest tests/features tests/models
jupyter lab notebooks/tracklet_classifier.ipynb
```

## Pitfalls
- **Random `KFold` on grouped data:** inflated scores (exercise 3).
- **Trusting AP from a handful of negatives.**
- **Class balance that's the reverse of reality:** read precision per night, not AP.
- **In-sample CNN scores as features:** we used cross-fitted scores.

## Quiz
1. What does each new tree in gradient boosting try to fix?
2. Four models score 0.98–0.999 with overlapping folds. How do you choose, and why might logistic regression be fine?
3. Why is "bogus tracklets per night" more useful than AP here?
4. The test night shows 9 false positives. What did we check before believing that number?
5. Which feature would you add to catch a satellite streak, and why?
