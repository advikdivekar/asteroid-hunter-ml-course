# Phase: Classical ML real/bogus model (bundle v6, the deployed model)

## Goal
Replace the CNN in stage 1 with **classical machine learning only**: measure numbers from each blob's three stamps, compare
classical models on the same nights as the CNN, deploy the best one, and prove end to end (the scoreboard) that the whole
pipeline loses nothing. The CNN (`realbogus_cnn.md`) stays in the project as a **comparison**, not as part of the deployed system.

**Files built:**
- `features/compute_stamp_features.py`: 29 numbers per stamp (6 from the blob finder + 23 measured on the stamps)
- `models/train_realbogus_trees.py`: PCA (20 components of the central 15×15 pixels of all 3 channels) + XGBoost
- `models/predict_realbogus_trees.py`: the same inputs at prediction time; `models/predict_realbogus.py` now scores with either model
- `models/save_model_bundle.py`, `load_model_bundle.py`: a bundle can hold the classical model (`realbogus_trees.json`, `realbogus_pca.npz`)
- `scripts/compare_realbogus_models.py`: the model comparison and the k-means clustering of junk
- `scripts/train_classical_bundle.py`: builds bundle **v6** (the v4 design with classical ML in stage 1)
- `scripts/evaluate_bundle.py`: puts a bundle on the scoreboard
- tests: `tests/features/test_compute_stamp_features.py`, `tests/models/test_realbogus_trees.py`, and the end-to-end pipeline test now runs with both models

---

## How I thought about it
1. **A person looking at the three pictures asks a handful of questions.** Is there something in the middle of tonight's picture and of the difference? Was it already in the old picture? Is it shaped like a star, or a single hot pixel, or a plus/minus pair? Each question becomes one number (a *feature*).
2. **Let PCA find what we did not think of.** PCA squeezes the 675 central pixels into the 20 directions in which stamps differ most. It is unsupervised: it never sees the labels.
3. **Compare several classical models on exactly the CNN's data:** the same 8 training nights and balanced set, the same 2 validation nights for every choice, the test night scored once at the end.
4. **Judge the whole system, not just stage 1.** Stage 1 is tuned for recall; what matters is depth, known asteroids found and unexplained alarms, on the scoreboard.

---

## Concepts

### Features (hand-made)
| group | features | the question |
|---|---|---|
| blob finder | `snr`, `peak`, `npix`, `a`, `b`, `flux` | how strong and how big is the blob? |
| brightness | centre and peak in science / reference / difference, a ring around it | is something there, in which pictures? |
| old vs new | `ref_to_sci`, `diff_to_sci` | was it already in the old picture? how much of tonight's light is new? |
| shape | `diff_psf_match`, `diff_sharpness`, `diff_concentration`, `diff_size`, `diff_elongation`, `diff_offset`, `diff_symmetry`, `diff_roughness` | star-shaped? one hot pixel (cosmic ray)? stretched? centred? round? |
| bad subtraction | `diff_min`, `diff_negative_fraction`, `diff_dipole` | a dark hole next to the bright spot (a misaligned star)? |
| old picture | `ref_psf_match` | is there a star at this spot in the old picture? |

`diff_psf_match` is the correlation of the 11×11 difference patch with a round Gaussian of ZTF's blur: 1 = exactly star-shaped.
All stamp numbers are on the normalized scale (`dataset/normalize_stamps.py`), so "5" means 5× the noise on any night.

### PCA (unsupervised)
Principal component analysis finds the directions of largest variance in the 675 pixel values. The first 20 keep **78%** of the
variance. A stamp's position along those 20 directions is 20 more numbers for the trees.

### The models compared
| model | idea |
|---|---|
| logistic regression | a weighted sum of the features through a sigmoid; one straight boundary |
| SVM (RBF kernel) | the widest margin between classes, in a curved space; trained on a random 20,000 (cost grows with n²) |
| random forest | 400 deep trees on random samples, averaged (bagging) |
| XGBoost | 600 shallow trees built one after another, each fixing the errors of the ones before (boosting) |

### k-means clustering (unsupervised)
On the PCA numbers of the 44,290 junk training stamps, k-means with 6 clusters finds the kinds of junk without labels:
93% faint noise specks; about 5% star leftovers (present in science and reference, dark hole in the difference); 1% bright stars
that did not subtract; and a small group that is bright only in the old picture. (`reports/junk_clusters.png`)

---

## Results

### Stage 1: the blob model (same data and rules as the CNN)
| model | AP validation | AP test | precision @90% recall | false alarms per photo @90% recall |
|---|---|---|---|---|
| logistic regression, 6 blob numbers (the old baseline) | 0.783 | 0.508 | 19.2% | 806 |
| logistic regression, 29 features | 0.887 | 0.655 | 41.6% | 269 |
| SVM (RBF), 29 features | 0.886 | 0.877 | 40.9% | 277 |
| random forest, 29 features | 0.895 | 0.930 | 44.9% | 236 |
| **XGBoost, 29 features + 20 PCA (deployed, v6)** | **0.902** | **0.938** | **54.8%** | **158** |
| CNN (deep learning, comparison only) | 0.898 | 0.944 | 51.4% | 181 |

(Validation nights: 67,041 blobs, 3.2% real. Test night: 41,522 blobs, 3.2% real. Precision and false alarms on validation, at the
cut chosen there. `reports/realbogus_comparison.csv`.)

- Good features close almost the whole gap between the old baseline and the CNN; PCA and boosting close the rest.
- Logistic regression and the SVM look fine on validation but drop on the test night, which is sharper and deeper than the
  training nights (a distribution shift). The tree models hold up.
- XGBoost settings were chosen on validation only: depth 4 (0.9034) beat 5 and 6 (0.9025 each); adding PCA raised AP by 0.005 to 0.007;
  training on all 305,463 training blobs instead of the balanced 53,178 did not help (0.896).
- The most important input by far is `diff_psf_match` ("looks like a star"), then `snr`, then the two "was it already in the old picture" numbers.

### The whole pipeline: v6 (classical) vs v4 (CNN), same scoreboard
| nights | depth v4 → v6 (mag) | known asteroids found | unexplained per night | junk kept per photo |
|---|---|---|---|---|
| test (3 quadrants) | 21.34 → 21.28 | 40/44 → 40/44 | 5.0 → 6.3 | 4,293 → 3,568 |
| validation (6) | 20.49 → 20.48 | 24/28 → 24/28 | 0 → 0 | 3,901 → 3,717 |
| holdout (3) | 20.47 → 20.48 | 17/18 → 17/18 | 0.3 → 0.3 | 4,716 → 4,708 |

Re-running v4 the same day reproduced its stored row exactly, so the comparison is like for like. **v6 ties v4:** the same known
asteroids on every split, depth within 0.07 mag, slightly fewer junk blobs kept. So the deployed system uses classical ML only.

---

## Steps, in order
1. `compute_stamp_features`: 29 numbers per stamp; a test checks that they tell a new object, a dipole, a cosmic ray and noise apart.
2. `scripts/compare_realbogus_models.py`: train 5 classical models on the CNN's training set, compare on validation, score test once; cluster the junk.
3. `scripts/train_classical_bundle.py`: train the chosen model, cross-fit scores for the training nights (4 folds by night), choose the stage-1 cut (99% recall on validation), relink and retrain the tracklet model, save bundle v6.
4. `scripts/evaluate_bundle.py v6`: the scoreboard on validation, test and holdout. `configs/live.yaml` now uses v6.

## Commands
```bash
python scripts/compare_realbogus_models.py     # ~2 min: the comparison table and the junk clusters
python scripts/train_classical_bundle.py       # ~2 min: bundle v6
python scripts/evaluate_bundle.py v6           # ~25 min: scoreboard rows for v6
pytest tests/features tests/models tests/pipeline -q
```

## Pitfalls
- A feature that is only measurable on training data (like the true magnitude of a fake) is leakage: every feature here is computed from the stamps and the blob finder alone.
- Compare models at the same recall, not each at its own threshold.
- Do not tune on the test night: every setting above was chosen on validation.
- Python computes the features in 32-bit floats, the web app in 64-bit: a blob whose feature sits exactly on a split can take the other branch of one tree. Scores move by up to 0.01; no keep/drop decision changes (`web/scripts/check_models.mjs`).

## Quiz
1. Why does `ref_to_sci` help separate a new object from a badly subtracted star?
2. PCA never sees the labels. Why can its numbers still help a classifier?
3. Why did logistic regression drop from 0.887 to 0.655 AP on the test night, while XGBoost did not?
4. The classical model and the CNN tie. Name two practical reasons to deploy the classical one.
5. What does k-means tell us that a classifier cannot?
