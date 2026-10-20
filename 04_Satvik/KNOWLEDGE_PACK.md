# Satvik: knowledge pack

## 0. Read first (in this order)
`00_Start_Here/01_Project_Overview` -> `02_Domain_Primer` (sections 8-10) -> `03_ML_Primer` (sections B, C, D) -> then `project/docs/ml_basics.md`, `linking.md`, `tracklet_classifier.md`, `metrics_evaluation.md`, `crossmatch_vetting.md`.

---

## 1. Linking: from blobs to tracks
**Goal:** join detections in the 5 frames into **tracklets**: sets of detections on one straight line at a plausible speed. It is a *search*, not a learning step. Physics supplies the rule: over about 30 minutes an asteroid moves in a straight line at near-constant speed.

### 1.1 Never try every combination
5 frames x about 8,000 blobs: about 2 x 10^19 combinations. Instead (see `diagrams/linking.svg`):
1. **Seed:** for each pair of frames, take every pair of detections whose implied speed is between the minimum and maximum. A KD-tree `query_ball_point` finds all neighbours within `max_rate x dt` instantly.
2. **Predict:** a seed fixes a velocity, so we know where the object must be in every other frame; a KD-tree `query` finds the nearest detection there (within 2 px).
3. **Keep:** at least 3 of 5 frames (v4; v1 required 4 of 5), straight-line residual at most 1.5 px, **fitted** speed within limits.
4. **De-duplicate:** many seeds find the same tracklet; take them best-first (most detections, smallest residual) and drop any that re-use a detection.
**Using the CNN to shrink the search:** keeping 95% of real detections (a threshold chosen on validation) removes about 76% of detections; seed pairs fall from 52,000 to 3,500 (about 15x cheaper); linking a night takes about 0.2 s.
**Why fits and not only seeds:** the speed limit must be checked on the *fitted* track, not just on seed pairs (a bug we had and fixed with a test).

### 1.2 The biggest bug: the telescope moves between exposures
On the test night bright known asteroids failed to link while fakes linked perfectly (100%). Every real asteroid had the **same** residual pattern (0.6, 2.5, 2.4, 0.6, 1.7 px): identical residuals across different objects point to a **per-frame** error, not to orbits.
- **Cause:** pointing shifts between exposures (up to 12 px on the test night, 4-12 px on others), and we linked in each frame's own pixels.
- **Why it hid:** fakes were planted moving straight *in pixel coordinates*, so they ignored the shifts. On most nights the drift was smooth and looked like extra velocity; on the test night it jumped.
- **Fix:** every detection gets `x_common, y_common` (its RA/Dec projected onto frame 1's pixel grid) and linking uses those; fakes are planned on frame 1's grid and converted through the sky into each frame's pixels.
- **Check:** real asteroids' residuals fell from about 2.5 px to **0.05-0.47 px**.
### 1.3 The minimum speed must beat measurement noise
After the fix the test night still had 78 "unknown" tracklets at a median 2.5 arcsec/hour: stationary sources (bad subtractions, variable stars) whose centroids wobble. With about 0.5 px centroid noise over 28 minutes the fitted speed noise is about 1.4 px/hour per axis, so a stationary source can "move" at 3-6 arcsec/hour. New minimum: **8 arcsec/hour**, still far below the speeds of the asteroids we recover (28-48). Cost: fakes planted slower than 8 (18 of 300 per night) cannot be linked, by design.
### 1.4 Results
Linking efficiency (linked among linkable fakes) **100%** on the training night shown; tracklets per night 176-212 on training/validation nights (mostly our 300 fakes) with only **1-8 bogus**; on the test night 273 tracklets (34 labelled bogus, of which 14 were known asteroids our labels missed). Bogus ones are **near misses** (a fake plus a noise detection) or chance alignments: valuable hard negatives.
### 1.5 Two-night linking (brief)
Tracklets from different nights are joined with a tolerance that grows with the distance travelled; 5 of 5 known asteroids were linked over 3 days with 0 wrong. A candidate means little until it is seen on a second night (the Minor Planet Center needs two).

---

## 2. The tracklet classifier
### 2.1 The 17 features and the idea behind each
| group | features | idea |
|---|---|---|
| motion | `n_detections`, `rate_arcsec_per_hour`, `max_residual` | steady, straight motion |
| brightness | `mag_mean`, `mag_std` | an asteroid keeps nearly the same brightness for 30 minutes; noise blips vary |
| strength | `snr_mean`, `snr_min` | one weak link makes a weak track |
| CNN opinion | `score_mean`, `score_min` | what the CNN thought of each detection (cross-fitted for training nights) |
| shape | `npix_std`, `elongation_mean` | a point source keeps the same size and roundness |
| static neighbours (v4) | `static_neighbours_mean`, `_max` | detections left behind at the track's own positions in other frames: a mover leaves none, an artifact walker leaves many |
| forced photometry (v4) | `forced_snr_min`, `forced_snr_median`, `forced_snr_std`, `stack_snr` | the SNR at the predicted position in *every* frame, detected or not; real objects leave a weak blob even in frames where they were missed; noise alignments leave nothing |
**Why the last two groups exist:** vetting 15 false alarms of v2 by eye showed two failure modes (artifact walkers, noise alignments); each got one feature. **A bug in my own feature:** it counted a slow asteroid's own detections as "static neighbours", flagging 22% of real tracklets; fixed to 0.4% with a test that reproduces it.

### 2.2 Decision tree -> random forest -> gradient boosting (`diagrams/boosting.svg`)
- **Decision tree:** repeated yes/no splits that best separate the classes; readable, but one tree over- or under-fits.
- **Random forest (bagging):** hundreds of trees, each on a random sample of rows and features; average their votes; lower variance.
- **Gradient boosting (XGBoost):** trees built one after another, each fitting the *errors* of the trees so far, scaled by a small learning rate. Usually the most accurate on tables.
- **Our settings** (deliberately modest: about 1,600 tracklets): 300 trees, depth 3, learning rate 0.05, each tree sees 80% of rows and 80% of columns.
- **Why trees for this table:** features on different scales, no scaling needed, handles missing values and interactions.

### 2.3 Cross-validated by night (`GroupKFold`; average precision per fold)
| model | folds | mean |
|---|---|---|
| logistic regression | 0.999 / 0.999 / 0.999 / 0.999 | 0.999 |
| decision tree (depth 3) | 0.987 / 0.979 / 0.992 / 0.974 | 0.983 |
| random forest | 0.999 / 0.999 / 1.000 / 0.999 | 0.999 |
| XGBoost | 1.000 / 0.999 / 1.000 / 0.999 | 0.999 |
Everything except the single tree is near-perfect and the folds overlap: **a statistical tie.** XGBoost is used because it handles missing values and interactions without extra work, but logistic regression would be a defensible, simpler choice. Fancier is not automatically better.

### 2.4 Why it looks so easy, and what really matters
In this dataset **96% of tracklets are real** (300 planted fakes per night, junk mostly removed by the CNN and linker): the opposite of the real sky. So AP about 0.999 only says the features separate *these* examples. The useful numbers: **bogus tracklets per night above the threshold** (about 2 per night from 9 validation bogus examples, "a few", not "exactly 2"), depth, known asteroids recovered.
### 2.5 Operating point
Keep 95% of real tracklets on validation. In the first lesson that gave a cut of 0.980 (precision 0.989); the live bundle v4 was retrained on harder tracklets with 17 features and uses **0.601**. Recall by brightness (validation, v1): at least 98% brighter than 19.5, 93% at 19.5-20, **79%** at 20-20.5: the faint end is where it loses.
### 2.6 The test night (looked at once)
AP 0.988, recall 0.887, 9 tracklets labelled "bogus" were kept: **all 9 were real known asteroids** (V 17.5-21.2) that labels missed (fainter than the V <= 20.5 labelling cut, or a frame where the match failed). True false positives on the test night: **0**; the labels were wrong.

---

## 3. Metrics and evaluation basics (`diagrams/confusion_metrics.svg`)
### 3.1 The baseline (ML basics): logistic regression on 6 numbers
`p = sigmoid(w.x + b)`, trained by minimising log-loss; features `snr, peak, npix, a, b, flux`; `StandardScaler` inside a pipeline (fitted on training data only); stratified split with a fixed seed; L2 regularisation (strength `C`). On the toy tracklet data: 99.8% (confusion 440 / 2 / 0 / 441); log10 of rate and residual gave 100%, which meant the data was too easy (see Aaryan's leakage audit).
### 3.2 Confusion matrix on 67,041 validation detections (2,129 real)
`TP = 1,547, FN = 582, FP = 493, TN = 64,419` at threshold 0.5. Recall = 1,547 / 2,129 = 72.7%; precision = 1,547 / 2,040 = 75.8%; FPR = 493 / 64,912 = 0.76%; 493 false positives over 10 frames = about 49 per frame.
### 3.3 Accuracy fails
"Always bogus" gets **96.8%** accuracy and 0% recall; the baseline gets 98.4% with 72.7% recall. Almost the same accuracy, completely different usefulness.
### 3.4 Precision-recall vs ROC
ROC AUC 0.95 looked excellent; average precision was 0.78. ROC's x-axis divides by all bogus (65,000), so hundreds of false alarms look like a tiny step; precision divides by the number of alerts. **For rare positives use precision-recall.**
### 3.5 Choosing the operating point
A missed asteroid is a lost discovery; a false alarm costs a person a minute. So fix a recall target (90%) and minimise false alarms. `choose_threshold(labels, scores, 0.90)` gave 0.082: recall 90.0%, precision 19.3%, **804 false positives per frame** (the CNN brings this to 176). Pushing from 73% to 90% recall multiplied false alarms by 16. **Always choose thresholds on validation, never on test.**
### 3.6 Calibration: do scores mean probabilities?
Training was balanced (16.7% real) but the truth is 3.1%, so raw scores are inflated (a prediction of 0.6 was real only about 30% of the time). Two fixes: **base-rate adjustment** (Bayes in odds form: `new odds = model odds x (true prior odds / training prior odds)`; example from the docstring: 0.9 becomes 0.043 when real is 1 in 1,000 instead of 1 in 6) and **isotonic regression** (a never-decreasing curve from scores to observed fractions), fitted on one half of validation and checked on the other half (fitting and judging on the same rows would flatter it). Calibration changes what you can *say*, not the ranking.
### 3.7 Completeness curves
Completeness by brightness with error bars `sqrt(p(1-p)/n)`: with p = 0.7 and n = 30 the uncertainty is +/- 0.084, so bins of 30 objects mostly show noise.

---

## 4. Cross-match and vetting
- **Cross-match:** for every detection find the nearest catalogue prediction in its frame (4 px, covering catalogue offsets); a tracklet is "known" if at least half its detections match the *same* object, so chance coincidences are negligible.
- **Real search on validation night 2023-09-15:** 31,953 detections -> 13 tracklets -> 10 known asteroids, 0 unknown candidates; of 7 known asteroids with V <= 20 present in at least 4 frames, 5 recovered.
- **Look-elsewhere effect:** thousands of seed combinations per night means some junk survives every cut; estimate the expected number before looking at candidates (about 2 per night): if a night yields about that many unknown candidates, most are probably bogus.
- **Vetting checklist** for every unknown candidate: (1) stamps: point source in science and difference, nothing in reference, every frame? (2) motion steady and plausible (main-belt near opposition: 25-50 arcsec/hour roughly westward)? (3) brightness roughly constant? (4) not on a galaxy or bright star, not along a bleed trail, not a satellite streak? (5) no catalogue match even at a looser radius (15 px)? (6) seen on another night (the MPC needs at least two)?
- **Lesson from the test night:** three interesting unknowns looked like main-belt asteroids in the numbers but were a galaxy residual, satellite trails and a saturated-star bleed in the pictures. **Always look at the stamps.**
- **ADES draft:** the format for reporting to the Minor Planet Center; ours uses mid-exposure times (shutter-open + 15 s), a temporary designation and a header saying "DRAFT ... NOT FOR SUBMISSION".

---

## 5. Cross-part cheat sheet
- **Aaryan:** ZTF difference images; unit = field+CCD+quadrant+night; split by night (8/2/1/1); noise via MAD; blobs by SEP at 2 sigma; the leakage audit.
- **Paras:** labels (planted fakes, Horizons, "the rest"); 3x31x31 stamps; CNN with 72,145 weights; AP 0.899 vs 0.780; cross-fitting by night; the label bug found by the model.
- **Advik:** one deployed path; scoreboard on planted fakes and held-out nights; depth 20.5 -> 21.4 -> 22.1; deep search with a 6-sigma cut; live viewer; a multi-frame CNN tied XGBoost.

## 6. Self-quiz (answers in `VIVA_QA.md`)
1. Why do identical residual patterns across different asteroids point to a per-frame problem? 2. How does the CNN filter make linking about 15x cheaper? 3. Centroid noise 0.5 px over 30 minutes: what false speed can a stationary source show? 4. What does each new tree in boosting try to fix? 5. Four models score 0.98-0.999 with overlapping folds: how do you choose? 6. Why is "bogus tracklets per night" more useful than AP? 7. What did you check before believing 9 false positives on the test night? 8. Why do we fix recall first? 9. 2,099 real, 64,974 bogus, FPR 2%, recall 95%: precision? 10. A model trained on 50% real says 0.8; real is 3%: adjusted probability?
