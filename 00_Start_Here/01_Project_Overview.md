# 01. Project overview

## The idea in plain words
Telescopes photograph the same patch of sky again and again. Stars stay put. **Asteroids move**, so over a few minutes an asteroid
is a dot that creeps in a straight line. Surveys like the Zwicky Transient Facility (ZTF) take millions of images, and almost all
the dots they flag are junk (bad image subtraction, cosmic rays, noise). Our project builds a pipeline that:

1. takes ZTF's **difference images** (tonight's picture minus an old picture of the same sky, so everything that did not change disappears),
2. finds every blob left over,
3. uses **classical machine learning** (29 measured numbers + PCA per blob, then XGBoost decision trees) to decide which blobs look like a real moving object (not junk),
4. **links** blobs from different frames into straight-line tracks,
5. uses a second model (**gradient-boosted trees**) to score each track,
6. compares the tracks with the catalogue of known asteroids, and keeps what is unknown for a person to examine.

It was inspired by an online post where someone found a planet candidate in old Kepler telescope data using AI-assisted code
(we checked that post: the signal belongs to an already confirmed planet, see `project/docs/kepler_post_check.md`).

## Why machine learning is needed here
- One night of one CCD quadrant (5 frames) gives about **36,000-41,000 blobs**, and only about **3% are real** (validation nights: 3.2%). Hand-written rules for "looks like a real object" fail because junk comes in many shapes (dipoles, spikes, edges, ghosts).
- A **classical ML model** learns those shapes from labelled examples: we measure 29 numbers per blob (is it star-shaped? was it already in the old picture? one hot pixel?) plus 20 PCA numbers, and XGBoost learns which combinations mean real. A rule-based filter could not. (We first used a deep-learning CNN; it scored the same, so the deployed system uses classical ML only and the CNN is kept as a comparison: `project/docs/classical_realbogus.md`.)
- The **tracklet classifier** combines about 17 numbers per track (speed, straightness, steady brightness, what the first model thought, ...) better than any single threshold.
- Everything physical (matched filtering, linking by straight lines, shift-and-stack) is *not* ML: it is search and statistics. Knowing which part needs learning and which does not is part of good ML design.

## The pipeline (see `11_Diagrams/01_system_overview.svg`)
| Stage | What happens | Owner |
|---|---|---|
| Data | ask the archive, download 4 files per exposure, mask bad pixels, estimate noise, detect blobs, split by night | Aaryan |
| Labels + ML model | cut 3x31x31 stamps, labels from planted fakes and known asteroids, normalise/balance, 29 features + PCA, XGBoost gives p(real); SVM/forest/CNN compared; k-means on junk | Paras |
| Linking + classifier | join blobs into tracks, 17 features, XGBoost, cross-match with known asteroids, metrics | Satvik |
| Evaluation + discovery engine | scoreboard, honest testing, shift-and-stack (deep search), fast movers, live viewer | Advik |

## Headline results (all from `project/reports/scoreboard.csv` or the lesson documents)
| What | Number |
|---|---|
| Our blob model (XGBoost, 29 features + 20 PCA) vs the old 6-number baseline (validation nights, 67,041 detections, 3.2% real) | average precision **0.902** vs **0.783** |
| False alarms per frame at 90% recall | **158** (our model) vs **806** (baseline): 5.1x fewer |
| Our model on the untouched test night | average precision **0.938** |
| Deep learning (CNN), for comparison only | 0.898 validation, 0.944 test: a tie, so we deploy classical ML |
| Whole pipeline, v6 (classical) vs v4 (CNN) on the scoreboard | same known asteroids on every split (40/44 on test); depth 21.28 vs 21.34 |
| Tracklet classifier | near-perfect average precision (about 0.999) on planted fakes, which is why we do not use it to judge progress |
| Depth: faintest brightness where 50% of planted fakes are found (test night 2019-02-08) | v1 single frames **20.5**, v4/v6 **21.3-21.4**, deep search **22.1** (bigger = fainter) |
| Real search of the test night | **31 known asteroids recovered** (brightness 16.2 to 21.8), 0 unexplained candidates above threshold |
| Fast (trailed) objects | 97% of planted trails recovered on the nights used to choose the cuts; 93% on a later 10-frame night |
| 3x more training data (v5) | did **not** help at equal false alarms |

## What we are careful NOT to claim
- **No discovery.** One faint object in a related two-night study passes every check we have (orbit fit, noise tests, no catalogue match). It still needs a human reading the images and the Minor Planet Center's process. Nothing has been reported as a discovery.
- The best depth number (22.1) is for one night and one CCD quadrant.
- Our "real" labels come partly from simulation (planted fakes). We checked that fakes look like real asteroids (a classifier cannot tell them apart: AUC 0.47) but that is a check on certain features, not a proof.

## The story of how it got better (good for the viva)
1. **v1**: CNN + XGBoost on single frames. Looked great (AP 0.999 for tracklets). The scoreboard showed the truth: depth only ~20.5.
2. We **measured where asteroids were lost**: at linking, not at the classifier. Relaxing linking found 11% more asteroids but created ~1,500 chance tracks per night (these became the hard negatives).
3. A bigger multi-frame CNN **tied** the simple XGBoost model, so we kept the simple one (**v4**).
4. 3x more training data (**v5**) did **not** help: the limit is how many photons one 30-second frame holds.
5. Therefore we added the **deep search** (adding frames along a trial motion): +0.7 magnitude.
6. **Classical ML only (v6)**: we replaced the CNN with XGBoost on 29 measured features + PCA. It tied the CNN on blobs and on the whole scoreboard, trains in seconds, and every input has a meaning, so it is the deployed system; the CNN stays only as a comparison.
7. Along the way, real bugs were found by checking results against reality: SkyBoT catalogue positions were ~2.9 arcsec off; telescope pointing shifts between exposures; a feature that wrongly flagged 22% of real tracks. Each was fixed and a test written.

## Where everything is
- Lessons (one per phase, with a quiz): `project/docs/*.md`; matching notebooks: `project/notebooks/*.ipynb`.
- Code: `project/src/asthunt/<stage>/<verb_object>.py` (one purpose per file). Tests: `project/tests/`.
- Settings: `project/configs/*.yaml`. Trained model versions: `project/models/v1 ... v5`.
- The plan and status table: `project/PLAN.md`.
