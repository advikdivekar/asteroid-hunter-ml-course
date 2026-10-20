# 01. Project overview

## The idea in plain words
Telescopes photograph the same patch of sky again and again. Stars stay put. **Asteroids move**, so over a few minutes an asteroid
is a dot that creeps in a straight line. Surveys like the Zwicky Transient Facility (ZTF) take millions of images, and almost all
the dots they flag are junk (bad image subtraction, cosmic rays, noise). Our project builds a pipeline that:

1. takes ZTF's **difference images** (tonight's picture minus an old picture of the same sky, so everything that did not change disappears),
2. finds every blob left over,
3. uses a **neural network** to decide which blobs look like a real moving object (not junk),
4. **links** blobs from different frames into straight-line tracks,
5. uses a second model (**gradient-boosted trees**) to score each track,
6. compares the tracks with the catalogue of known asteroids, and keeps what is unknown for a person to examine.

It was inspired by an online post where someone found a planet candidate in old Kepler telescope data using AI-assisted code
(we checked that post: the signal belongs to an already confirmed planet, see `project/docs/kepler_post_check.md`).

## Why machine learning is needed here
- One night of one CCD quadrant (5 frames) gives about **36,000-41,000 blobs**, and only about **3% are real** (validation nights: 3.2%). Hand-written rules for "looks like a real object" fail because junk comes in many shapes (dipoles, spikes, edges, ghosts).
- A **CNN** learns those shapes from labelled examples; a rule-based filter could not.
- The **tracklet classifier** combines about 17 numbers per track (speed, straightness, steady brightness, what the CNN thought, ...) better than any single threshold.
- Everything physical (matched filtering, linking by straight lines, shift-and-stack) is *not* ML: it is search and statistics. Knowing which part needs learning and which does not is part of good ML design.

## The pipeline (see `diagrams/system_overview.svg`)
| Stage | What happens | Owner |
|---|---|---|
| Data | ask the archive, download 4 files per exposure, mask bad pixels, estimate noise, detect blobs, split by night | Aaryan |
| Labels + CNN | cut 3x31x31 stamps, labels from planted fakes and known asteroids, normalise/augment/balance, CNN gives p(real) | Paras |
| Linking + classifier | join blobs into tracks, 17 features, XGBoost, cross-match with known asteroids, metrics | Satvik |
| Evaluation + discovery engine | scoreboard, honest testing, shift-and-stack (deep search), fast movers, live viewer | Advik |

## Headline results (all from `project/reports/scoreboard.csv` or the lesson documents)
| What | Number |
|---|---|
| CNN vs a simple feature model (validation nights, 67,041 detections, 3.2% real) | average precision **0.899** vs **0.780** |
| False alarms per frame at 90% recall | **176** (CNN) vs **813** (logistic regression): 4.6x fewer |
| CNN on the untouched test night | average precision **0.938** |
| Tracklet classifier | near-perfect average precision (about 0.999) on planted fakes, which is why we do not use it to judge progress |
| Depth: faintest brightness where 50% of planted fakes are found (test night 2019-02-08) | v1 single frames **20.5**, v4 **21.4**, deep search **22.1** (bigger = fainter) |
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
6. Along the way, real bugs were found by checking results against reality: SkyBoT catalogue positions were ~2.9 arcsec off; telescope pointing shifts between exposures; a feature that wrongly flagged 22% of real tracks. Each was fixed and a test written.

## Where everything is
- Lessons (one per phase, with a quiz): `project/docs/*.md`; matching notebooks: `project/notebooks/*.ipynb`.
- Code: `project/src/asthunt/<stage>/<verb_object>.py` (one purpose per file). Tests: `project/tests/`.
- Settings: `project/configs/*.yaml`. Trained model versions: `project/models/v1 ... v5`.
- The plan and status table: `project/PLAN.md`.
