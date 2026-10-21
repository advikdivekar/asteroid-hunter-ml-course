# Aaryan: knowledge pack

## 0. Read first (in this order)
`00_Start_Here/01_Project_Overview` -> `02_Domain_Primer` (all of it; you own the domain) -> `03_ML_Primer` (sections A, C, D) -> then the lessons `project/docs/project_setup.md`, `astronomy_data_primer.md`, `data_acquisition.md`, `difference_imaging.md`, `leakage_audit.md`, and the side check `kepler_post_check.md`.

---

## 1. Problem formulation (3 marks: write this section well)
**The task.** Given several consecutive 30-second exposures of the same patch of sky, find the objects that move across them (asteroids), decide which detections are real, and tell the known ones from the unknown ones.
**Input:** a night of difference images of one CCD quadrant (5 frames, about 7,800 blobs each). **Output:** a ranked list of tracks with a score and a known/unknown flag.
**Why it is hard:** (1) 97% of blobs are junk; (2) asteroids that matter are faint, barely above the noise; (3) there is no ground truth in real images (we must create labels); (4) the telescope pointing shifts between exposures; (5) public catalogues themselves have position errors.
**Why ML:** junk comes in many shapes (dipoles, cosmic rays, edges, noise); a CNN learns them from examples where rules cannot; tree models combine many weak track clues; and the rest is physics (matched filter, linking, stacking), not learning.
**Success is measured by** completeness (how many planted fakes are recovered, by brightness) at a fixed number of false alarms, **depth** (the brightness where 50% are recovered), known asteroids recovered, and unexplained alarms per night. Accuracy is *not* used (3% positives).
**A baseline** is required: logistic regression on six numbers per blob (AP 0.780). Every later model must beat it.
**Scope and honesty:** public data only; one field mostly; a research pipeline, not a discovery claim.
**Origin:** a post claimed a planet candidate in old Kepler data found with AI-assisted code. We checked it (`kepler_post_check.md`): the 13.77-day signal matches the confirmed planet Kepler-169 e (chance of the period match about 1 in 350), simple blending cannot make a dip as deep as claimed (0.0685% maximum vs 0.24% reported), and the decisive test is whether the dips fall at that planet's transit times. We kept the habits: cross-match, quantify the coincidence, name the decisive test.

---

## 2. The data
### 2.1 Where it comes from
ZTF images, served by IRSA. For every exposure: science image (38 MB), mask (19 MB), difference image (7 MB, compressed), PSF (6 KB). The difference image is science minus a deep reference that ZTF has already resampled and blurred to match, so `reference = science - difference` gives us the reference on the same grid for free.

### 2.2 Metadata first
A single web query returned a table of 1,552 exposures of our quadrant (field 618, CCD 3, quadrant 4): 898 r-band, 478 g-band, 176 i-band; 16.4% had quality flags (`infobits != 0`), which we exclude. Choosing nights from this table costs seconds; downloading the wrong 3.5 GB costs an hour. Linking needs several exposures on the same night: only 40 nights had at least 4 good r-band frames.

### 2.3 The split (the most important design decision of the data stage)
| split | nights | median seeing | median depth |
|---|---|---|---|
| train | 8 | 1.86 px | 20.32 |
| validation | 2 | 2.08 px | 19.85 |
| test | 1 (2019-02-08) | 1.60 px | 21.08 |
A holdout night (2022-06-17) was added later for the scoreboard. The test night is the **best** night (sharp, deep, near opposition): the most promising for discovery, so it must not be tuned on. It is also a deliberate **distribution shift**: expect fainter sources than the model has seen much of.
*Why a config file:* `configs/acquire.yaml` fixes the split before modelling, so it cannot drift after disappointing results. `check_split_leakage` must return an empty list for every pair of splits.

### 2.4 The downloader (engineering rules you can explain)
- **Idempotent:** running it twice equals running it once (existing files are skipped).
- **Atomic:** a file is written as `.part` and renamed only when complete; a crash cannot leave a half file that looks finished.
- **Retry with growing waits** (20 s, 40 s, 60 s...) when the archive says 503 (busy); it did, twice, and recovery worked.
- **404 handling:** some exposures lack a product for good (e.g. the subtraction failed); they are recorded in `missing_exposures.csv` and skipped everywhere.
- **Total:** 11 nights x 5 frames + 1 reference = 220 files, 3.55 GB, nothing missing.
- **Provenance:** the metadata table is cached (`metadata__000618.csv`), so re-runs work offline and are repeatable even if the archive changes.

### 2.5 The mask
Each pixel has 16 yes/no flags. We mask: satellite track (0), bad responsivity (2, 3), noisy (4), spike (7), saturated (8), dead (9), NaN (10), halo (12). We do **not** mask the "ghost" bit (5): ghosts cover about 4% of the chip, masking them would hide real asteroids that cross them, and the model can learn what ghosts look like. We read the meaning of each bit from the mask file's own header (BIT00...BIT15) instead of guessing.

---

## 3. Preprocessing: what `load_exposure` does
1. Read the difference image, mask and PSF (and the science image if present).
2. Build the **bad-pixel mask**: ZTF flags + pixels near saturated stars (grown) + NaN pixels. About 6-12% of a difference image is NaN (regions that could not be subtracted); one example frame has 11.9% masked.
3. `reference = science - difference`.
4. Return images, mask, PSF (normalised to sum 1), header, time, zero point, gain.
**Background and noise** (`estimate_background`): the sky level varies slowly (moonlight, clouds). SEP measures the sky in 64 x 64 boxes ignoring stars and interpolates a smooth map; the noise map is the scatter in each place. Robust noise = 1.4826 x MAD, because a bright blob must not inflate it (a plain standard deviation on our example frame was 15.1 vs 8.0 robust).
**Sky coordinates (WCS):** pixel <-> RA/Dec. Pointing shifts between exposures; the "common grid" (Satvik's linking) depends on this.

---

## 4. Detection and exploration (EDA)
### 4.1 `extract_sources`
Mark NaN/bad pixels, subtract background, smooth with a star-shaped kernel (matched filter: random noise averages down, a star-shaped signal does not), keep groups of at least 5 connected pixels at least 2 sigma above the noise, measure flux in a 3-pixel circle, SNR = flux / flux error. Only **positive** blobs are kept: an asteroid is in tonight's image, not in the reference.

### 4.2 The threshold surprise (one training frame)
| threshold | detections | median size |
|---|---|---|
| 1.5 sigma | 6,141 | 11 px |
| **2.0 sigma** | **7,764** | 6 px |
| 3.0 sigma | 731 | 8 px |
| 5.0 sigma | 334 | 9 px |
At 1.5 sigma neighbouring noise peaks touch and **merge** into big blobs, so there are *fewer* detections and a faint asteroid next to a noise blob gets swallowed. A rule of thumb is a start; measuring is better. 2 sigma was chosen and later checked with injections.

### 4.3 Most detections are noise, and that is fine
At 2 sigma about 7,800 detections per frame, only 173 with SNR > 5 (see `07_Diagrams/03_preprocessing_funnel.svg`). Detection is tuned for **completeness** (miss nothing); the CNN (Paras) is tuned for **purity**. Science image: 1,950 sources at 5 sigma; difference image: 334 (about 6x fewer).

### 4.4 The stamp zoo (what the three panels show)
See Paras's `triplet_stamp.svg`: a real object appears in science and difference but not in reference; a bad subtraction shows a star in both and a dipole in the difference; a cosmic ray is a sharp spike narrower than a star; noise is a faint speck; a masked edge is a step.

### 4.5 The combinatorics problem
5 frames x about 8,000 blobs: one blob per frame in every combination is 8,000^5 = about 3 x 10^19 possible tracks. Two fixes later in the pipeline: the classifier removes most junk first, and the linker only tries combinations consistent with straight-line motion.

---

## 5. The leakage audit (your signature story)
Our very first dataset (`legacy/tracklets.csv`, 3,530 rows, 50% real) gave 99.8% accuracy, then 100% with log features. A perfect score is a symptom; we asked what would have to be true of the data.
| Kind | What it is | Ours |
|---|---|---|
| Target leakage | a feature contains the answer | `mag`, `true_rate`, `true_angle` filled for every planted fake and for no bogus row |
| Group leakage | train and test share a source | all tracklets came from the same 4 images |
| Shortcut learning | classes differ in an easy unreal way | random bogus tracklets had residual >= 222.7 px; real ones <= 2.6 px, so one rule separates them |
| Distribution shift | deployment differs from training | test night sharper and deeper |
**The experiment:** trained on easy data and tested on easy data: **100%**; trained on easy data, tested on realistic near-miss tracklets: **50%** (a coin flip); trained on realistic data (real + near misses), tested on realistic: **99%**.
**Hard negatives teach most.** By how far one detection is moved off the line: 3-10 px -> 99%; 1-3 px -> 96%; 0.5-1.5 px -> 89% (a source 1 px off a line is nearly invisible to position-only features, which is why Paras's CNN looks at pixels).
**Rules that came out of it:** features must be measurable on new data; split by night; derived examples go with their source; negatives must be realistic.

---

## 6. Project hygiene (Project setup lesson; two sentences each)
- **Environment:** `environment.yml` rebuilds the same software with one command; the laptop had at least 6 Pythons, so "which Python ran this" was unknowable before.
- **Layout:** `src/asthunt/<stage>/<verb_object>.py`, one purpose per file; settings in `configs/*.yaml`; tests offline; old scripts kept in `legacy/` unchanged and rebuilt properly.

---

## 7. Cross-part cheat sheet
- **Paras:** labels from planted fakes, known asteroids (Horizons) and "the rest"; 3x31x31 stamps normalised by MAD; CNN with 72,145 weights; AP 0.899 vs 0.780; cross-fitting.
- **Satvik:** linking by seed-and-predict with KD-trees on the common sky grid (minimum 8 arcsec/hour, at least 3 of 5 frames); 17 features; XGBoost; metrics (accuracy paradox, PR vs ROC); thresholds chosen on validation.
- **Advik:** one deployed path and bundles; evaluation on planted fakes and held-out nights; depth 20.5 -> 21.4 -> 22.1; deep search by shift-and-stack with a 6-sigma cut; the live viewer.

## 8. Self-quiz (answers in `04_Viva_QA.md`)
1. Why query metadata before downloading? 2. Why must the split live in a config written before modelling? 3. What two properties make the downloader safe to re-run? 4. Why is the ghost bit not masked? 5. The test night is sharper and deeper than training: is that a problem? 6. Why do we keep only positive blobs? 7. Why did 1.5 sigma give fewer detections than 2 sigma? 8. Why is it acceptable that most detections are noise? 9. Name the four leakage types with one example each. 10. What did the 100% -> 50% -> 99% experiment prove?
