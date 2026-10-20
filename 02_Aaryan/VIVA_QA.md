# Aaryan: viva questions and model answers

## A. Problem formulation
**1. What problem does the project solve?**
Finding moving objects (asteroids) in consecutive telescope images, deciding which detections are real, and separating known from unknown ones, with an honest measurement of how deep and how reliable the search is.

**2. What are the input and output of the whole system?**
Input: a night of difference images for one CCD quadrant (5 frames). Output: a ranked list of tracks with a score and a known/unknown flag, for a human to examine.

**3. Why is the problem hard?**
Only about 3% of blobs are real; the interesting asteroids are barely above the noise; real images have no ground truth; the telescope pointing shifts between exposures; and the catalogues we compare with have position errors.

**4. Why use machine learning?**
Junk blobs have many shapes that rules cannot capture, so a CNN learns them from examples; trees combine many weak track-level clues. The rest (matched filter, linking, shift-and-stack) is physics and search.

**5. How do you measure success?**
Completeness (fraction of planted fakes recovered) by brightness, depth (50% completeness magnitude), known asteroids recovered, unexplained alarms per night, and for classifiers recall at a fixed false-alarm level and average precision. Not accuracy.

**6. Which paper does the project relate to?**
The closest is Duev et al. 2019 (real-bogus classification for ZTF with a CNN); also Mahabal et al. 2019 (ML for ZTF), Whidden et al. 2019 (shift-and-stack on GPUs), Masci et al. 2019 (ZTF data processing). Say what we replicated (a CNN on science/reference/difference triplets) and what we changed (31x31 stamps, our own labels).

**7. What was the origin of the project and what did you check?**
A post that claimed a planet candidate in old Kepler data. We checked it: the period matches a confirmed planet (Kepler-169 e, 13.7671 days; chance about 1 in 350), simple blending cannot make a dip as deep as claimed, and the decisive test is whether the dips fall at that planet's transit times.

## B. Data
**8. Where does the data come from and what is in it?**
ZTF images from the IRSA archive. Per exposure: science image (38 MB), mask (19 MB), difference image (7 MB), PSF model (6 KB). A "unit" is field + CCD + quadrant + night.

**9. Why query the metadata before downloading?**
It takes seconds, shows what exists (1,552 exposures, 898 r-band, 16.4% flagged, 40 usable nights), and lets us choose data on purpose; downloading the wrong 3.5 GB costs an hour.

**10. How was the split decided and why by night?**
8 train, 2 validation, 1 test (2019-02-08), 1 holdout (2022-06-17), written in `configs/acquire.yaml` before modelling. Frames of one night share stars, noise and artifacts; random splits would leak that.

**11. Is it a problem that the test night is better than the training nights?**
It is a deliberate distribution shift: the test night is the sharpest and deepest, near opposition, so the most promising for discovery. We expect fainter sources than the model saw much of and read its scores with that in mind.

**12. What makes your downloader safe to re-run after a crash?**
It is idempotent (existing files skipped) and atomic (written as `.part`, renamed when complete). It also retries with growing waits and records exposures that lack a product.

**13. What is a unit and why are results per unit?**
One field + CCD + quadrant + night, e.g. `000618_c03_q4_2019-02-08`. Linking needs frames of the same sky on the same night, and each quadrant has its own pixel grid and metadata.

## C. Preprocessing and detection
**14. What are the four files and what does the mask do?**
Science (tonight), difference (science minus aligned reference), mask (16 flags per pixel: saturated, dead, cosmic ray, ghost...), PSF (blur shape). We mask defects but not the ghost bit, because ghosts cover about 4% of the chip and masking them would hide real asteroids.

**15. How do you get the reference image?**
`reference = science - difference`. ZTF already resampled and blurred the reference to match the science image, so no reprojection code is needed.

**16. How is noise estimated and why robustly?**
SEP estimates the sky level and scatter in 64x64 boxes ignoring stars; for single images we use sigma = 1.4826 x MAD. Bright blobs inflate a plain standard deviation (15.1 vs 8.0 on an example frame).

**17. Why keep only positive blobs?**
An asteroid is present tonight and absent in the reference, so it is a positive blob; negative blobs are things that faded or moved away.

**18. Why did 1.5 sigma give fewer detections than 2 sigma?**
At the lower threshold neighbouring noise peaks touch and merge into large blobs, so the count drops (6,141 vs 7,764), and a faint asteroid next to a noise blob gets swallowed.

**19. 7,800 blobs per frame, 173 with SNR > 5. Is that acceptable?**
Yes: detection is tuned for completeness (do not miss real things), classification for purity (throw junk away). Two steps, each doing what it is good at.

**20. What is the combinatorics problem?**
5 frames x about 8,000 blobs: all one-per-frame combinations are 8,000^5 = about 3 x 10^19. The CNN removes most junk first and the linker only tries combinations consistent with straight-line motion.

**21. What is the WCS and why does it matter?**
The mapping between pixels and sky coordinates. Pointing shifts up to 12 px between exposures, so asteroids move in straight lines on the sky, not in each frame's pixels; linking must use sky-based common coordinates.

## D. Leakage and exploration
**22. Name the four kinds of leakage with an example each.**
Target (`mag` existed only for fakes), group (all tracklets from 4 images), shortcut (random bogus tracklets >= 222 px off a line), distribution shift (test night sharper/deeper).

**23. What did the 100% -> 50% -> 99% experiment show?**
A model trained on easy data scored 100% on easy test data but 50% (a coin flip) on realistic near-miss tracklets; trained on realistic data it scored 99%. A perfect score was a symptom of a too-easy dataset, not of a good model.

**24. What are hard negatives and why do they help?**
Negatives that look like the positives (here, near misses a few pixels off a line). The model learns most from them: accuracy was 99% with offsets of 3-10 px, 96% at 1-3 px and 89% at 0.5-1.5 px, which also shows how hard such cases are for position-only features.

**25. How do you check that no night appears in two splits?**
`check_split_leakage(a, b)` lists groups appearing on both sides; it must return an empty list for train/validation, train/test and validation/test.

**26. What would you do if a result looked too good?**
Ask what would have to be true of the data for it to be so easy; check features for target leakage, splits for shared sources, negatives for realism, and test on a different kind of data.

## E. General ML and hygiene
**27. What is overfitting and how does the split protect against it?**
Overfitting is memorising training noise. Only data the model has never seen shows whether it learned a general rule, so settings are chosen on validation and the test is opened once.

**28. Why is accuracy a poor metric here?**
3% positives: "always bogus" gets 96.8% accuracy with zero recall. Use recall, precision and average precision.

**29. What does `environment.yml` give you?**
One command rebuilds the same software; without it "it worked on my machine" is unprovable (the laptop had at least 6 different Pythons).

**30. What would you improve in the data stage?**
More fields and nights, especially near opposition; a formal learning-curve study (how performance grows with the number of nights); and an automatic check that archive quality flags and our mask agree.

**31. "Did you write this code yourself?"** Answer truthfully (see `00_Start_Here/07_AI_Use_and_Integrity.md`): the project was developed with an AI coding assistant under Advik's direction; say what you re-ran, changed or extended in the data stage, then explain a step (for example the threshold table) without notes.
