# Satvik: viva questions and model answers

## A. Linking
**1. What is a tracklet and how is it found?**
A set of detections in different frames on one straight line at a plausible speed. Seed: a pair of detections in two frames at a plausible speed; predict: that fixes a line, so look up the nearest detection where the object must be in the other frames (KD-tree, 2 px); keep: at least 3 of 5 frames, within 1.5 px of the fitted line, fitted speed 8-120 arcsec/hour; de-duplicate best-first.

**2. Why not try every combination?**
5 frames x about 8,000 blobs gives about 2 x 10^19 combinations. A pair of blobs defines a line, so we look the rest up instead of enumerating.

**3. How does the CNN make linking cheaper?**
Keeping 95% of real detections removes about 76% of blobs; seed pairs fall from 52,000 to 3,500 (about 15x) and a night links in about 0.2 s.

**4. Why do identical residual patterns across different asteroids point to a per-frame problem?**
Different orbits would give different residuals. The same pattern (0.6, 2.5, 2.4, 0.6, 1.7 px) for every asteroid means something common to each frame: the pointing shift.

**5. Explain the common sky grid.**
Every blob's RA/Dec is projected onto frame 1's pixel grid (`x_common`, `y_common`), because asteroids move in straight lines on the sky, not in each frame's pixels. Residuals dropped from about 2.5 px to 0.05-0.47 px.

**6. Why was the bug hidden?**
Fakes were planted moving straight in pixel coordinates, so they ignored the pointing shifts and linked perfectly. Our simulation shared the pipeline's blind spot. Lesson: test the simulation against real objects.

**7. Why is the minimum speed 8 arcsec/hour?**
Centroid noise (about 0.5 px over 28 minutes) makes a stationary source appear to move at 3-6 arcsec/hour; 78 such tracklets appeared on the test night. 8 is still far below the 28-48 arcsec/hour of the asteroids we recover. Cost: fakes slower than 8 (18 of 300) cannot be linked, by design.

**8. Centroid noise is 0.5 px over 30 minutes. Roughly what false speed can a stationary source show?**
Fitted speed noise is about 1.4 px/hour per axis (px/hour is about arcsec/hour here), so false speeds of several arcsec/hour appear; hence a minimum well above that.

**9. Why check the speed on the fitted track and not only on seeds?**
A seed pair can pass the limit while the final fitted track does not; the check on seeds alone let slow tracks through. Fixed, with a test.

## B. Features and models
**10. What features describe a tracklet and why?**
17 numbers: motion (count, rate, max residual), brightness (mean, std), strength (mean and min SNR), CNN opinion (mean and min score), shape (size spread, elongation), static neighbours (mean, max), forced photometry (min, median, std of the SNR at the predicted spot, stack SNR). Each reflects physical behaviour: a mover is steady, round, leaves nothing behind and leaves a weak blob where it was missed.

**11. What are static neighbours and forced photometry?**
Static neighbours count detections left at the track's own positions in other frames: a mover leaves none, an artifact walker leaves many. Forced photometry measures SNR at the predicted position in every frame, detected or not; real objects leave a weak blob, noise alignments leave nothing.

**12. What bug did your static-neighbour feature have?**
It counted a slow asteroid's own detections as neighbours and flagged 22% of real tracklets. Found by checking the feature on real examples; fixed to 0.4%; a test reproduces it; v3 (trained with the bug) was retired and v4 trained correctly.

**13. Why trees for this problem?**
A small table of numbers on different scales, about 1,600 examples: trees need no scaling, handle interactions and missing values, and are the standard strong choice.

**14. Explain decision tree, random forest and gradient boosting.**
Tree: yes/no splits; forest: many random trees averaged (bagging, lower variance); boosting: trees built sequentially, each fitting the errors of the previous ones with a small learning rate.

**15. What are your XGBoost settings and why?**
300 trees, depth 3, learning rate 0.05, subsample 0.8, colsample 0.8: modest on purpose for about 1,600 tracklets; shallow trees and subsampling limit overfitting.

**16. Four models scored 0.98-0.999 with overlapping folds. How do you choose?**
They are statistically tied; choose on practical grounds (handles missing values and interactions) or simplicity. Logistic regression would be defensible. We say so.

**17. Why cross-validate by night?**
Rows from the same night share stars, noise and artifacts; random KFold would give inflated scores. GroupKFold keeps each night entirely in one fold.

**18. Why is "bogus tracklets per night" more useful than AP here?**
96% of our tracklets are real because we plant 300 fakes per night, so AP saturates (about 0.999). What a human experiences is how many junk tracks pass the threshold per night (about 2 from 9 validation examples: "a few").

**19. The test night showed 9 false positives. What did you check?**
We compared them with all known asteroids: all 9 were real known objects (V 17.5-21.2) that our labels missed (fainter than the V <= 20.5 label cut, or a frame with a failed match). True false positives: 0; the labels were wrong.

## C. Metrics
**20. Why is accuracy useless here?**
"Always bogus" scores 96.8% with zero recall; the baseline scores 98.4% with 72.7% recall. Accuracy mostly measures the common class.

**21. Define recall, precision and false-positive rate with our numbers.**
TP 1,547, FN 582, FP 493, TN 64,419. Recall = 1,547 / 2,129 = 72.7%; precision = 1,547 / 2,040 = 75.8%; FPR = 493 / 64,912 = 0.76%.

**22. ROC AUC was 0.95 and AP 0.78. Why the difference?**
ROC divides false positives by all 65,000 negatives, so hundreds of false alarms look tiny; precision divides by the number of alerts. With rare positives, precision-recall is the honest picture.

**23. How and why do you choose the threshold at a recall target?**
A missed asteroid is a lost discovery, a false alarm costs a minute, so fix recall (90%) on validation and minimise false alarms. For the baseline: threshold 0.082, precision 19.3%, 804 false positives per frame. Never choose on test.

**24. Worked example: 2,099 real, 64,974 bogus, FPR 2%, recall 95%. What is the precision?**
TP = 0.95 x 2,099 = 1,994; FP = 0.02 x 64,974 = 1,299; precision = 1,994 / (1,994 + 1,299) = 60.6%.

**25. Worked example: a model trained on 50% real outputs 0.8, but real objects are 3%. What is the adjusted probability?**
Odds = 0.8 / 0.2 = 4. Prior-odds ratio = (0.03 / 0.97) / (0.5 / 0.5) = 0.0309. New odds = 4 x 0.0309 = 0.124; probability = 0.124 / 1.124 = 11%.

**26. What is calibration and how do you do it?**
Making scores mean probabilities. Raw scores are inflated because training was balanced (16.7% real vs 3.1% true). Base-rate adjustment (Bayes in odds form) needs no fitting; isotonic regression fits a never-decreasing curve; fit on one half of validation and test on the other. Ranking is unchanged.

**27. What does a completeness error bar mean?**
Standard error sqrt(p(1-p)/n): with p = 0.7 and n = 30 it is +/- 0.084, so differences between bins of 30 objects are mostly noise.

**28. What is the baseline model and why have one?**
Logistic regression on six numbers per detection (AP 0.780). Every fancier model must beat it, or it is not worth the complexity.

## D. Cross-match and vetting
**29. How do you decide a tracklet is a known asteroid?**
At least half of its detections match the same catalogue object within 4 px (any brightness). Requiring several matches makes chance coincidences negligible.

**30. A night gives 3 unknown candidates; validation predicts about 2 bogus per night. What do you conclude?**
Consistent with background: most are probably bogus; vet them, and treat any real one as needing a second night. A non-discovery is the expected outcome.

**31. Walk through the vetting checklist.**
Stamps (science and difference show a point source, reference nothing, every frame); motion steady and plausible; brightness steady; not on a galaxy, bright star, bleed trail or satellite streak; no catalogue match at a looser radius; seen on another night.

**32. Why look at the pictures if the numbers look convincing?**
On the test night track #45 had a steady main-belt-like motion and brightness, but the stamps showed it sat on an elongated galaxy visible in the reference: a subtraction residual. Numbers can mislead; pixels settle it.

**33. Why does the ADES draft use mid-exposure times?**
OBSJD is the shutter-open time; the object was at its recorded position on average 15 s later (JPL Horizons confirmed mid-exposure fits better: 0.13 vs 0.21 arcsec). The draft is marked "NOT FOR SUBMISSION": real reports need vetted multi-night objects.

**34. "Did you write the models yourself?"** Answer truthfully (see `00_Start_Here/07_AI_Use_and_Integrity.md`): the project was built with an AI coding assistant under Advik's direction; say what you re-ran or changed (for example the confusion matrix, the model comparison, the hand calculations) and explain a model without notes.
