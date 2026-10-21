# Satvik: presentation guide

**Your slot: 5 minutes, 5 slides.** (In the 12-minute plan: 3 minutes, slides 1, 3 and 4.) Pictures are in `07_Diagrams/`.

---

### Slide 1: "From blobs to tracks" (1 min)
**On the slide:** `linking.svg`.
**Say:** "The network gives us thousands of plausible blobs per frame. An asteroid, though, moves in a nearly straight line at a steady speed over the half hour of five exposures. So we join blobs from different frames that line up that way. This is a search, not learning. Trying every combination would be about two times ten to the nineteen tracks, so instead two blobs in two frames define a line, and we look up where the object must be in the other frames with a KD-tree. Because the network has already removed about three quarters of the blobs, a night takes about a fifth of a second. The biggest bug of the whole project lived here: the telescope does not point at exactly the same pixels each exposure, up to twelve pixels of shift, so we first put every blob on one common sky grid. Real asteroids' straight-line error dropped from about two and a half pixels to under half a pixel."

### Slide 2: "Seventeen numbers per track" (1 min)
**On the slide:** the feature table from your knowledge pack, grouped: motion, brightness, strength, CNN opinion, shape, static neighbours, forced photometry.
**Say:** "Each track becomes seventeen numbers, and each has a physical reason: steady speed and a straight line; steady brightness, because noise blips vary; the weakest link; what the neural network thought; a steady shape. Two groups were added after we looked at false alarms by eye: 'static neighbours', to catch artifacts that sit in the same place in every frame, and 'forced photometry', which measures the brightness at the predicted spot even in frames where nothing was detected, because a real object leaves a weak blob there and noise does not. I should also say that I found a bug in my own feature: it counted a slow asteroid's own detections as static neighbours and wrongly flagged 22% of real tracks; I caught it by checking on real examples, fixed it and wrote a test."

### Slide 3: "Trees: from one to boosting" (1 min)
**On the slide:** `boosting.svg`.
**Say:** "Tables of numbers on different scales are where tree models work best. A decision tree asks a few yes or no questions; a random forest averages hundreds of random trees; gradient boosting builds trees one after another, each correcting the errors of the ones before. We used XGBoost with deliberately modest settings: 300 shallow trees, small learning rate, each tree seeing 80% of rows and columns, because we have only about 1,600 tracklets. We cross-validated by night, never by random rows. The honest result: logistic regression, forest and XGBoost all scored about 0.999 and their folds overlap: a tie. We kept XGBoost because it handles missing values and interactions without extra work, but a simpler model would also be defensible."

### Slide 4: "Why we do not report accuracy" (1 min)
**On the slide:** `confusion_metrics.svg`.
**Say:** "Only about 3% of blobs are real. A model that says 'bogus' every time is 96.8% accurate and finds nothing. So we use the confusion matrix: of 2,129 real objects the baseline found 1,547, recall 73%; of the 2,040 things it flagged, 76% were real, precision. We prefer precision-recall to ROC for rare positives: ROC AUC looked excellent at 0.95 while precision at a useful recall was under 20%. And we choose the threshold on validation nights, never on test."

### Slide 5: "Operating point, cross-match, and the human" (1 min)
**On the slide:** `operating_point.svg` plus three bullets: fix recall; compare with the catalogue; a person looks.
**Say:** "A missed asteroid is a lost discovery; a false alarm costs a person a minute. So we fix recall at 90% on validation and then count false alarms: 804 per frame for the simple baseline, 176 for the network. Each surviving track is compared with the catalogue of known asteroids; nearly everything is known. For anything unknown there is a written checklist: does it look like a point source in science and difference but not in the reference, is the motion plausible, is it on a galaxy or a bleed trail, is there any catalogue match, was it seen on another night? On our test night the numbers made three unknown tracks look like asteroids, but the pictures showed a galaxy residual, satellite trails and a bleed trail. We always look at the pictures. Over to Advik, who will show how we measure all of this honestly."

---

## Delivery tips
- Slide 4 is your strongest: say the 96.8% line slowly.
- Use "recall" and "precision" with one plain sentence each the first time.
- For numbers, round and say what they are: "about 800 false alarms per frame", not "804.2".

## Anticipated first questions
1. "Why XGBoost and not a neural network?" -> small table of 17 numbers, about 1,600 examples: trees are strong, need no scaling; networks would overfit. 2. "Why did models tie?" -> the features carry the signal; simplicity. 3. "How do you pick the threshold?" -> recall target on validation. 4. "What is the baseline?" -> logistic regression on six numbers. 5. "Why cross-validate by night?" -> same-night leakage.
