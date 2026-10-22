# Paras: presentation guide

> **Updated slide plan (classical ML only).** Keep slides 1, 2 and 5 below. Replace slides 3 and 4 with these two. The CNN appears
> only as one row of the comparison table.

### NEW Slide 3: "Measure, then let the trees vote" (1 min 30 s)
**On the slide:** `00_Start_Here/00_How_Our_Model_Works.svg`.
**Say:** "For every blob we measure 29 numbers that a person would look at: does the difference picture look like a star, was the spot
already there in the old picture, is it one hot pixel like a cosmic ray, is there a dark hole next to it like a misaligned star. PCA adds
20 more numbers that summarise the pixels without looking at any labels. Then XGBoost, 600 small decision trees, each fixing the errors
of the ones before, turns those 49 numbers into one probability. The most useful number by far is 'looks like a star'."

### NEW Slide 4: "Five classical models, one fair comparison" (1 min)
**On the slide:** the comparison table from `project/docs/classical_realbogus.md` (or the web app's model page) and the junk clusters picture.
**Say:** "We trained logistic regression, an SVM, a random forest and XGBoost on exactly the same nights, chose every setting on validation
and scored the test night once. XGBoost with PCA wins: average precision 0.902, and at 90% recall it lets through 158 junk blobs per photo
against 806 for the old baseline. We had also built a neural network first; it scored 0.898, a tie, so we deploy the simpler model whose
every input we can explain. And k-means on the junk, with no labels, finds the kinds of junk by itself: mostly faint noise, then leftovers of stars."

---


**Your slot: 5 minutes, 5 slides.** (In the 12-minute plan: 3 minutes, slides 2, 3 and 5.) Pictures are in `07_Diagrams/`.

---

### Slide 1: "Where do the labels come from?" (1 min)
**On the slide:** `label_sources.svg`.
**Say:** "The sky does not come labelled, so we make labels. First, we plant fake asteroids into the real images with known brightness and speed, using the real blur shape of each frame, so we have perfect labels. Second, we look up known asteroids in the catalogue, which gives us real labels, using positions from NASA's JPL Horizons because they are ten times more accurate than the other service we tried. Third, everything else is called bogus, which means a little label noise, since it includes the rare real object nobody has catalogued. On one training night that gives 1,059 real detections and 36,161 bogus: one to thirty-four."

### Slide 2: "Three pictures per blob" (1 min)
**On the slide:** `triplet_stamp.svg`.
**Say:** "For every blob we cut a 31 by 31 pixel stamp from three images: science, reference and difference. A real new object shows up in science and difference but not in the reference. A badly subtracted bright star is bright in science and reference and leaves a dipole in the difference. A cosmic ray is a spike narrower than a star. Giving the network all three at once is what lets it tell these apart. Each channel is normalised by its own noise, so a '5' means five sigma on any night. We augment training stamps with rotations and mirror images, because the sky has no 'up', and we balance training to five bogus per real, but keep validation realistic."

### Slide 3: "The neural network" (1 min 30 s)
**On the slide:** `cnn_architecture.svg`.
**Say:** "This is the network: three blocks of convolutions with 3 by 3 filters, each followed by pooling that halves the size, then a small dense head. Convolutions look for local patterns like a round blob or a sharp spike; pooling lets deeper layers see larger areas; dropout switches off 30% of neurons while training so none is relied on. It has only 72,145 weights, which is small on purpose: the stamps are small, the patterns are local, and a small model is fast on a laptop GPU and harder to overfit. We trained it with binary cross-entropy and the Adam optimiser, and we stop when validation average precision has not improved for four epochs."

### Slide 4: "Results" (45 s)
**On the slide:** `cnn_results.svg`.
**Say:** "We always compare against a simple baseline: logistic regression on six numbers. On validation nights the CNN reaches average precision 0.899 against 0.780. At the same completeness of 90%, false alarms fall from 813 per frame to 176, four and a half times fewer, and on the untouched test night the average precision is 0.938. That purity matters because it makes the next step, linking, cheap."

### Slide 5: "The model found a bug in our labels" (45 s)
**On the slide:** `cross_fitting.svg` small, and one sentence in large type: "The most confident 'mistakes' were known asteroids."
**Say:** "When we looked at the network's most confident disagreements with our labels, they were clean new point sources, and every one was a known asteroid. Our labels had missed them because the catalogue's predicted positions were about 2.9 arcseconds off, always in the same direction. We confirmed with the stars and with JPL Horizons that the error was in the catalogue, not in our images, and fixed the labels. That is data-centric ML: the model's most confident mistakes are the cheapest way to find label errors. One more design point: the next model uses this network's scores as features, so we score each training night with a network trained on the other nights, called cross-fitting, to avoid over-confident scores. Over to Satvik."

---

## Optional whiteboard (if asked "explain backpropagation")
Draw: x -> [W1, ReLU] -> h -> [W2, sigmoid] -> p -> loss. Say: "Forward pass computes p. The loss compares p with the label y. Going backwards, the error at the output is simply p minus y for sigmoid with cross-entropy. That error times the transpose of W2, times 1 where the hidden neuron was active and 0 where it was not, gives the error at the hidden layer. Each weight's gradient is its input times the error arriving at it. Then gradient descent moves each weight a small step against its gradient. We wrote this by hand in numpy and checked it against a numerical gradient to eight decimals."

## Delivery tips
- Point at the picture on slide 2: the three panels are the heart of your part.
- Say "average precision" with one sentence of meaning the first time: "a single number that summarises precision and recall for rare positives".
- Do not claim the network discovers asteroids; it removes junk blobs.

## Anticipated first questions
1. "Why a CNN and not just the six numbers?" -> pixels carry shape information numbers do not; AP 0.899 vs 0.780. 2. "Why 31 x 31?" -> blur is about 2 px; 15 px of context shows edges and neighbours; small = fast. 3. "How did you avoid overfitting?" -> dropout, augmentation, early stopping, validation on other nights. 4. "Are your labels correct?" -> best available; noise sources named; two bugs found and fixed. 5. "What is cross-fitting?" -> see slide 5.
