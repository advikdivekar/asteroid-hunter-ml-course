# Paras: knowledge pack

## 0. Read first (in this order)
`00_Start_Here/01_Project_Overview` -> `02_Domain_Primer` (sections 3-6, 11) -> `03_ML_Primer` (sections A, B, C) -> then `project/docs/ml_basics.md` (logistic regression, features, labels), `injection_labelling.md`, `label_quality.md`, `dataset_building.md`, `neural_networks.md`, `realbogus_cnn.md`.

---

## 1. Labels: where they come from and how good they are
There is no list of "all asteroids in this image". There is a list of things we put there ourselves, and a catalogue of things others found.

### 1.1 Injection-recovery (planted fakes) -> perfect labels
- `plan_injections`: for each of 300 fakes per night: magnitude uniform 17-22, speed uniform 5-100 arcsec/hour, random direction, random start; positions follow the **real frame times** on **frame 1's sky grid**, then are converted through the sky to each frame's own pixels (after we found pointing shifts ignored the old way).
- `render_point_source`: the fake is drawn from ZTF's **own PSF** of that frame (not a textbook Moffat), shifted by a sub-pixel amount with cubic interpolation and rescaled so the total flux is exact (otherwise every fake would sit on a pixel centre and a model could learn "perfectly centred = fake").
- `inject_point_source`: Poisson noise (`Poisson(stamp x gain) / gain`), added to **both science and difference**, so `reference = science - difference` is unchanged: exactly the situation of a new object.
- **Weakness:** fakes are only as realistic as the simulation. Any difference is a distribution shift the model may exploit. So: real PSF, real noise, real times, correct sky motion, and a realism test (1.5).
- Magnitudes are drawn **uniformly**, not like real asteroids (which are mostly faint), so every brightness bin has enough fakes to measure completeness. Remember this when quoting rates.

### 1.2 Known asteroids -> real labels
Positions of catalogued asteroids at each frame time. We match detections to positions within 2 px with a **KD-tree** (the nearest neighbour in about log n steps instead of comparing 7,700 x 300 pairs per frame). We use only known asteroids brighter than V = 20.5: fainter ones are mostly **chance matches**. The check: P(random point lands within 2 px of some detection) = detections x pi x 2^2 / good-pixel area = 1.15%; 114 known asteroids x 1.15% = 1.3 expected chance matches per frame, which is what we saw, so the early faint "matches" were mostly wrong.
**Observer location matters:** SkyBoT defaults to the centre of the Earth; ZTF is about 6,400 km away, shifting an asteroid 1 AU away by up to about 9 arcsec (4 pixels). We pass `location="I41"` (ZTF's observatory code).

### 1.3 Everything else = bogus
Includes variable stars, artifacts, and the rare real object nobody has catalogued: **label noise** we accept. Consequence: a "false positive" on real data may be a real discovery, which is exactly what the project hunts.
**Class imbalance:** one training night: 1,059 real vs 36,161 bogus (1 : 34).

### 1.4 The label-quality story (two bugs, both found by looking at where the model disagreed with the labels)
1. **SkyBoT offset.** The CNN's most confident "bogus" detections were clean new point sources and **every one was a known asteroid** (V 17-20). SkyBoT's predictions sat about 2.9 arcsec from our detections *in a constant direction* (-2.4, +1.5 px) on every night, with scatter of only 0.1-0.5 px. Our WCS matches Gaia stars to 0.04 px and JPL Horizons matches our detections to 0.1-0.3 arcsec, so the error is in SkyBoT. First fix: a 4-px match radius (AP 0.877 -> 0.894). Better fix (label-quality phase): use **JPL Horizons** positions where available (median error 0.26 px vs 2.72 px, 314 matches over 12 nights) and a 2-px radius (4x less area, 4x fewer chance matches); SkyBoT only as fallback, tagged by source so a detected faint known asteroid is never mistaken for a discovery.
2. **Fakes moved straight in pixel coordinates**, ignoring the telescope's pointing shifts, so real asteroids failed to link while fakes did. Fix: plan fakes on frame 1's sky grid and convert.
**Lessons:** check the *direction* of offsets, not only the size (a systematic error hid behind chance coincidences); a model's most confident "mistakes" are the cheapest way to find label errors; test your simulation against real objects, not only against itself.

### 1.5 Do the fakes look real? (adversarial validation)
Train a classifier to separate fakes from real known asteroids. AUC 0.5 = cannot. Rules: match the confounders (sample fakes with the same magnitude distribution as the real asteroids) and cross-validate by night. Result: AUC 0.46-0.48 (two seeds: 0.481 and 0.463; report 0.47 +/- 0.02, not three decimals of false precision). The same trick detects train/test distribution shift in general.

---

## 2. Building the dataset
`build_night_detections` per unit: load frames -> plant fakes -> detect -> label -> describe (RA/Dec, common-grid x/y, magnitude, seeing) -> cut stamps -> save `detections__<unit>.parquet`, `stamps__<unit>.npy` (float16), `injections__<unit>.parquet`. Built once, re-used many times (about 40 s per night).
- **Save everything, choose later:** every detection's stamp is kept (about 230 MB per night as float16); balancing happens when assembling, so the same files serve training (balanced), validation (realistic) and the real search (all).
- **Seeds:** each night's generator is seeded with `[project seed, night]`: repeatable (an early draft used Python's `hash`, which changes every run).
- **A float16 bug:** saturated star cores exceed 65,504 (float16's maximum) and became `inf`; fix: clip before converting.
- **Sizes:** 11 nights; per night 32-41k detections, 1,053-1,199 real, 27-37 bogus per real.

### 2.1 Stamps (the CNN's input)
Three channels of 31 x 31 pixels around the blob: science, reference, difference. Why three? A real new object: blob in science and difference, nothing in reference. Bad subtraction: bright star in both science and reference, dipole in difference. Cosmic ray: sharp spike in science and difference, narrower than a star. Noise: faint speck. Masked edge: step in all. A person compares panels; the CNN learns to. 31 px because the blur is about 2 px wide and 15 px of surroundings is enough to see a neighbouring star or an edge (ZTF's own classifier uses 63 x 63).

### 2.2 Normalisation, augmentation, balancing
- **Normalise** per stamp and channel: `(value - median) / (1.4826 x MAD)`, clipped to [-10, 50]. Median (not mean) so a star barely moves it; MAD x 1.4826 equals the standard deviation for Gaussian noise but ignores outliers; "5" now means 5 sigma on *any* night; clipping stops a saturated star dominating. Nothing is fitted on training data, so there is no leakage here (contrast `StandardScaler`).
- **Augment** (training only): rotations by 0/90/180/270 degrees and optional mirror = 8 versions of every stamp; the sky has no "up". Acts like more data, teaches orientation does not matter. *Never augment validation or test.*
- **Balance** training: all real rows + 5 bogus per real (8,863 real -> 53,178 stamps, 16.7% real) because with 3% positives the gradient is dominated by easy bogus examples and "always bogus" is a comfortable optimum. Cost: inflated probabilities (the model believes real is 16.7% common). Validation stays realistic (67,041 detections, 3.2% real) so precision and false-alarm numbers are honest.

---

## 3. Neural-network theory (the from-scratch notebook)
- **Neuron:** `output = activation(w.x + b)`; with a sigmoid it is logistic regression.
- **Why stack:** with ReLU (`max(0,z)`) each hidden neuron contributes a "bent line"; 16 bent lines make a curved boundary (the half-moons: logistic regression 0.88, network 0.94). Without a non-linear activation, stacked layers collapse to one linear function.
- **Forward pass:** `z1 = X W1 + b1; h = ReLU(z1); z2 = h W2 + b2; p = sigmoid(z2)`.
- **Loss:** binary cross-entropy `L = -mean(y log p + (1-y) log(1-p))`. Confident and wrong costs a lot (-log 0.01 = 4.6); confident and right costs almost nothing (-log 0.99 = 0.01).
- **Backpropagation** = chain rule from the output back. Output: for sigmoid + cross-entropy, `dL/dz2 = p - y` (the sigmoid's slope cancels the log's slope). Output weights: `dL/dW2 = h^T (p - y) / n`. Back into the hidden layer: `error_hidden = (p - y) W2^T x ReLU'(z1)` where ReLU' is 1 if z1 > 0 else 0. Hidden weights: `dL/dW1 = X^T error_hidden / n`. Each layer reuses the one after it: one backward sweep gives every gradient.
- **Gradient check:** nudge one weight by epsilon = 1e-5 on both sides: `(L(w+e) - L(w-e)) / 2e` must equal the backprop gradient (ours: -0.02896520 both).
- **Gradient descent:** `w <- w - lr x gradient`. Too small: slow; too large (try 5.0): overshoots and diverges. **Adam** adapts each weight's step using its recent gradients.
- **Initialisation:** small and random; if all weights started equal every neuron would stay identical.
- **PyTorch's four steps:** `optimizer.zero_grad()`, `loss = loss_fn(model(X), y)`, `loss.backward()`, `optimizer.step()`. `BCEWithLogitsLoss` fuses sigmoid and cross-entropy (stable), so the model outputs raw **logits**; `torch.no_grad()` and `model.eval()` when scoring.
- **Overfitting demo:** a 256 x 256 network on 40 points: training loss falls toward 1e-5 (memorising noise) while validation loss is lowest at step 22 then climbs: hence **early stopping**.

---

## 4. The CNN in detail
### 4.1 Architecture (`define_realbogus_cnn.py`; see `07_Diagrams/03_cnn_architecture.svg`)
Input (3, 31, 31) -> [Conv3x3 16 + ReLU] -> [Conv3x3 16 + ReLU] -> MaxPool (31->15) -> [Conv3x3 32 + ReLU] -> [Conv3x3 32 + ReLU] -> MaxPool (15->7) -> [Conv3x3 64 + ReLU] -> MaxPool (7->3) -> Flatten (64 x 3 x 3 = 576) -> Dropout 0.3 -> Dense 64 + ReLU -> Dense 1 -> (sigmoid at scoring time).
- **Convolution:** a 3x3 filter slides over the stamp; at each position it multiplies 9 pixels by 9 learned weights and sums. With 3 input channels each filter has 3x3x3 weights and sees science, reference and difference together, which is how it learns "bright in science and difference, nothing in reference". Padding 1 keeps the size.
- **Pooling:** max-pool halves the size, so deeper filters see larger areas (edges and blobs first, then "point source", "dipole", "streak").
- **Dropout 0.3:** randomly switches off 30% of neurons while training so no neuron is relied on.
### 4.2 Counting the 72,145 weights (do this yourself)
`params = (in_channels x 3 x 3 + 1) x out_channels` for a conv layer: conv1 (3->16) = 448; conv2 (16->16) = 2,320; conv3 (16->32) = 4,640; conv4 (32->32) = 9,248; conv5 (32->64) = 18,496; dense (576 -> 64) = 576 x 64 + 64 = 36,928; output (64 -> 1) = 65. Sum = 72,145.
### 4.3 Training recipe (`train_realbogus_cnn.py`, `configs/train.yaml`)
| setting | value | why |
|---|---|---|
| loss | `BCEWithLogitsLoss` | stable cross-entropy |
| optimiser | Adam, lr 0.001 | robust default |
| batch | 256 | stable gradients, fits easily |
| augmentation | random 90-degree rotations + mirror | sky has no up |
| early stopping | patience 4 on validation AP | keep the best epoch |
Judged by **validation AP on realistic data**, not training loss (training loss 0.08 vs validation 0.035 can both be right: training is balanced and augmented, validation is easier). The final run used all 25 epochs; best validation AP (0.899) at epoch 21.
### 4.4 Results (validation, 67,041 detections, 3.2% real)
| model | AP | precision at 90% recall | false positives per frame |
|---|---|---|---|
| logistic regression, 6 features | 0.780 | 0.191 | 813 |
| **CNN** | **0.899** | **0.522** | **176** |
At the same completeness the CNN cuts false alarms **4.6x**; on the untouched test night AP **0.938**. The completeness-vs-magnitude curves look alike at 90% recall because we fixed recall: the CNN's gain is **purity**, which makes linking cheap: after the CNN filter only about 24% of detections remain, and seed pairs drop from 52k to 3.5k.
### 4.5 Cross-fitting (stacking without leakage; `07_Diagrams/05_cross_fitting.svg`)
8 training nights -> 4 folds of 2 nights. For each fold train a CNN on the other 6 nights and score the 2 held-out nights; validation and test nights are scored by the final CNN. Without it, the tracklet model would see over-confident CNN scores on training nights and learn "score_min is always high for real things", which would not hold on new nights.

---

## 5. Cross-part cheat sheet
- **Aaryan:** ZTF difference images; split by night; masks; noise via MAD; blobs by SEP at 2 sigma (7,800 per frame, 173 with SNR > 5); leakage audit (100% -> 50% -> 99%).
- **Satvik:** linking by seed-and-predict with KD-trees on the common sky grid; 17 features (11 + 2 static-neighbour + 4 forced-photometry); XGBoost (300 trees, depth 3, lr 0.05); metrics and thresholds; models tie in cross-validation.
- **Advik:** one deployed path; scoreboard on planted fakes and held-out nights; depth 20.5 -> 21.4 -> 22.1; deep search; live viewer; more data did not help.

## 6. Self-quiz (answers in `04_Viva_QA.md`)
1. Why does a 3-channel filter help? 2. Training loss is 0.08 and validation 0.035: why can validation be lower? 3. Why do the CNN and baseline completeness curves look alike at 90% recall? 4. Explain cross-fitting in two sentences.
5. How did the CNN reveal the SkyBoT offset? 6. Count the parameters of the first conv layer.
7. Why `BCEWithLogitsLoss`? 8. Why not augment validation data? 9. Why do we add the fake to both science and difference images? 10. Why are fake magnitudes drawn uniformly?
