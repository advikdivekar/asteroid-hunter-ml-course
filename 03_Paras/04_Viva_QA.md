# Paras: viva questions and model answers

## A. Labels and data preparation
**1. Where do your labels come from?**
Three sources: planted fake asteroids (perfect labels), known asteroids from catalogues (real labels, positions from JPL Horizons), and everything else called bogus (some label noise). The sky itself is not labelled.

**2. What is injection-recovery and why do you add the fake to both science and difference images?**
Plant objects with known properties, run the normal pipeline, count what comes out. A real asteroid is in tonight's science image but not in the reference; adding the same stamp to science and difference keeps `reference = science - difference` unchanged, which is exactly that situation.

**3. Why draw fake magnitudes uniformly instead of like real asteroids?**
So every brightness bin has enough fakes to measure completeness. Real asteroids are mostly faint; remember this when quoting rates, since our training mix is not the sky's mix.

**4. How do you make the fakes realistic?**
Use ZTF's own PSF for each frame, sub-pixel rendering with exact flux, Poisson noise, the real frame times, and motion straight on the sky (converted to each frame's pixels). Then test realism with adversarial validation.

**5. What is adversarial validation and what did it show?**
Train a classifier to separate two datasets that should look the same (fakes vs real known asteroids, magnitude-matched, cross-validated by night). AUC about 0.5 means indistinguishable; we got 0.46-0.48 (about 0.47 +/- 0.02).

**6. How many chance matches do you expect when matching detections to catalogue positions?**
P(random point within 2 px of a detection) = detections x pi x 4 / good area = 1.15%. With 114 known asteroids in the field that is about 1.3 chance matches per frame, which is what we saw, so faint "known" matches are mostly wrong. Hence only V <= 20.5 is used.

**7. What was wrong with SkyBoT positions and how did you find out?**
The CNN's most confident "bogus" detections were known asteroids. SkyBoT's predictions sat about 2.9 arcsec (2.7 px) from detections in a constant direction on every night. Our WCS matches Gaia to 0.04 px and JPL Horizons to 0.1-0.3 arcsec, so SkyBoT was off. We switched to Horizons positions (median error 0.26 px vs 2.72 px) with a 2-px radius.

**8. Why did you pass the observatory code I41 to the catalogue query?**
By default positions are for the centre of the Earth; ZTF is about 6,400 km away, which shifts a 1 AU asteroid by up to about 9 arcsec, more than 4 pixels.

**9. What is label noise in your "bogus" class?**
Variable stars, other real transients, and undiscovered asteroids are all labelled bogus. So a "false positive" on real data may be a real object, which is the thing we hunt. On a training night it is a tiny fraction of 36,000 detections, so the model barely notices.

**10. Describe the stamp and why three channels.**
3 x 31 x 31: science, reference, difference. A real new object shows in science and difference but not reference; a bad subtraction shows a star in both and a dipole in difference; a cosmic ray is a spike narrower than a star. The CNN compares the three panels like a person does.

**11. How do you normalise stamps and why?**
`(value - median) / (1.4826 x MAD)` per stamp and channel, clipped to [-10, 50]. Median and MAD are robust to a bright star; "5" then means 5 sigma on any night; clipping stops a saturated star dominating. Nothing is fitted on training data, so there is no leakage.

**12. Why augment, and why only the training data?**
Rotations by 0/90/180/270 plus mirror give 8 versions per stamp: the sky has no up, and it acts like more data. Validation and test must measure performance on data as it really is, so they are never augmented.

**13. Why balance the training set and what is the cost?**
With 3% positives the gradient is dominated by easy bogus examples. We use all real + 5 bogus per real (16.7% real). The cost is inflated probabilities (the model thinks real is 16.7% common), fixed afterwards by base-rate adjustment or calibration. Validation is not balanced, so precision and false-alarm numbers stay honest.

**14. What went wrong with the first float16 stamps?**
Saturated star cores exceed 65,504, the largest float16 value, and became `inf`. Fix: clip before converting. Found by scanning every stamp file after the build, not just trusting the code.

## B. Neural network theory
**15. Why do you need a non-linear activation?**
Without it, stacked linear layers collapse into one linear function and depth adds nothing. ReLU makes each hidden neuron a "bent line"; many bent lines give a curved boundary (half-moons: logistic regression 0.88, network 0.94).

**16. Explain the forward pass of a one-hidden-layer network.**
`z1 = X W1 + b1`, `h = ReLU(z1)`, `z2 = h W2 + b2`, `p = sigmoid(z2)`.

**17. What is the loss and why does a confident wrong answer cost so much?**
Binary cross-entropy `-mean(y log p + (1-y) log(1-p))`. At p = 0.01 for a true positive the loss is -log 0.01 = 4.6, while a confident right answer (p = 0.99) costs 0.01.

**18. Derive why dL/dz = p - y for sigmoid with cross-entropy.**
sigma'(z) = sigma(z)(1 - sigma(z)); the derivative of the loss with respect to p is (p - y) / (p(1 - p)); multiplying gives p - y: the sigmoid's slope cancels the log's slope.

**19. What is backpropagation?**
The chain rule applied from the output backwards. Output weights: h^T (p - y) / n. Hidden error: (p - y) W2^T x ReLU'(z1). Hidden weights: X^T (hidden error) / n. Each layer reuses the next layer's result, so one backward sweep gives every gradient.

**20. How did you check your hand-written gradients?**
Numerical gradient: nudge one weight by epsilon = 1e-5 on both sides, `(L(w+e) - L(w-e)) / 2e`. It matched backprop to 8 decimals (-0.02896520). Removing the ReLU' term makes the check fail at once.

**21. What happens with a learning rate of 5.0, and what does Adam do?**
Gradient descent overshoots and diverges. Adam adapts each weight's step size using its recent gradients; a robust default.

**22. Why must initial weights be random?**
If all weights start equal, every neuron gets the same gradient and stays identical forever.

## C. The CNN
**23. How many parameters does the first conv layer have?**
(3 x 3 x 3 + 1) x 16 = 448. In total the network has 72,145.

**24. Why do spatial sizes go 31 -> 15 -> 7 -> 3?**
Padded 3x3 convolutions keep the size; each max-pool halves it (integer division). 64 channels x 3 x 3 = 576 numbers feed the dense layer.

**25. What does dropout do?**
It randomly switches off 30% of the neurons in the head during training so the network cannot rely on any single neuron; it is turned off when scoring (`model.eval()`).

**26. How do you know when to stop training?**
Validation average precision is checked every epoch; we keep the best epoch and stop after 4 epochs without improvement (the best was epoch 21, AP 0.899).

**27. Why judge by validation AP and not training loss?**
Training loss measures memorising; only held-out nights show generalisation. Training is balanced and augmented, validation is realistic, so their losses are not even comparable (training 0.08 vs validation 0.035 can both be right).

**28. Why is the CNN "better" if the completeness curves look alike at 90% recall?**
Because we fixed recall; the gain is purity: 176 instead of 813 false alarms per frame at the same recall. Purity makes linking cheap: only about 24% of detections remain and seed pairs drop from 52,000 to 3,500.

**29. What is cross-fitting?**
Score each training night with a CNN trained on other nights (4 folds of 2 nights). The tracklet model uses CNN scores as features; scoring nights a CNN trained on gives over-confident scores that would not hold on new nights. Folds are by night, never by detection.

**30. What mistakes did the CNN make that taught you something?**
Its top-scoring "bogus" detections were known asteroids our labels had missed: a label error, not a model error. On the test night, 9 tracklets counted as false positives by the tracklet model were also real known asteroids, so the effective false-positive count was 0.

## D. Judgement
**31. Is 0.938 AP on the test night impressive?**
It beats the baseline everywhere, but AP on a set rich in planted fakes flatters; the evaluation chapter uses depth, known asteroids recovered and unexplained alarms per night.

**32. What would you change with more time?**
Human-labelled real/bogus examples to complement injections (as ZTF's own classifier uses), a bigger stamp (63 x 63), a calibration step so scores read as probabilities, and an ablation of the three channels.

**33. "Did you write the model yourself?"** Answer truthfully (see `00_Start_Here/07_AI_Use_and_Integrity.md`): the project was built with an AI coding assistant under Advik's direction; say what you re-ran or changed (for example the from-scratch notebook, the parameter count) and explain the network on the whiteboard without notes.
