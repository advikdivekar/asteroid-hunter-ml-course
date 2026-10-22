# 03. ML primer: the ideas we used, tied to the syllabus

Each idea has three lines: **what it is**, **where it appears in our project**, **the syllabus session it matches** (from the course policy document).
If you can explain these in your own words, you can pass a viva on any part.

## A. The problem setting
**Supervised learning**: learn a function from examples with known answers. *Ours:* XGBoost learns real/bogus blobs from 49 measured numbers and real/bogus tracklets from 17 numbers. **Unsupervised learning** finds structure without labels. *Ours:* PCA compresses each stamp's pixels; k-means groups the junk blobs into 6 kinds. *Syllabus: types of learning, supervised vs unsupervised.*
**Classification vs regression**: classification predicts a category (real/bogus), regression a number. *Ours:* classification; one small regression (a straight-line fit from stack SNR to magnitude in the deep search: mag = 26.25 - 5.23 log10(SNR)). *Syllabus: linear regression.*
**Features and labels**: a feature is a number measurable on new unlabelled data; the label is the answer. *Ours:* `mag` and `true_rate` existed only for planted fakes, so they were illegal features (target leakage). *Syllabus: ML requirements (CO1).*

## B. Models
**Logistic regression**: `p = sigmoid(w.x + b)`, trained by minimising log-loss. *Ours:* the baseline on 6 numbers per detection (AP 0.780), and in the model comparison on tracklets. *Syllabus: logistic regression, hypothesis, cost function.*
**Gradient descent**: repeat `w <- w - learning_rate x gradient`. *Ours:* written by hand in numpy in `notebooks/neural_networks.ipynb`, then Adam in PyTorch for the CNN (kept as a comparison). *Syllabus: gradient descent, cost function.*
**Neural network**: layers of "weighted sum, then non-linearity". Without the non-linearity, stacked layers collapse into one linear function. *Ours:* the from-scratch notebook and the CNN, which we keep only as a comparison: the deployed model is classical. *Syllabus: NN hypothesis, cost, activation functions.*
**Backpropagation**: the chain rule applied from the output back, so one sweep gives every gradient. For sigmoid + cross-entropy the output error is simply `p - y`. *Ours:* coded by hand and checked numerically (matches to 8 decimals). *Syllabus: backpropagation.*
**Convolutional neural network (CNN)**: filters (3x3 grids of learned weights) slide across an image looking for a local pattern; pooling shrinks the image so deeper filters see larger areas. *Ours:* 5 convolution layers over 3 stacked pictures; 72,145 weights; AP 0.898, a tie with our classical model (0.902), so comparison only. *Syllabus: neural networks (applied).*
**Feature engineering**: turning raw data into numbers with a meaning. *Ours:* 29 numbers per stamp (star shape, "already in the old picture?", sharpness, dark hole next to it...). The most useful one: how much the difference picture looks like a star.
**SVM (support vector machine)**: finds the widest margin between the classes; with an RBF kernel the boundary can curve. *Ours:* compared as a stage-1 model (AP 0.886 validation, 0.877 test). *Syllabus: SVM, relation to logistic regression.*
**PCA**: rotates the data onto the directions of largest variance and keeps the first few. *Ours:* 675 pixels -> 20 numbers (78% of the variance), used as extra features. *Syllabus: PCA.*
**k-means clustering**: places k centres and assigns each point to the nearest, repeating until stable. *Ours:* 6 clusters of junk: 93% faint noise, star leftovers, bright stars, old-picture-only. *Syllabus: clustering, k-means.*
**Decision tree -> random forest -> gradient boosting**: a tree asks yes/no questions; a forest averages many random trees (bagging, lowers variance); boosting builds trees one after another, each correcting the errors so far. *Ours:* XGBoost on 49 numbers per blob (600 trees, depth 4, the deployed stage-1 model) and on 17 numbers per tracklet (300 trees, depth 3, learning rate 0.05). *Syllabus: trees, bagging, random forests, boosting.*

## C. Making models behave
**Overfitting**: the model memorises noise; training error falls, validation error rises. *Ours:* a demo on 40 points (validation loss lowest at step 22 then climbs), and early stopping in the CNN. *Syllabus: bias and variance, evaluating a hypothesis.*
**Regularisation**: anything that stops the model fitting noise: a penalty on large weights (L2 in logistic regression, strength `C`), **dropout** (0.3 in the CNN head), **early stopping** (patience 4), **subsampling** (XGBoost uses 80% of rows and columns per tree), **data augmentation** (random rotations and mirrors). *Syllabus: regularization (CO2).*
**Feature scaling / normalisation**: put numbers on a comparable scale. *Ours:* per-stamp robust normalisation `(x - median) / (1.4826 MAD)`, clipped to [-10, 50]. Trees need no scaling.
**Class imbalance**: real objects are about 3% of detections. We balanced *training* (all real + 5 bogus per real = 16.7% real) but kept *validation* realistic. Side effect: raw probabilities are inflated; fix with base-rate adjustment or isotonic calibration.

## D. Evaluating honestly (the heart of "model evaluation", worth 3 marks)
**Train / validation / test**: train fits the model; validation chooses settings and thresholds; test is opened once. *Ours:* **split by night**, decided in a config file before modelling. *Syllabus: train-validation-test.*
**Data leakage**: information at training time that will not exist at use time. Four forms we found: target leakage (`mag`), group leakage (all tracklets from 4 images), shortcut learning (random bogus tracklets 222+ px off a line), distribution shift (test night sharper). Evidence: accuracy 100% -> 50% -> 99% depending on the test data.
**Cross-validation**: repeat train/test over several splits. *Ours:* `GroupKFold` by night (never random rows). **Cross-fitting** is the same idea used so a second model does not see over-confident first-model scores.
**Confusion matrix, precision, recall, false-positive rate**: see `04_Satvik/07_Diagrams/04_confusion_metrics.svg`. *Syllabus: error metrics for skewed classes, precision/recall trade-off.*
**Accuracy paradox**: "always bogus" scores 96.8% accuracy and finds nothing. With rare positives use recall, precision and the **precision-recall curve (average precision)**. ROC AUC looked excellent (0.95) when precision at useful recall was under 20%. *Syllabus: ROC curves.*
**Threshold / operating point**: pick the cut on validation to keep 90% of real objects, then count false alarms. **Calibration**: do scores mean probabilities? (reliability plot; isotonic regression; Bayes base-rate correction).
**Error analysis**: look at the model's most confident mistakes. *Ours:* the CNN's top "bogus" scores (in the first version) were known asteroids -> exposed a 2.9 arcsec catalogue offset (label error). 9 "false positives" on the test night were real known asteroids our labels missed. *Syllabus: error analysis.*
**Completeness curve (injection-recovery)**: the domain's version of a recall-by-condition plot (recall vs brightness). **Adversarial validation**: train a classifier to tell fakes from real; AUC 0.47 means it cannot -> fakes are realistic.
**Fair comparison**: compare models at the same false-alarm level, not at each one's own threshold.

## E. Ideas from the research side
**Stacking**: one model's output is another's input (stage-1 scores are features for the tracklet XGBoost). **Ensembling**: averaging the two models gave only a tie.
**Occam's rule**: when models tie, use the simplest (the CNN and a multi-frame CNN both tied XGBoost; we deploy XGBoost).
**Negative results are results**: 3x more data did not help; the limit was physics (photons in one frame), not model size.
**Look-elsewhere effect / multiple testing**: pick the best of 4,888 trials and noise looks significant. Calibrate on pure noise (6 sigma), and run a **null test** (shuffle the frame times; real movers stop lining up; we got 0 false tracks).

## F. Syllabus topics we did NOT use (be honest about them)
Linear regression as a main model, polynomial regression, the normal equation, hierarchical clustering, regression trees. (SVM, PCA and k-means are now used: section B.) See `04_Course_Mapping.md` for how to cover them with small extra experiments if your faculty expects them.
We touch **anomaly detection** only in spirit: "tracks that match nothing in the catalogue" is an anomaly filter, but it is a catalogue lookup, not a learned anomaly model.

## Self-test
1. Why does a network need a non-linear activation? 2. What does `p - y` have to do with backprop? 3. Why is accuracy useless for 3% positives?
4. What is the difference between bagging and boosting? 5. Why split by night instead of randomly? 6. What is cross-fitting and what goes wrong without it?
7. Why might a model with a worse depth on the scoreboard be just as good as another? 8. What is the look-elsewhere effect?
