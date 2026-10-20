# Phase: ML basics on old data

## Goal
Train a first model on the existing `legacy/tracklets.csv`, understand every part of it, and learn to distrust a score that looks too good.

**Files built:**
- `src/asthunt/models/train_baseline_classifier.py`: scaler + logistic regression
- `tests/models/test_train_baseline_classifier.py`
- `notebooks/ml_basics.ipynb`

---

## How I thought about it
1. **Start with the simplest model that could work.** Logistic regression is fast, has only a few knobs, and you can read what it learned. Every fancier model later has to beat this **baseline**, or it isn't worth its complexity.
2. **Use the data we already have.** `tracklets.csv` exists, so we learn the mechanics on it now, before spending time downloading new data.
3. **Do a toy problem first.** One feature (speed), two classes (stars, asteroids). You can see the whole problem in one histogram, so you can check the model's answer by eye.
4. **Before training, decide which columns are allowed as features.** I checked every column. `mag`, `true_rate` and `true_angle` exist only because we *made* the fakes, so they could never be measured on real sky data.
5. **Look at the data, not just the score.** The score was 99.8%. The histograms explained why, and that explanation matters more than the number.

---

## Concepts

### What a model is
A model is a function with adjustable numbers inside (**parameters**). Training means choosing those numbers so the outputs match the labels.

Logistic regression with one feature:
```
p(asteroid) = sigmoid(w * speed + b)        sigmoid(z) = 1 / (1 + e^-z)
```
- `w` (weight): how strongly speed pushes towards "asteroid".
- `b` (bias): shifts where the 50% point sits.
- The sigmoid squashes any number into 0–1, so the output can be read as a probability.

*Example:* with w = 1 and b = −5, speed 5 gives sigmoid(0) = 0.5, speed 10 gives sigmoid(5) = 0.993, and speed 0 gives sigmoid(−5) = 0.007.

### How training picks w and b
The model is scored with **log loss** (cross-entropy). For each example it adds:
- −log(p) if the truth is 1,
- −log(1 − p) if the truth is 0.

Confident and right costs almost nothing (−log 0.99 = 0.01). Confident and wrong costs a lot (−log 0.01 = 4.6). An optimizer adjusts w and b step by step to make the total smaller. (In the *Neural networks* phase we write that optimizer by hand.)

### Features and labels
- **Feature:** a number you can measure on *new, unlabelled* data. `measured_rate` and `linearity_residual` qualify.
- **Label:** the answer we want to predict (`label`: 1 = real, 0 = bogus).
- **Not a feature:** anything that only exists because you know the answer. In our file, `mag` is missing for 100% of bogus rows and present for 100% of real rows. Using it would be **target leakage**: the label sneaks in through a feature.

### Train / test split
Split the data before training. The model learns only from **train** (75%) and is judged on **test** (25%), which it has never seen.
- `stratify=y` keeps the same real/bogus ratio in both parts.
- `random_state=42` makes the split repeatable.

*Why it matters:* a model can memorise its training data. Only unseen data shows whether it learned a general rule.

### Scaling features
`StandardScaler` rescales each feature to mean 0 and spread 1: `(x − mean) / std`.

Rate is measured in thousands and residual in tenths. Without scaling, the regularization penalty treats them unfairly and the optimizer struggles. The scaler learns its mean and std **from training data only**: `make_pipeline` handles this for us, because `fit` happens on `X_train`.

### Regularization
By default scikit-learn adds a penalty for large weights (strength set by `C`; smaller C = stronger penalty). Large weights make steep, overconfident curves that can fit noise.

In the toy problem this is visible: the classes split around 2 "/hr, but the learned 50% point is near 5 "/hr, because the penalty keeps the curve gentle. That's a trade-off, not a bug. Exercise 4 lets you weaken the penalty.

### Accuracy and the confusion matrix
On our test set:
```
                predicted bogus   predicted real
truly bogus           440                2
truly real              0              441
```
Accuracy = (440 + 441) / 883 = 99.8%. The confusion matrix shows *which* mistakes were made: 2 bogus tracklets called real. In this project that kind of mistake (a false positive) costs a human time to vet. The *Metrics and evaluation* phase goes deeper.

### Feature engineering: same data, better numbers
Rates span 2–5400 and residuals 0.004–2300. A linear model on such skewed raw numbers draws a poor boundary. Taking `log10` of both features spreads them evenly, and test accuracy goes from 99.8% to **100%**. Choosing how to represent the data often matters more than choosing the model.

### Why 100% is bad news here
The smallest bogus residual is **222.7 px**; the largest real one is **2.6 px**. A one-line rule separates them perfectly. That's because the bogus rows were built by picking 4 *random* sources anywhere on the image. Real false tracklets are near-misses, a few pixels off a line. The test set came from the same easy process, so it can't reveal how the model would do on the real sky.

---

## Steps, in order
1. Wrote `train_baseline_classifier()`: a `StandardScaler` → `LogisticRegression` pipeline.
2. Wrote a test: on data where `x > 0` means class 1, the model must predict −2 → 0 and +2 → 1.
3. Toy notebook section: simulated 200 stars and 200 asteroids, trained, printed probabilities, plotted the sigmoid.
4. Loaded `tracklets.csv` and checked every column for leakage (found `mag`).
5. Stratified 75/25 split, trained on `measured_rate` + `linearity_residual`, measured accuracy and the confusion matrix.
6. Plotted feature histograms per class, found the classes don't overlap, and concluded the data is unrealistically easy.
7. Checked that log features reach 100% (exercise 1).

## Commands
```bash
conda activate asthunt
pytest tests/models
jupyter lab notebooks/ml_basics.ipynb
```

## Pitfalls
- **Fitting the scaler on all the data** before splitting leaks test information into training. Keep the scaler inside the pipeline.
- **Judging on training data.** A train score only tells you the model can memorise.
- **Trusting accuracy when classes are unbalanced.** Here it's 50/50; on the real sky it won't be (see *Metrics and evaluation*).
- **Using columns that only exist because you made the data** (`mag`, `true_rate`).

## Quiz
1. In `sigmoid(w * x + b)`, what happens to the curve when `w` gets larger?
2. Why does −log(p) punish confident wrong answers so heavily?
3. Why is `mag` not allowed as a feature, even though it would improve accuracy?
4. Why should the scaler be fitted on training data only?
5. The one-line rule `residual < 100` gets 100%. Give two reasons why that still doesn't mean we can find real asteroids.
