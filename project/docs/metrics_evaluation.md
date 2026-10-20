# Phase: Metrics and evaluation

## Goal
Learn to judge a classifier the way this problem demands: real objects are rare (3% of detections), bogus ones are plentiful, and a human has to look at every alert. Then measure a simple baseline that every later model must beat.

**Files built:**
- `evaluate/compute_classification_metrics.py`: confusion matrix, recall, precision, FPR, average precision, ROC AUC
- `evaluate/choose_threshold.py`: the threshold that keeps a target fraction of real objects
- `evaluate/calibrate_probabilities.py`: isotonic regression from scores to honest probabilities
- `evaluate/adjust_for_base_rate.py`: Bayes' rule to undo balanced-training inflation
- `evaluate/measure_completeness.py`: completeness ± uncertainty per magnitude (or speed) bin
- tests for all five, `notebooks/metrics_evaluation.ipynb`

**The baseline:** logistic regression on 6 numbers per detection (`snr, peak, npix, a, b, flux`), trained on balanced training nights, evaluated on **all 67,041 validation detections** (2,129 real, 3.18%).

---

## How I thought about it
1. **What does a mistake cost?** A missed asteroid is a lost discovery. A false alarm costs a person a minute of vetting. So we fix a **recall target** (90%) and then minimise false alarms. We don't just maximise accuracy.
2. **Evaluate on realistic data.** Validation keeps the true 1:33 imbalance, so precision and false alarms are what a real night would give.
3. **Choose thresholds on validation, never on test.** Every choice made using test data leaks into the final score.
4. **Report per magnitude.** One overall recall number hides where the model fails. Astronomers report completeness curves, and so do we.
5. **Check that probabilities mean what they say.** Training was balanced (16.7% real), so raw scores are inflated. I compared two corrections.

---

## Concepts

### The confusion matrix (baseline, threshold 0.5, validation)
```
                 called real   called bogus
truly real          1,547           582        → recall    = 1547 / 2129  = 72.7%
truly bogus           493        64,419        → FPR       = 493 / 64912  = 0.76%
                                                 precision = 1547 / 2040  = 75.8%
```
493 false positives across 10 frames = **49 per frame** for a human to look at.

### Why accuracy fails here
| model | accuracy | recall |
|---|---|---|
| always "bogus" | **96.8%** | 0% |
| baseline | 98.4% | 72.7% |

Almost the same accuracy, completely different usefulness. With rare positives, accuracy mostly measures how well you recognise the common class.

### Recall, precision, false-positive rate
- **Recall (completeness):** of all real objects, what fraction did we keep?
- **Precision (purity):** of everything we flagged, what fraction is real?
- **False-positive rate:** of all bogus, what fraction did we wrongly flag?

The base-rate trap: a 1% FPR sounds tiny, but 1% of 65,000 is 650 false alarms, a big fraction of the ~2,100 real objects.

### ROC vs precision–recall
- ROC AUC = **0.95**: looks excellent.
- Average precision = **0.78**: the honest summary.

ROC's x-axis divides by *all* bogus (65k), so hundreds of false alarms look like a tiny step. Precision divides by the number of *alerts*, so it shows directly how polluted the output is. **For rare positives, use precision–recall.**

### Choosing the operating point
`choose_threshold(labels, scores, 0.90)` → threshold **0.082**:
- recall 90.0%, precision **19.3%**, **804 false positives per frame**.

Pushing from 73% to 90% recall multiplies false alarms by 16. (The CNN, next phases, brings this down to 176 per frame.) That's the trade-off a better model (the CNN) must improve.

### Calibration: do scores mean probabilities?
Bin the predictions and compare each bin's average prediction with the fraction that's actually real (a **reliability plot**):
- **Raw scores:** far below the diagonal. A prediction of 0.6 is real only ~30% of the time, because the model learned that real is 16.7% common when it's really 3.1%.
- **Base-rate adjustment** (`adjust_for_base_rate`): Bayes' rule in odds form, no fitting:
  `new odds = model odds × (true prior odds / training prior odds)`.
  Example from the docstring: 0.9 → **0.043** when real is 1 in 1,000 instead of 1 in 6.
- **Isotonic calibration** (`calibrate_probabilities`): learns a never-decreasing curve from scores to observed fractions. We fitted it on one half of validation and checked it on the other half; fitting and judging on the same rows would flatter it.

Both corrections end up close to the diagonal. Calibration doesn't change the *ranking*, so recall and precision at matching thresholds stay the same. It changes what you can honestly *say* ("this candidate is ~70% likely real").

### Completeness curves
- **Detected:** ~90–95% for bright fakes (the rest landed on masked pixels), falling through 50% near mag 20.6.
- **Detected and kept at 90% recall:** follows the detected curve to mag ~19.8, then drops faster. At mag 21 the baseline throws away most of what was detected.

Every point has an error bar √(p(1−p)/n). **The gap between the two curves is the target for the CNN.**

---

## Steps, in order
1. Wrote the five evaluation functions + tests (each test uses a hand-checkable example: confusion counts, worked Bayes example, overconfident synthetic scores).
2. Trained the feature baseline on balanced training nights.
3. Notebook: accuracy paradox, confusion matrix and false alarms per frame, ROC vs PR, 90%-recall threshold, reliability plot with both corrections (split-half), completeness vs magnitude.
4. Corrected one notebook sentence after seeing the numbers (I'd claimed the do-nothing model had *higher* accuracy; it's 96.8% vs 98.4%).
5. Numbers here are from the final rebuild (after the label and sky-coordinate fixes).

## Commands
```bash
pytest tests/evaluate
jupyter lab notebooks/metrics_evaluation.ipynb
```

## Pitfalls
- Reporting accuracy or ROC AUC alone for a rare-class problem.
- Choosing a threshold on the test set.
- Calibrating and evaluating the calibration on the same data.
- Quoting balanced-training probabilities as real-world probabilities.
- Completeness without error bars: small bins jump around by chance.

## Quiz
1. With 2,099 real and 64,974 bogus, a model has FPR 2% and recall 95%. What's its precision?
2. Why can ROC AUC be 0.95 while precision at useful recall is under 20%?
3. A model trained on 50% real says 0.8. Real objects are 3% of the data. What's the adjusted probability?
4. Why fit the calibrator on half of validation and check it on the other half?
5. Why do we fix recall first and then minimise false alarms, instead of maximising accuracy?
