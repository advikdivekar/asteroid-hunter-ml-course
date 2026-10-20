# Phase: Leakage audit

## Goal
Find every way the old dataset lets a model cheat, prove each one with an experiment, and write a spec for a *fair* dataset.

**Files built:**
- `src/asthunt/evaluate/check_split_leakage.py`: lists groups (nights, images) that appear in both train and test
- `src/asthunt/link/fit_linear_motion.py`: fits straight-line motion to detections, returns velocity and residual (the Linking phase reuses it)
- tests for both
- `notebooks/leakage_audit.ipynb`

---

## How I thought about it
1. **A perfect score is a symptom.** In *ML basics* we got 99.8–100%. Before improving the model, I asked: *what would have to be true about the data for this to be so easy?*
2. **Use a checklist.** Leakage has four well-known forms (below). I checked our data against each one instead of guessing.
3. **Prove it, don't just claim it.** "The bogus rows are too easy" is an opinion. Building realistic bogus rows and watching the score fall from 100% to 50% is evidence.
4. **Turn the findings into rules.** Each problem becomes one line in a spec, and every later phase follows that spec.

---

## Concepts

### Leakage
Information reaches the model during training that it **won't have** when used for real. The model looks great in testing and fails in practice.

**1. Target leakage.** A feature contains the answer.
*Ours:* `mag`, `true_rate` and `true_angle` are filled for 100% of real rows and 0% of bogus rows. "Is mag missing?" = the label.
*Rule:* a feature must be measurable on new, unlabelled data.

**2. Group leakage (train/test contamination).** Train and test examples are near-copies, because they share a source.
*Ours:* every tracklet came from the same 4 images. The test set has the same stars, the same noise and the same artifacts as training.
*Rule:* split by **group** (night/field). `check_split_leakage` returns the groups that appear on both sides; it must be empty.

*A second, subtler case:* our near misses are made *from* real tracklets. If a near miss lands in test while its original is in training, the model has seen almost the same example. So derived examples go to the same side as their source.

**3. Shortcut learning.** The classes differ in an easy way that the real world doesn't share. The model learns the shortcut, not the concept.
*Ours:* random bogus tracklets are hundreds of pixels off a line and move "2000 "/hr". No real bogus tracklet looks like that.

**4. Distribution shift.** The data the model is tested on (or used on) differs from the training data.
*Ours, measured:*
| Trained on | Tested on | Accuracy |
|---|---|---|
| legacy (random bogus) | legacy | **100%** |
| legacy (random bogus) | real + near misses | **50%** (coin flip) |
| real + near misses | real + near misses | **99%** |

The model trained on easy data says "real" for anything roughly straight. It never had to learn finer distinctions.

### Hard negatives
Negatives (bogus examples) that look like the positives. The model learns most from them. In our test, adding near misses to training lifted realistic accuracy from 50% to 99%.

How hard they are matters a lot (offset = how far we moved one detection):
| Offset | Accuracy |
|---|---|
| 3–10 px | 99% |
| 1–3 px | 96% |
| 0.5–1.5 px | 89% |

A wrong source 1 px off a line is nearly invisible to position-only features. That's why the project adds the CNN (does each detection *look* real?) and brightness consistency.

### Fitting straight-line motion
Over one night an asteroid moves in a nearly straight line at constant speed:
```
x(t) = x0 + vx · t        y(t) = y0 + vy · t
```
`fit_linear_motion` fits both lines with least squares (`np.polyfit`, degree 1). It returns the velocity and the **max residual**: the largest distance between a detection and its predicted position.

*Example:* detections at x = 100, 105, 118, 115 at t = 0, 0.5, 1, 1.5 h. The third point is 8 px off the trend, so the residual is large and the tracklet is suspicious.

Speed in arcsec/hour = √(vx² + vy²) × pixel scale (1.012 "/px for ZTF).

---

## Steps, in order
1. Listed the four leakage types and checked each against `tracklets.csv`.
2. Target leakage: measured the fraction of missing values per class for `mag`, `true_rate` and `true_angle`.
3. Group leakage: confirmed in `legacy/generate_dataset.py` that all rows used `FITS_FILES[:4]`; showed `check_split_leakage` reports the shared group.
4. Wrote `fit_linear_motion` + tests (perfect line → residual 0; one bad point → large residual).
5. Built one near-miss bogus tracklet per real tracklet (move one detection 3–10 px).
6. Plotted residual histograms: near misses overlap real tracklets, random bogus doesn't.
7. Split real tracklets 75/25 and made near misses follow their source to the same side.
8. Ran the three train/test experiments in the table above.
9. Repeated with smaller offsets to measure how hard near misses get.
10. Wrote the fair-dataset spec.

## The fair-dataset spec (carried forward)
| Problem found | Rule for the new dataset |
|---|---|
| `mag`, `true_*` only filled for real rows | Features = only things measured from detections |
| All rows from the same 4 images | Many nights; split by night; `check_split_leakage` must return `[]` |
| Random bogus = trivially easy | Bogus = real linker output: near misses, artifacts, chance alignments |
| Frames from different months pretending to be 15 min apart | Same-night frames with real timestamps |
| Derived examples could straddle the split | Derived examples follow their source |

## Commands
```bash
pytest tests/evaluate tests/link
jupyter lab notebooks/leakage_audit.ipynb
```

## Pitfalls
- A random row split looks fine but hides group leakage. Ask: *what do my rows share?*
- Fixing leakage usually makes scores **drop**. That's progress: the new number is honest.
- Synthetic negatives are only as good as your imagination. Real linker output (later phases) is the best source of bogus examples.

## Quiz
1. Name the four kinds of leakage and give our example of each.
2. Why did the model trained on legacy data score exactly ~50% on realistic data? What was it effectively doing?
3. Why must a near miss go to the same split as the real tracklet it was made from?
4. A detection is 1 px off a perfect line. Why can't position features alone catch this, and what extra information could?
5. Your friend reports 99.9% accuracy on a new asteroid dataset. What three questions do you ask first?
