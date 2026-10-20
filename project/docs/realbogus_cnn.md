# Phase: Real/bogus CNN

## Goal
Train a convolutional neural network that looks at the science / reference / difference stamps of a detection and outputs p(real). Prove it beats the feature baseline, then produce scores for every detection of every night for the linking stage, without leaking training data into those scores.

**Files built:**
- `models/define_realbogus_cnn.py`: 5 conv layers in 3 blocks + a small dense head (72,145 weights)
- `models/train_realbogus_cnn.py`: training loop: augment → forward → loss → backward → Adam; validation AP each epoch; keep the best epoch; early stopping
- `models/predict_realbogus.py`: batched scoring in eval mode
- `models/cross_fit_realbogus_scores.py`: out-of-fold scores for training nights (4 folds by night)
- `configs/train.yaml`, tests (including a tiny CNN that must learn "blob in the centre = real"), `notebooks/realbogus_cnn.ipynb`

---

## How I thought about it
1. **Small network, honest data.** Our stamps are 31×31 and the patterns are local (blob, spike, dipole, edge). A few conv layers suffice, and a small model is fast on the M3's GPU and harder to overfit.
2. **Judge by validation AP on realistic data**, not training loss. Training is balanced and augmented; validation has the true 1:30 imbalance.
3. **Compare against the baseline on identical detections** at the same recall, so the comparison is fair.
4. **Look at the model's most confident mistakes.** That's where I found a labelling bug (below).
5. **Plan for stacking.** The CNN's scores become features for the tracklet classifier. A model scoring its own training data is over-confident, so training nights are scored by models that never saw them (**cross-fitting**).

---

## Concepts

### Convolution
A 3×3 filter slides over the stamp; at each position it multiplies 9 pixels by 9 learned weights and sums them. One filter = one pattern detector (e.g. "bright centre, dark ring"). With 3 input channels, each filter has 3×3×3 weights and looks at science, reference and difference at once. That's how it can learn "bright in science and difference, *nothing* in reference".

### Pooling and depth
Max-pooling halves the image (31 → 15 → 7 → 3), so deeper filters see bigger areas: early layers find edges and blobs, later layers combine them into "point source", "dipole" or "streak".

### The head
Flatten (64×3×3 = 576 numbers) → dropout 0.3 → 64 neurons → 1 output logit. `sigmoid(logit)` = p(real).

### Training recipe
| setting | value | why |
|---|---|---|
| loss | `BCEWithLogitsLoss` | binary cross-entropy, numerically stable |
| optimizer | Adam, lr 0.001 | robust default |
| batch | 256 | fits easily, stable gradients |
| augmentation | random 90° rotations + mirror | the sky has no "up" |
| early stopping | patience 4 on validation AP | keeps the best epoch, stops when it stalls |

The final run used all 25 epochs; the best validation AP (0.899) was at epoch 21.

### Results (validation nights, all 67,041 detections, 3.2% real)
| model | AP | precision at 90% recall | false positives per frame |
|---|---|---|---|
| logistic regression on 6 features | 0.780 | 0.191 | 813 |
| **CNN** | **0.899** | **0.522** | **176** |

At the same completeness, the CNN cuts false alarms **4.6×**. On the untouched test night: **AP 0.938**.

(The baseline's numbers differ slightly from the metrics lesson: 813 vs 804 false positives per frame. Each notebook draws its balancing sample of bogus rows with a different random seed. Differences of that size are noise, not signal.)

The completeness-vs-magnitude curves of both models look alike at 90% recall, because we *fixed* recall. The CNN's gain is purity, which is what makes linking cheap: after the CNN filter only ~24% of detections remain, and seed pairs drop from 52k to 3.5k.

### The model found our labelling bug
The CNN's top-scoring "bogus" detections were clean new point sources, and **every one was a known asteroid** (V = 17–20). Investigation:
- SkyBoT's predicted positions sat **~2.9" from our detections in a constant direction** (−2.4, +1.5 px) on every night.
- Our WCS matches Gaia stars to 0.04 px. **JPL Horizons** matches our detections to 0.1–0.3".
- So SkyBoT's predictions carry an offset, and our 2-px match radius missed bright real asteroids.
- Fix: 4-px radius for known asteroids → +2 to +70 real labels per night → retrained → AP 0.877 → 0.894.

**Lesson (data-centric ML):** a model's most confident disagreements with the labels are the cheapest way to find label errors.

### Cross-fitting (out-of-fold predictions)
The tracklet classifier uses CNN scores as features. If the CNN scored nights it trained on, those scores would be unrealistically confident. The tracklet classifier would learn "score_min is always high for real things", which wouldn't hold on new nights. So:
```
8 training nights → 4 folds of 2 nights
for each fold: train a CNN on the other 6 nights, score the 2 held-out nights
validation + test nights: scored by the final CNN (never trained on them)
```
This is the standard way to **stack** models without leakage.

---

## Steps, in order
1. Wrote the CNN definition, training loop and prediction function + tests (shape test; a tiny CNN must reach AP > 0.95 on a synthetic blob task).
2. Hit `OMP: Error #15` (two OpenMP runtimes, from pip PyTorch + conda) → switched to conda-forge PyTorch.
3. Trained → AP 0.877. Inspected mistakes → found the SkyBoT offset → verified with Gaia and JPL Horizons → relabelled → AP 0.894.
4. Found the pointing-shift problem in the Full-run phase (see `docs/linking.md`) → rebuilt all data → retrained → **AP 0.899**.
5. Cross-fitted 4 folds and saved `realbogus_scores__<field>_<night>.npy` for every night.
6. Saved `models/realbogus_cnn__<date>__<git sha>.pt` + a JSON with nights, settings and history.

## Commands
```bash
pytest tests/models
jupyter lab notebooks/realbogus_cnn.ipynb      # ~7 minutes on an M3 Pro (MPS)
```

## Pitfalls
- **Mixing pip and conda PyTorch:** a duplicate OpenMP runtime crashes the process.
- **Calling `.numpy()` inside training:** use `torch.no_grad()` and `model.eval()` for scoring (dropout off).
- **Comparing models at different thresholds:** fix recall, then compare false positives.
- **Feeding in-sample scores to a second model:** use cross-fitting.
- **Assuming the labels are right.**

## Quiz
1. Why does a 3-channel filter help tell a real source from a bad subtraction?
2. Training loss is 0.08 and validation loss 0.035. Why can validation loss be lower here?
3. At 90% recall, why do the completeness curves of the CNN and the baseline look so similar?
4. Explain cross-fitting in two sentences. What would go wrong without it?
5. How did the CNN reveal the SkyBoT offset?
