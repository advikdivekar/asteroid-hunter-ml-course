# Paras: labels, dataset building and the neural network (CNN)

**Marks this part carries:** Model Development (4, shared with Satvik) and the model-ready data part of Data Preprocessing (3, shared with Aaryan).
**Why it matters:** this is the deep-learning part of the project, and the part where "garbage labels in, garbage model out" is most visible. It also contains the best data-centric story: the model found a labelling error.

## Your four jobs
1. **Where labels come from:** injection-recovery (planted fakes), known asteroids from catalogues, and "everything else is bogus", with the label-noise caveats and the quality checks.
2. **Building the dataset:** cutting 3 x 31 x 31 stamps, normalising, augmenting, balancing classes, assembling train/validation sets without leakage.
3. **The neural network:** from "a neuron" to backpropagation (the from-scratch notebook), then the real CNN: architecture, training loop, regularisation, results versus the baseline.
4. **Error analysis and cross-fitting:** what the model's most confident mistakes revealed, and how its scores are produced without leakage for the next stage.

## What is NOT yours
Downloads and the first detection step: Aaryan. Linking, tracklet features, trees and the metric definitions: Satvik. Scoreboard, deep search and viewer: Advik. You hand Satvik a `detections` table with a `score` column and a threshold.

## Where your code is (in `project/`)
| Area | Files | What they do |
|---|---|---|
| Simulate | `src/asthunt/simulate/render_point_source.py`, `plan_injections.py`, `inject_point_source.py`, `plant_fakes_in_exposure.py` (+ trailed versions) | draw a fake from ZTF's own PSF, choose brightness/speed/position, add it to science and difference |
| Label | `src/asthunt/label/match_detections_to_positions.py` (KD-tree), `query_known_asteroids.py`, `query_horizons_positions.py`, `refine_known_positions.py`, `load_known_asteroids.py`, `label_detections.py` | decide real or bogus for each blob |
| Dataset | `src/asthunt/dataset/build_night_detections.py`, `normalize_stamps.py`, `augment_stamps.py`, `balance_classes.py`, `assemble_stamp_dataset.py`, `relabel_night_detections.py`, `make_unit_tag.py`, `make_unit_rng.py` | one night -> detections + stamps; model-ready arrays |
| Stamps | `src/asthunt/detect/cut_triplet_stamps.py` | 31x31 science / reference / difference cut-outs |
| Models | `src/asthunt/models/define_realbogus_cnn.py`, `train_realbogus_cnn.py`, `predict_realbogus.py`, `cross_fit_realbogus_scores.py` | the CNN: define, train, score, cross-fit |
| Realism check | `src/asthunt/evaluate/measure_injection_realism.py` | adversarial validation of fakes vs real asteroids |
| Config | `configs/inject.yaml`, `dataset.yaml`, `train.yaml` | all the settings |
| Tests | `tests/simulate`, `tests/label`, `tests/dataset`, `tests/models` | includes a tiny CNN that must learn "blob in the centre = real" |
| Lessons and notebooks | `docs/injection_labelling.md`, `label_quality.md`, `dataset_building.md`, `neural_networks.md`, `realbogus_cnn.md`; notebooks with the same names | |

## What you hand over
- The model and its numbers for the report: architecture, training recipe, results versus baseline, error analysis, cross-fitting.
- Figures: the triplet picture, the CNN architecture, the training/validation curves, a few correct and wrong predictions (from the notebook).
- A clear statement of label sources and their weaknesses (this is critical for the evaluation chapter).

## Your seven key messages
1. **The sky is not labelled, so we make labels** (planted fakes + known asteroids + "the rest"), and we say honestly where each one is weak.
2. **A real asteroid shows in science and difference but not in the reference**, which is why the CNN gets all three pictures at once.
3. **A small CNN is enough:** 31x31 stamps, local patterns, 72,145 weights; fast on a laptop GPU and hard to overfit.
4. **We judge by validation average precision on realistic (imbalanced) data, not by training loss.** Training is balanced and augmented; validation keeps the true 1:33.
5. **CNN vs baseline:** AP 0.899 vs 0.780; at 90% recall false alarms fall from 813 to 176 per frame (4.6x); on the untouched test night AP 0.938.
6. **The CNN found a labelling bug:** its most confident "bogus" detections were known asteroids; the catalogue positions (SkyBoT) were about 2.9 arcsec off. Fixed by using JPL Horizons.
7. **Cross-fitting** stops the next model from learning over-confident scores.

## Numbers you must know by heart
| Number | Meaning | Source |
|---|---|---|
| 31 x 31 x 3 | stamp size and channels | `configs/detect.yaml` |
| 72,145 | learnable weights in the CNN | `docs/realbogus_cnn.md` (derivation in your knowledge pack) |
| 31 -> 15 -> 7 -> 3 | spatial size after the three max-pools | `define_realbogus_cnn.py` |
| 0.001 / 256 / 25 / 4 | learning rate / batch / max epochs / patience | `configs/train.yaml` |
| 0.3 | dropout | `define_realbogus_cnn.py` |
| 0.780 -> 0.899 -> 0.938 | AP: baseline (val), CNN (val), CNN (test night) | `docs/realbogus_cnn.md` |
| 813 -> 176 | false positives per frame at 90% recall | same |
| 1,059 : 36,161 | real : bogus detections on one training night (1 : 34) | `docs/injection_labelling.md` |
| 8,863 real + 5 bogus per real = 53,178; 16.7% real | the balanced training set | `docs/dataset_building.md` |
| 67,041; 3.2% real | the realistic validation set | same |
| 2.72 px -> 0.26 px | SkyBoT vs Horizons median position error | `docs/label_quality.md` |
| 0.46-0.48 | AUC of "fake vs real" classifier (0.5 = indistinguishable) | same |
| 1.15% and 1.3 | chance of a random point landing within 2 px of a detection; expected chance matches per frame | `docs/injection_labelling.md` |

## Hands-on tasks (do these yourself)
1. `cd project && pytest tests/simulate tests/label tests/dataset tests/models -q`.
2. Re-run `notebooks/neural_networks.ipynb` end to end; reproduce the gradient check; change the learning rate to 5.0 and describe what you see.
3. Re-run `notebooks/realbogus_cnn.ipynb` (about 7 minutes on an M3); look at the 12 most confident mistakes yourself and say what each one is.
4. Do the parameter-count derivation by hand (knowledge pack, section 4.2) and check it equals 72,145.
5. Change `dropout` from 0.3 to 0.0 in a copy and predict (then observe) the effect on the gap between training and validation.

## Pitfalls
- Saying "the CNN is 99% accurate": accuracy is meaningless at 3% positives; use AP and false alarms at fixed recall.
- Forgetting that training is balanced (16.7% real) while the real sky is about 3%: raw probabilities are inflated.
- Calling labels "ground truth": they are "best available", with known noise.
- Calling `.numpy()` on a tensor with gradients, or forgetting `model.eval()` when scoring (dropout must be off).
- Comparing models at different thresholds instead of at fixed recall.
