# Phase: Dataset building

## Goal
Turn 11 nights of images into model-ready data, and learn the processing steps between raw pixels and a training batch:
**build → check the split → explore → normalize → augment → balance → assemble → version**.

**Files built:**
- `dataset/build_night_detections.py`: one night: load → plant fakes → detect → label → describe → stamps; saves 3 files
- `dataset/normalize_stamps.py`: (value − median) / robust noise, per stamp and channel
- `dataset/augment_stamps.py`: random rotation / mirror (training only)
- `dataset/balance_classes.py`: all real rows + a sample of bogus rows
- `dataset/assemble_stamp_dataset.py`: several nights → (stamps, labels, table), balanced or realistic
- `configs/dataset.yaml` (`bogus_per_real: 5`), tests (including an end-to-end run on tiny fake images), `notebooks/dataset_building.ipynb`

---

## How I thought about it
1. **One driver per night, saving to disk.** Building a night takes ~40 s; training takes many runs. Building once and re-using the files separates "make data" from "use data", so both can be re-run on their own.
2. **Save everything, choose later.** We save *every* detection's stamp (~230 MB per night as float16), not just a training sample. Balancing happens when assembling, so the same files serve training (balanced), validation (realistic) and the search (all detections).
3. **Different nights, different random fakes, repeatable.** Each night's random generator is seeded with `[project seed, night date]`. A first draft used Python's `hash(night)`, which changes on every run; I caught it before it ran.
4. **Test end to end on tiny fake data.** The test writes 2 fake 200×200 frames and runs the whole night build without network access. It caught a real bug: astropy's `pixel_to_world` choked on pandas columns.
5. **Check the output, not just the code.** After the build I scanned every stamp file: ~10 stamps per night held `inf`. Saturated star cores exceeded float16's maximum (65,504). Fix: clip before converting.

---

## Concepts

### The pipeline from pixels to batch
```
raw FITS ──build_night_detections──► detections table + float16 stamps + injections table   (per night, on disk)
                                         │
             assemble_stamp_dataset ─────┤  select nights → (balance) → normalize
                                         ▼
                              stamps (n, 3, 31, 31) float32 + labels
                                         │
                   training loop ────────┤  augment each batch (random rotate / flip)
                                         ▼
                                       model
```

### What we built (all 11 nights)
| split | nights | detections per night | real per night | bogus per real |
|---|---|---|---|---|
| train | 8 | 32–41k | 1,053–1,199 | 27–37 |
| validation | 2 | 32–35k | 1,017–1,112 | 27–34 |
| test | 1 | 41.5k | 1,342 | 30 |

The test night has **83 known-asteroid detections**; the others have 2–32. That's the opposition effect. (With the first, too-tight 2-px match radius it was 12 vs 0–5. See the update in `docs/injection_labelling.md`.)

### Splitting by night (and checking it)
`check_split_leakage(train, validation)` → `[]`, and the same for train/test and validation/test. Every detection, fake and stamp from a night lives in exactly one split.

### Exploratory data analysis (EDA)
Before modelling, we compared real and bogus on simple numbers (SNR, peak, pixel count, shape). Real detections are brighter and larger *on average*, but the distributions overlap heavily at the faint end. That predicts:
- a feature model will do OK on bright objects and badly on faint ones,
- pixels (a CNN) should help.

The metrics phase confirmed both.

### Normalization
Each night has its own sky level and noise. For each stamp and channel:
```
normalized = (value − median) / (1.4826 × median(|value − median|)),   clipped to [−10, 50]
```
- **Median, not mean:** a star in the stamp barely moves it.
- **MAD × 1.4826** equals the standard deviation for Gaussian noise but ignores outliers ("robust").
- Afterwards, "5" means 5σ above the sky on *any* night, so the model compares shapes, not raw brightness.
- **Clipping** stops one saturated star from dominating the network's arithmetic.

Here normalization is per stamp, so there's nothing to "fit" on training data, and there's no leakage risk. (Contrast with `StandardScaler` in *ML basics*, which learns a mean and std and must only see training data.)

### Augmentation
The sky has no "up". Rotating by 0/90/180/270° and optionally mirroring gives 8 versions of every stamp. Training on random versions:
- teaches the model that orientation doesn't matter,
- acts like more data, which reduces overfitting.

**Never augment validation or test data:** they must measure performance on data as it really is.

### Class balance
- Real data: ~1 real per 33 bogus (3.1% real in validation).
- Training: all 8,863 real + 5 bogus per real → 53,178 stamps, 16.7% real.

Why? With 3% positives, the gradient is dominated by easy bogus examples, and "always bogus" is a comfortable local optimum. The cost: the model's probabilities are inflated (it believes real is 16.7% common). The metrics phase shows how to correct that (base-rate adjustment, calibration).

Validation is **not** balanced, so its precision and false-alarm numbers are honest.

### Versioning by name
```
detections__000618_2019-02-08.parquet       stage__field_night
stamps__000618_2019-02-08.npy
injections__000618_2019-02-08.parquet
search_detections__…                        the no-fakes search run (never overwrites)
```
Together with the configs in git, a file name tells you exactly which settings produced it.

### Memory and disk
- float16 stamps: 2.5 GB for 11 nights on disk.
- Training set in RAM: 53k × 3 × 31 × 31 × 4 bytes = 0.61 GB (float32).
- Validation (all detections): 67k stamps, 0.8 GB.

`np.load(..., mmap_mode="r")` lets the notebook peek at a file without loading all of it.

---

## Steps, in order
1. Wrote `build_night_detections` (+ an end-to-end test on 2 fake frames → found and fixed the pandas→WCS bug).
2. Wrote `normalize_stamps`, `augment_stamps`, `balance_classes`, `assemble_stamp_dataset` (+ tests).
3. Fixed the seed (`hash()` → date digits) before the batch build started.
4. Built all 11 nights in the background after the download finished (~8 min).
5. Scanned the stamps → found `inf` values → clipped in code + patched the saved files identically.
6. Notebook: per-night summary, leakage check, EDA, normalization before/after, augmentation gallery, assembled sizes, files on disk.
7. Added a `search_` prefix for no-fakes runs, and made `plan_injections` keep its columns when there are zero fakes (found while wiring up the pipeline).

## Commands
```bash
pytest tests/dataset
jupyter lab notebooks/dataset_building.ipynb    # builds missing nights, then explores
```

## Pitfalls
- **Seeding with `hash(string)`:** Python randomizes string hashes per process.
- **float16 overflow:** clip to ±65,000 first.
- **Balancing validation data:** inflates precision, and you'd ship a model that drowns humans in false alarms.
- **Augmenting validation:** the evaluation no longer matches reality.
- **Rebuilding by accident:** cached files are re-used. Delete them deliberately when the build code changes.

## Quiz
1. Why do we save stamps for *every* detection instead of only the training sample?
2. Why is the MAD multiplied by 1.4826?
3. Training has 16.7% real, validation 3.1%. Which one gives honest precision numbers, and why?
4. Why does per-stamp normalization carry no leakage risk, while `StandardScaler` does?
5. How did an `inf` get into a stamp, and why might it not have broken training anyway?
