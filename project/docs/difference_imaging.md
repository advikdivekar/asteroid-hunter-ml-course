# Phase: Difference imaging and detection

## Goal
Find candidate moving objects: detect every positive blob in the difference images, cut a 3-panel stamp around each one, and get to know the kinds of bogus detections the classifier will have to reject.

**Files built:**
- `src/asthunt/clean/load_exposure.py`: loads science, difference, mask and PSF for one exposure, builds the bad-pixel mask, and derives the reference as science − difference
- `src/asthunt/detect/extract_sources.py`: positive blobs in a difference image → table with x, y, flux, SNR, shape
- `src/asthunt/detect/cut_triplet_stamps.py`: 31×31 stamps of science / reference / difference around each detection
- `configs/detect.yaml`: threshold 2σ, minimum area 5 px, stamp size 31
- tests for all three
- `notebooks/difference_imaging.ipynb` (uses a **training** night only)

---

## How I thought about it
1. **Read the files before writing code.** I opened one difference image and checked four things:
   - it's on the same pixel grid as the science image (same size, same WCS),
   - it has the same zeropoint,
   - about 6–12% of it is NaN (areas that weren't subtracted),
   - its PSF model is a 25×25 stamp that sums to 1.
   These facts decided the design.
2. **Avoid work ZTF already did.** ZTF resamples and blurs the reference to match each science image before subtracting. So `reference = science − difference` gives us a matched reference on the right grid for free. No reprojection code needed.
3. **Measure, then choose the threshold.** I tried thresholds from 1.5σ to 5σ and found the counterintuitive merging effect. 2σ is a starting point; the injection phase will measure what it actually recovers.
4. **Look at the stamps.** A table of numbers won't tell you what the bogus detections are. Twelve pictures did.
5. **Never peek at the test night.** Everything in this phase runs on `2020-09-19` (training).

---

## Concepts

### Difference imaging
```
difference = science − (reference, resampled to the science grid and blurred to the same PSF)
```
Anything constant cancels: stars, galaxies, the sky. What remains:
- **new or moving things**: asteroids, supernovae, flaring stars → **positive** blobs
- **things that faded or moved away**: a variable star getting dimmer, an asteroid that was in the reference → **negative** blobs
- **imperfect cancellation**: bright stars rarely subtract perfectly, leaving **dipoles** (a positive and a negative half side by side)

On our training frame, at 5σ: **1,950** sources in the science image, **334** in the difference image. The search space shrank about 6×, and what's left is mostly interesting or artifacts.

We keep only positive blobs: an asteroid is in tonight's image, not in the reference stack.

### Detection with SEP
`extract_sources` does:
1. NaN and bad pixels → set to 0 and masked.
2. Background and noise maps (as in the primer).
3. **Matched filtering:** SEP smooths the image with a small kernel shaped like a star. Noise is random from pixel to pixel and averages down; a star-shaped signal doesn't. Faint sources stand out more.
4. **Thresholding + connected pixels:** a detection is ≥ 5 touching pixels, each ≥ 2σ above the noise.
5. **Aperture photometry:** sum the flux inside a 3-px circle → `flux`, `flux_error`, and `snr = flux / flux_error`.

### The threshold surprise
| Threshold | Detections | Median size |
|---|---|---|
| 1.5σ | 6,141 | 11 px |
| 2.0σ | 7,764 | 6 px |
| 3.0σ | 731 | 8 px |
| 5.0σ | 334 | 9 px |

At 1.5σ, neighbouring noise peaks touch and **merge** into large blobs, so you get *fewer* detections, and a faint asteroid next to a noise blob gets swallowed. Rules of thumb are good; measuring is better.

### Detections are mostly noise
At 2σ we get about 7,800 detections per frame, but only 173 have SNR > 5. With pure noise, SNR is symmetric around 0, and many 2σ blips pass the threshold by chance. That's fine: **detection is tuned for completeness (don't miss real things), and classification is tuned for purity (throw out junk).** This two-step design (a generous detector, then a strict classifier) is used across astronomy and much of ML (e.g. candidate generation + ranking in recommender systems).

### Triplet stamps
For each detection we cut a 31×31 square from all three images. Why three?
| What it is | science | reference | difference |
|---|---|---|---|
| Real new source (asteroid) | star-shaped blob | nothing | star-shaped blob |
| Bad subtraction | bright star | bright star | dipole / ring |
| Cosmic ray | sharp spike | nothing | sharp spike (narrower than a star) |
| Noise | nothing | nothing | faint speck |
| Edge of masked area | stripe / step | stripe / step | sharp edge |

A person tells these apart by comparing panels. The CNN will learn to do the same.

**Why 31×31?** The PSF is about 2 px wide; 31 px gives 15 px of surroundings on each side, enough to see a nearby bright star or an edge, while staying small (fast training). ZTF's own classifier uses 63×63.

### The combinatorics problem
5 frames × ~8,000 detections. Trying one detection per frame in every combination would be 8,000⁵ ≈ 3 × 10¹⁹ tracklets. Impossible. Two fixes, built in later phases:
1. **The classifier** removes most noise detections first.
2. **The linker** only tries combinations consistent with straight-line motion at asteroid speeds.

---

## Steps, in order
1. Downloaded one test-night exposure into the scratchpad (headers only, to check formats) and confirmed: same grid, same MAGZP, NaN regions, PSF sums to 1, `SEEING` in pixels.
2. Wrote `load_exposure` (+ a test that writes 4 fake FITS products to a temp folder).
3. Wrote `extract_sources` (+ test: finds a bright positive blob, ignores a negative one, survives NaN).
4. Wrote `cut_triplet_stamps` (+ tests: centring, zero padding at edges).
5. Ran detection on a training frame at six thresholds and chose 2σ.
6. Notebook: science vs difference counts, threshold table, SNR histogram, random and brightest stamps, per-frame summary for the night.

## Commands
```bash
pytest tests/detect tests/clean
jupyter lab notebooks/difference_imaging.ipynb
```

## Pitfalls
- **Treating NaN as 0 without masking.** The edge of a NaN region then looks like a sharp step, and becomes a "detection".
- **Pixel coordinates.** SEP and numpy use 0-based indices: pixel (0, 0) is the first pixel. FITS/WCS conventions often use 1-based. Mixing them shifts everything by 1 px.
- **A threshold that's too low merges sources; one that's too high misses faint asteroids.** Check with injections, not intuition.
- **Looking at the test night while tuning.** Every choice made after looking at test data leaks information into the final score.

## Quiz
1. Why do we keep only positive blobs in the difference image?
2. Explain why 1.5σ gave fewer detections than 2σ.
3. Most of our 7,800 detections are noise. Why is that acceptable?
4. Describe what a bad subtraction looks like in each of the three stamp panels.
5. How does `reference = science − difference` avoid reprojection work?
