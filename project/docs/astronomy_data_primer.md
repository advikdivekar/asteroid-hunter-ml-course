# Phase: Astronomy data primer

## Goal
Understand what's inside a ZTF image well enough to clean it, and build the three cleaning files every later stage uses.

**Files built:**
- `src/asthunt/clean/read_fits_image.py`: pixels + header from plain or compressed FITS
- `src/asthunt/clean/build_bad_pixel_mask.py`: True where a pixel can't be trusted
- `src/asthunt/clean/estimate_background.py`: smooth sky map + noise map
- tests for all three (they run on tiny fake images, so no real data is needed)
- `notebooks/astronomy_data_primer.ipynb`

---

## How I thought about it
1. **ML is only as good as its inputs.** Before detecting or classifying anything, we need to know: what is a pixel value, how noisy is it, which pixels are broken, and how bright is "mag 19" in counts?
2. **Check the physics against the data.** For every formula I compared the theory with a measurement on a real image. If they disagree, either the formula or our understanding is wrong, and both are worth knowing.
3. **Read the header before writing code.** I printed the real header first (`SATURATE`, `GAIN`, `MAGZP`, `SEEING`, ...) so the code uses what ZTF actually provides.
4. **Don't trust what I can't verify.** I couldn't find the official ZTF mask-bit table, so `build_bad_pixel_mask` takes the bits as an *argument*. We'll read their meanings from the mask file's own header once we download masks.
5. **When a measurement disagrees, investigate.** My first FWHM measurement was 65% off. Finding out why taught more than getting it right first time.

---

## Concepts

### FITS files
A FITS file = one or more **extensions**, each with a **header** (text key = value pairs) and optionally **data** (the pixel array).
- Plain ZTF images: pixels in extension 0.
- Compressed `.fits.fz` (ZTF difference images): empty extension 0, pixels in extension 1.

`read_fits_image` returns the first extension with data, so callers don't need to care which kind of file it is.

### Counts, gain and noise
The camera counts **electrons** freed by photons and reports **counts** (DN). `GAIN` = electrons per count (6.2 for ZTF).

Photon arrival is random (**Poisson**): if you expect N electrons, the scatter is √N. The camera adds **read noise** (10 electrons). So for one sky pixel:
```
noise (counts) = sqrt(sky_counts × GAIN + READNOI²) / GAIN
```
*Our image:* sky = 458 counts → predicted 8.75, measured **9.25** counts. The 6% extra comes from flat-fielding, resampling and faint undetected stars. That's why we measure the noise map rather than trust the formula.

### Background
The sky isn't flat: moonlight and scattered light vary across the chip. `estimate_background` (via SEP):
1. cuts the image into 64×64-pixel boxes,
2. finds the typical sky value in each box, ignoring stars,
3. interpolates a smooth **background map**,
4. measures the scatter → **noise map** ("1 sigma" at each place).

We subtract the background, then compare signals to the noise. "5σ" means 5 × noise above the sky.

### Bad pixels
- **NaN:** no data.
- **Saturated:** the pixel hit its limit (`SATURATE` ≈ 48,728 counts). The true value is unknown, and charge bleeds into neighbours, so we also mark a 2-px border.
- **ZTF mask bits:** ZTF ships a mask image where each pixel's integer stores several yes/no flags as **bits**. Example: value 17 = binary `10001` = bits 0 and 4 set. To test bit 4: `(value & 16) != 0`.

Why it matters for ML: a saturated star's leftovers in a difference image look like a bright blob. If we don't mask it, the bogus class fills with junk the model shouldn't have to learn.

### Magnitudes and the zeropoint
```
mag = MAGZP − 2.5 × log10(total counts)
```
- Bigger magnitude = fainter. A difference of 5 magnitudes = 100× in brightness.
- `MAGZP` (26.30 here) = the magnitude that would give 1 count. It changes from night to night with clouds and airmass, so always use the image's own value.

*Worked example:* mag 19 → 10^((26.30 − 19) / 2.5) = **832 counts**, spread over many pixels.

### Signal-to-noise ratio (SNR)
For a source in an aperture of N pixels:
```
SNR = signal / sqrt(signal / GAIN + N × noise_per_pixel²)
```
On our image: mag 19 → SNR 22, mag 20 → 9.1, mag 20.5 → 5.8, mag 21 → 3.7. The header's `MAGLIM` = 20.79, which is where SNR ≈ 5.

**This defines our ML target:** real asteroids at SNR 3–6, where one detection alone is unconvincing but four detections in a straight line could be.

### Seeing and the PSF
The atmosphere blurs every point source into the same blob shape: the **point spread function (PSF)**. Its **FWHM** (full width at half maximum) is the header's `SEEING`: **2.18 pixels** (= 2.21"). The header comment says `[pixels]`; I first misread it as arcsec. With 1 px ≈ 1" the difference is small, but units matter: always read the header *comment*, not only the value.
- An asteroid looks like a star (the same PSF), unless it moves fast enough to **trail** during the 30 s exposure.
- Cosmic rays are *sharper* than the PSF, and bad subtractions are *weirder* than it. That difference is what the CNN learns.

**The measurement lesson:**
- My first try used SEP's second moments (`a`, `b`) and got **3.56 px**.
- Second moments weigh each pixel by distance squared, so the faint wings of a real PSF inflate them.
- Counting the pixels above half the peak (the definition of FWHM) gave **2.26 px**, close to the header's 2.18 px.

### WCS: pixels ↔ sky
The **World Coordinate System** keywords (`CTYPE`, `CD1_1`, ...) map pixel (x, y) to (RA, Dec). Catalogues (SkyBoT) and reports (MPC) use RA/Dec. Check: 100 pixels = 101.2" = 100 × `PIXSCALE` (1.012).

### Time
`OBSJD` = Julian Date (a running day count). `Time(jd, format="jd").iso` turns it into a calendar date. Motion is measured over time, so exact timestamps are essential. The legacy code's made-up 15-minute steps are gone from here on.

---

## Steps, in order
1. Printed the real header and pixel statistics to see what ZTF provides.
2. Searched for the ZTF mask-bit definitions (not found) → made `bad_bits` an argument.
3. Wrote `read_fits_image` (handles plain + compressed) + tests on generated FITS files.
4. Wrote `build_bad_pixel_mask` (NaN, saturated + border, optional ZTF bits) + tests.
5. Wrote `estimate_background` (SEP wrapper) + test (flat 500 ± 10 sky → recovers 500 and 10).
6. Notebook: header tour, pixel histogram, mask, background vs noise theory, magnitudes, SNR vs MAGLIM, PSF/FWHM, WCS.
7. Found and fixed the FWHM measurement (second moments → half-maximum area).

## Commands
```bash
pytest tests/clean
jupyter lab notebooks/astronomy_data_primer.ipynb
```

## Pitfalls
- **Integer pixels.** Adding a fake asteroid to an int16 image overflows. `read_fits_image` always returns float64.
- **Using one image's MAGZP for another.** It changes nightly.
- **Detecting before masking.** Saturated stars produce dozens of fake "detections".
- **SEP memory layout.** SEP needs C-contiguous float32 arrays; `estimate_background` converts for you.
- **Trusting a formula you didn't check.** Our noise and FWHM checks both turned up surprises.

## Quiz
1. A pixel's sky level is 900 counts with GAIN 6.2 and read noise 10 e⁻. What's its predicted noise in counts?
2. How many counts does a mag-20.5 source give with MAGZP = 26.3? About what SNR on our image?
3. Why does a saturated star need a border in the mask, not just its saturated pixels?
4. Why did second moments overestimate the FWHM?
5. Mask value 20: which bits are set? Is it bad if `bad_bits = (4,)`?
