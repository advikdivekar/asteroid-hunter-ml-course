# Phase: Injection and labelling

## Goal
Create training labels. ML needs examples with known answers, and the sky doesn't come labelled. So we:
1. **plant fake asteroids** where we know the truth (perfect labels),
2. **look up real known asteroids** (real labels, fewer of them),
3. call **every other detection bogus** (and accept a little label noise).

**Files built:**
- `simulate/render_point_source.py`: draws a point source from ZTF's own PSF model, with a sub-pixel shift
- `simulate/plan_injections.py`: random magnitude, speed, direction and start → straight-line positions at the real frame times (on frame 0's grid; converted to each frame's pixels via the sky when planted)
- `simulate/inject_point_source.py`: adds a Poisson-noisy source to the science and difference images
- `label/match_detections_to_positions.py`: nearest-neighbour matching with a KD-tree
- `label/query_known_asteroids.py`: SkyBoT lookup from Palomar (code I41), cached to disk, with a fix for a server bug
- `label/label_detections.py`: combines everything into `label` (1 real, 0 bogus)
- `configs/inject.yaml`, tests for all of the above
- `notebooks/injection_labelling.ipynb`

---

## How I thought about it
1. **Labels come from simulation.** There's no list of "all asteroids in this image". There *is* a list of things we put there ourselves. That's called **injection–recovery**, the standard method in survey astronomy.
2. **The fakes should match the real data, not textbook shapes.** The legacy code used a generic Moffat profile. ZTF ships the actual PSF of each difference image, so I used that: fakes then have the right blur, on the right night.
3. **Add fakes to both science and difference.** A real asteroid is in tonight's image but not in the reference. Adding the same stamp to both keeps `reference = science − difference` unchanged, which is exactly that situation.
4. **Use the real clock.** Fake positions follow the real frame times (0, 6.7, 13.5, 20.2, 26.9 min), not the legacy code's made-up 15-minute steps.
5. **Check real labels against coincidence.** The first "known asteroid" matches looked suspicious (low SNR, a different object in each frame). I estimated how many matches pure chance would produce, and it matched what we saw.
6. **When a service misbehaves, find the pattern.** SkyBoT failed for exactly one frame, every time. Shifting the query time by one second fixed it. That's a server bug we can't fix, so we work around it and write down why.

---

## Concepts

### Injection–recovery
Plant N synthetic objects with known properties → run the *normal* pipeline → count what came out.
- It gives **labels** for training (positives).
- It gives **completeness** (the fraction recovered) as a function of brightness, speed or position.
- Its weakness: the fakes are only as realistic as your simulation. Any difference between fakes and real asteroids is a **distribution shift** the model may exploit. That's why we use the real PSF and real noise, and why the known asteroids matter as a check.

### Sub-pixel rendering
A source centred at x = 120.37 is 0.37 px off a pixel centre. `render_point_source` slides the PSF by that fraction with cubic interpolation, clips tiny negative values, and rescales so the total flux is exactly right. Without this, every fake would sit perfectly on a pixel centre, and a model could learn "perfectly centred = fake".

### Photon noise for fakes
A real source giving N electrons fluctuates by √N. We draw `Poisson(stamp × gain) / gain`. The background noise is already in the image; we only add the source's own noise.

### Magnitudes drawn uniformly
Real asteroids are mostly faint (there are many more small ones). We still draw magnitudes **uniformly** from 17 to 22, so every brightness bin has enough fakes to measure completeness. Remember this when quoting a *rate* of real detections: our training set isn't the sky's mix.

### Matching with a KD-tree
For each detection, find the nearest fake. Comparing every detection with every fake costs 7,700 × 300 operations per frame. A **KD-tree** sorts the points in space, so each lookup takes about log(n) steps. Within 2 px = match.

### Completeness curve (measured on a training night)
- ~90% for bright fakes. The remaining ~10% land on masked pixels (11.9% of this image is masked).
- It falls through 50% near the header's `MAGLIM` (≈ 20.8) and reaches about 20% by mag 22.
- **Speed:** the bins range from 0.61 to 0.81, but each holds ~30 asteroids, so the uncertainty is about ±0.08. That's mostly noise. Physics agrees: at 100 "/hr an asteroid moves 0.8" during a 30 s exposure, less than a pixel.

**Uncertainty of a fraction:** p measured from n independent objects has a standard error of √(p(1−p)/n). With p = 0.7 and n = 30: ±0.084.

### Known asteroids and chance coincidences
SkyBoT listed ~110 known asteroids in the field, nearly all fainter than our limit. We still "matched" 1–2 per frame. The check:
```
P(random point lands within 2 px of some detection) = detections × π × 2² / good pixel area = 1.15%
expected chance matches = 114 known × 1.15% ≈ 1.3 per frame
```
That's what we saw. So most of those labels were wrong: **label noise**. Fix: only use known asteroids brighter than V = 20.5 (`known_asteroid_max_v_mag`). The same effect gives mag-22 fakes a few percent of fake "completeness".

**General lesson:** whenever you label by position matching, estimate the chance-match rate.

### Observer location matters
SkyBoT defaults to Earth's centre. ZTF sits on Earth's surface, about 6,400 km away. For an asteroid 1 AU away, that shifts its apparent position by up to ~9", more than 4 of our pixels. We pass `location="I41"` (ZTF's observatory code).

### Label noise in "bogus"
Bogus = everything not matched. That includes the occasional undiscovered asteroid, variable stars and other real transients. On the training night this is a tiny fraction of 36,000 detections, so the model barely notices. But it means **a "false positive" on real data may not be false at all**. That's exactly what we're hunting for.

### Class imbalance
One training night: **1,059 real vs 36,161 bogus (34 to 1)**, even with 300 fakes. On the real sky without fakes, it's a handful of real objects among ~40,000 detections.

### Update, found later (in the Real/bogus CNN phase)
The CNN gave its highest scores to some detections labelled "bogus". Looking at them: clean new point sources, and **all of them were known asteroids**, some as bright as V = 17. Our 2-px radius had missed them.
- Measured on all nights: SkyBoT's positions sit **~2.9" from our detections, in a constant direction** (scatter only 0.1–0.5 px).
- Our WCS agrees with Gaia stars to **0.04 px**, and **JPL Horizons** agrees with our detections to **0.1–0.3"**. So the offset is in SkyBoT's predictions, not in our pipeline.
- Fix: `known_match_radius_pixels: 4.0` for known asteroids (fakes keep 2 px). Re-labelling added 2–70 real labels per night; the test night went from 12 to 82 known-asteroid detections.

**Lessons:** (1) chance coincidences explained *some* matches, but a systematic error hid behind them. Always check the *direction* of offsets, not just their size. (2) A model's most confident "mistakes" are the best place to look for label errors.

### Update 2, found in the Full run phase: fakes must move straight on the SKY
On the test night, real known asteroids failed to link while fakes linked perfectly. Every real asteroid showed the *same* residual pattern (0.6, 2.5, 2.4, 0.6, 1.7 px), which means a per-frame error, not orbits.
- Cause: **the telescope pointing shifts between exposures** (up to 12 px on the test night, 4–12 px on others). We planted fakes moving straight in *pixel* coordinates, so they ignored the shifts. That's an unrealistic shortcut, a distribution shift between fakes and real asteroids.
- Fix (in `build_night_detections`): plan fakes on frame 0's grid, then convert each planned position via the sky (frame-0 WCS → RA/Dec → frame-k WCS) to that frame's pixels. Every detection also gets `x_common, y_common` on frame 0's grid, and linking uses those.
- Check: real asteroids' straight-line residuals dropped from ~2.5 px to **0.05–0.47 px**.

**Lesson:** test your simulation against real objects, not just against itself. Our fakes passed every test because the tests shared the same blind spot.

---

## Steps, in order
1. Inspected ZTF's PSF model (25×25, sums to 1) and decided to use it instead of a Moffat.
2. Wrote `render_point_source` (+ tests: exact flux, centre moves with the offset).
3. Wrote `plan_injections` (+ test: straight lines at the requested speed using real times).
4. Wrote `inject_point_source` (+ tests: correct flux in both images; edge sources skipped).
5. Wrote `match_detections_to_positions` and `label_detections` (+ tests).
6. Wrote `query_known_asteroids`. Then fixed, in order:
   - the default geocentric location → I41,
   - a units error in the magnitude column,
   - the variable `time` hiding the `time` module,
   - SkyBoT's one-timestamp server bug → nudge the time by +1 s and retry,
   - repeated queries → cache to CSV.
7. Measured completeness vs threshold on one frame (2σ beats 3σ at the faint end).
8. Notebook: PSF, plan, inject, detect, label; completeness vs magnitude and speed (with error bars); known-asteroid chance-match analysis; class counts.
9. Added `known_asteroid_max_v_mag` to the config.

## Commands
```bash
pytest tests/simulate tests/label
jupyter lab notebooks/injection_labelling.ipynb
```

## Pitfalls
- **Injecting into the original arrays.** `inject_point_source` changes images in place. Copy first.
- **Fakes too close together.** With thousands per image they overlap and steal each other's matches. 300 per night keeps them apart.
- **Match radius too big.** More fakes get "found", but by noise. Completeness looks better and is wrong.
- **Trusting position-matched labels without a chance estimate.**
- **Rows vs columns.** numpy indexes `image[row, column]` = `image[y, x]`. The tests check this explicitly.

## Quiz
1. Why do we add the fake to both the science and the difference image?
2. Why are fake magnitudes drawn uniformly instead of like real asteroids?
3. 7,600 detections, a 2-px match radius, 8.3 million good pixels: what's the chance a random point gets matched?
4. A completeness bin has p = 0.6 from 25 asteroids. What's its uncertainty? Is 0.6 vs 0.75 in another bin meaningful?
5. Why did we pass `location="I41"` to SkyBoT?
6. Give two kinds of real objects that end up labelled "bogus".
