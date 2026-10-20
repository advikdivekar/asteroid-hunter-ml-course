# Phase: Linking

## Goal
Join detections from the 5 frames of a night into **tracklets**: sets of detections that lie on one straight line at a plausible asteroid speed. Measure how many planted asteroids come out as clean tracklets.

**Files built:**
- `link/fit_linear_motion.py` (from the Leakage audit): least-squares straight lines x(t), y(t), plus the max residual
- `link/link_detections_into_tracklets.py`: seed-and-predict linker with KD-trees
- `label/label_tracklets.py`: a tracklet is real only if all its detections are the same fake or the same known asteroid
- `dataset/build_tracklet_dataset.py`: one night: CNN filter → link → label → features → save
- `configs/link.yaml`, tests (moving object among noise, one missing frame, stationary source, jittery stationary source, no shared detections), `notebooks/linking.ipynb`

---

## How I thought about it
1. **Linking is search, not learning.** Physics gives the rule: over 30–45 minutes an asteroid moves in a straight line at constant speed. We only need to search efficiently.
2. **Never try every combination.** 5 frames × ~7,500 detections → 2 × 10¹⁹ combinations. Instead, a pair of detections defines a line, and we *look up* the other frames.
3. **Use the CNN to shrink the search.** Keeping 95% of real detections (threshold chosen on validation) removes 76% of detections, cutting seed pairs from 52k to 3.5k. Linking a night takes 0.2 s.
4. **Test against real objects, not just fakes.** That's how the biggest bug in the project was found (below).

---

## Concepts

### Seed and predict
1. **Seeds:** for each pair of frames, every detection pair whose implied speed is between the minimum and maximum. A KD-tree `query_ball_point` finds all neighbours within `max_rate × Δt` instantly.
2. **Predict:** a seed fixes a velocity, so we know where the object must be in every other frame. A KD-tree `query` finds the nearest detection there (within 2 px).
3. **Keep:** ≥ 4 of 5 frames, straight-line residual ≤ 1.5 px, **fitted** speed within limits.
4. **De-duplicate:** many seeds find the same tracklet. Take tracklets best-first (most detections, smallest residual) and drop any that re-use a detection.

### The big bug: the telescope moves between exposures
On the test night, bright known asteroids weren't linked, yet fakes were (100%). Every real asteroid had the **same** residual pattern (0.6, 2.5, 2.4, 0.6, 1.7 px). Identical residuals across different objects point to a **per-frame** error, not orbits.
- **Cause:** the pointing shifts between exposures, by up to 12 px on the test night and 4–12 px on other nights. I was linking in each frame's own pixel coordinates.
- **Why it hid:** fakes were planted moving straight *in pixel coordinates*, so they ignored the shifts. On most nights the drift was smooth, so it just looked like a bit of extra velocity. On the test night it jumped.
- **Fix:** every detection gets `x_common, y_common` (its RA/Dec projected onto frame 0's pixel grid), and linking uses those. Fakes are now planned on frame 0's grid and converted through the sky into each frame's pixels, so they behave like real asteroids.
- **Check:** real asteroids' residuals dropped from ~2.5 px to **0.05–0.47 px**.

### The minimum speed must beat measurement noise
After the fix, the test night still had 78 "unknown" tracklets with a median speed of 2.5 "/hr: stationary sources (bad subtractions, variable stars) whose centroids wobble.
- Bug 1: the speed limit was only checked on seed pairs, not on the fitted track. Fixed (with a test).
- Bug 2: a 3 "/hr minimum is too low. With ~0.5 px centroid noise over 28 minutes, the fitted speed noise is ~1.4 px/hr per axis, so a stationary source can easily "move" at 3–6 "/hr. **New minimum: 8 "/hr**, still far below main-belt speeds near opposition (the 31 known asteroids we recovered move 28–48 "/hr).
- Cost: fakes planted slower than 8 "/hr (18 of 300 per night) can't be linked, by design.

### Results
- **Linking efficiency** (linked among linkable fakes: kept in ≥ 4 frames, speed in range): **100%** on the training night shown.
- One fake planted at 7.7 "/hr was linked anyway, because its *measured* speed came out just above 8: measurement noise in action.
- **Tracklets per night:** 176–212 on training/validation nights (mostly our 300 fakes), with only **1–8 bogus** per night; 273 on the test night (34 "bogus", of which 14 turned out to be known asteroids our labels missed: some fainter than the V ≤ 20.5 cut, some with a frame where the match failed). Bogus ones are **near misses** (a fake plus a noise detection) or **chance alignments** (noise only), the hard negatives predicted in the Leakage audit, now produced by the real linker.

---

## Steps, in order
1. Wrote the linker + tests (asteroid among 300 noise points/frame; missing frame; stationary; no shared detections).
2. Wrote `label_tracklets` and `build_tracklet_dataset` (+ tests).
3. Chose the CNN threshold for 95% recall on validation (0.029).
4. Linked all 11 nights.
5. In the Full run, real asteroids failed to link → found the pointing shift → common-grid coordinates + sky-based fake planting → rebuilt everything.
6. Found stationary "slow movers" → fitted-speed check + 8 "/hr minimum (+ a jittery-stationary test).
7. Fixed a notebook bug that overwrote `thresholds.json` (it now updates the file).

## Commands
```bash
pytest tests/link tests/dataset
jupyter lab notebooks/linking.ipynb
```

## Pitfalls
- **Linking in per-frame pixel coordinates.** Always use one common sky-based grid.
- **Simulations that share your blind spot.** Fakes passed because they were planted in the same (wrong) coordinates.
- **Speed limits below your measurement noise.**
- **Validating constraints only on seeds.** Check the final fitted track too.
- **Overwriting shared config files from two places.**

## Quiz
1. Why do identical residual patterns across different asteroids point to a per-frame problem?
2. How does the CNN filter make linking ~15× cheaper?
3. Centroid noise is 0.5 px over 30 minutes. Roughly what false speed can a stationary source show?
4. Why must fakes be planted via sky coordinates?
5. What's a "fake + noise" tracklet, and why is it a valuable training example?
