# Phase: Fast movers (round 2)

## Goal
Detect objects so fast they **trail** during one 30-second exposure, mostly near-Earth objects (NEOs). The point-source pipeline is built for round blobs and stops at 120 "/hr; a trail at 1,000 "/hr is ~8 px long and spreads its light, so a round matched filter under-counts it.

**Files built:**
- `simulate/render_trailed_source.py`: the PSF smeared along a line (exact total flux)
- `simulate/inject_trailed_source.py`: plant trailed fakes, same rules as point fakes
- `fast/build_streak_kernels.py`: trailed PSF kernels at several lengths × angles (0–180°)
- `fast/find_streaks.py`: best trailed-filter SNR per pixel (one Fourier transform reused for all kernels); a detection is a streak only if a trailed kernel beats the round PSF
- `fast/check_streak_consistency.py`: motion direction vs streak angle, speed × 30 s vs streak length, one trail length across frames
- `pipeline/run_fast_search.py`: scale → **standardise over the frames** → find streaks → common grid → link → consistency
- `configs/fast.yaml`, tests (render, finder, consistency, end-to-end on 3 synthetic frames)

---

## How I thought about it
1. **Physics first.** Trail length ≈ rate / 120 px (30 s, 1.012 "/px): 500 "/hr → 4 px, 3,600 "/hr → 30 px. That sets the kernel lengths.
2. **The strongest filter is physical consistency.** A real fast object's trail must point along its motion between frames, and its length must equal the distance moved during an exposure. Satellites, cosmic rays and artifacts almost never satisfy that across ≥ 3 frames.
3. **Measure on real data before trusting a synthetic test.** The synthetic test passed instantly; the first real night produced **3,770 "streaks" per frame**.
4. **Choose cuts from saved data, not by trial and error.** I saved every candidate track (real night = false alarms, injected night = finds) and compared cut choices offline.

---

## Before → After
| step | streak detections per frame (real night) | false fast tracks per night | injected trails recovered |
|---|---|---|---|
| raw trailed filter, threshold 7σ | 3,770 | (linking didn't finish) | — |
| + per-pixel standardisation over the frames | ~540 | 40 | 81% (61/75 that stay on the chip) |
| + data-chosen consistency cuts, lengths ≤ 32 px | ~540 | **0** (both test nights) | **97%** (141/146 linked fakes) |

Completeness by brightness before the final cuts: 92% at mag 15–16 down to 58% at 19–20; by speed 88–92% up to 2,000 "/hr, 55% at 3,500–5,000 "/hr.

## What went wrong, and what each fix taught
- **Extended residuals look like long streaks.** 78% of detections were 64-px "trails": long kernels integrate over star halos and detector-aligned structure (angles cluster near the pixel axes, not along one line, so they aren't satellites). → Standardising each pixel over the night removes static structure (a fast object is never on the same pixel twice, so it's untouched), and 64-px kernels were dropped (≈ 7,700 "/hr objects are vanishingly rare).
- **Linking was the slow part, again.** With 3,770 detections per frame and a search radius covering the whole chip, the old Python seed loop never finished. Vectorising it (KD-tree `sparse_distance_matrix`) made it seconds.
- **Cut values from data.** Fakes: angle mismatch median 3.8°, length spread 0; false tracks: 44.6°, 0.65. Chosen: angle ≤ 15°, length within 1.6× of expected, length spread ≤ 0.5.

## Honest limits
- The cuts were chosen on the same 2 nights they were measured on, so they need confirming on other nights.
- Speeds between 120 and ~480 "/hr fall between the two searches: too fast for the point linker's limit, trails too short (1–4 px) to look elongated. Raising the point linker's `max_rate` closes the gap at the cost of more chance alignments.
- No NEO was found. ZTF's own streak pipeline (DeepStreaks) has already mined these data for bright fast movers.

## Concepts
- **Matched filter for a known shape:** the optimal detector for a trail is a trail-shaped kernel. Brute force over lengths × angles is the simple, exact version.
- **Fourier trick:** convolution = multiplication of transforms. Transform the image once, reuse it for every kernel.
- **Consistency checks across measurements** beat any single-frame classifier: two independent measurements of the same physics (trail shape vs motion) must agree.

## Quiz
1. How long is the trail of a 2,400 "/hr object in a 30 s ZTF exposure?
2. Why does standardising over the night never remove a fast mover?
3. Why are angles compared modulo 180°?
4. Name two reasons a satellite streak fails the consistency check.
5. What's the risk of choosing cut values on the same nights you report results on?

## Update (2026-10-14): a 3-frame night
The two-night hunt ran the fast search on 2019-01-20, which has only **3 frames**. Result: 16,612 streak detections (≈ 5,500 per frame, against ≈ 540 on 5-frame nights) and **5 "consistent" false tracks**, scattered across the whole chip with mixed 16/32 px lengths.

**Why:** standardising over the night needs a decent per-pixel median and scatter. From 3 values both are noisy, so static structure survives. Chance alignments grow roughly with the cube of the detection count.

**Fix:** `min_frames: 5` in `configs/fast.yaml`; `run_full_search` skips the fast search on shorter nights and says so. I did not re-tune the cuts on this night: that would only hide the cause.

## Update (2026-10-14): 10-frame nights and the image edge
On 10-frame hunt nights the fast search (then using only the first 5 frames) reported 7 and 3 "consistent" tracks. Two causes:
1. **Five frames give a noisy night standardisation.** With all 10 frames, 618/c02/q3 on 2019-01-23 went from 7 false tracks to 1. → `max_frames: 10` in `configs/fast.yaml`.
2. **The last one slid along the image's bottom edge** (y ≈ −6, angle 0°): a long kernel lying along the border responds to the border itself. → `find_streaks` drops peaks closer to the edge than half the largest kernel. The test `test_a_trail_lying_on_the_image_border_is_not_reported` fails on the old code and passes on the new.

**After both fixes (10 frames where available):**
| unit | frames | false fast tracks |
|---|---|---|
| 618/c02/q3 2019-01-23 | 10 | 0 (was 7) |
| 618/c03/q4 2019-01-23 | 10 | 0 |
| 618/c03/q4 2019-02-08 | 10 | 1 (marginal: lengths 8/32/32 px where ~30 is expected, SNR 7–10) |
| 618/c03/q4 2023-09-15 | 5 | 0 |
| 618/c03/q4 2021-09-12 | 5 | 0 (a night not used for choosing the cuts) |

Injected trails on 2019-02-08 with 10 frames: **93% recovered** (66/71), 0 false tracks among the injected run.
