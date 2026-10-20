# Phase: Deep search (round 2)

## Goal
Find asteroids **too faint to see in any single frame**, the only realistic place for an undiscovered asteroid in ZTF data (ZTF's own pipeline has mined everything its single frames can see). Then prove which detections are real.

**Files built (`src/asthunt/deep/` unless noted):**
- `build_snr_image.py`: matched-filter SNR map (pure noise → N(0, 1))
- `compute_frame_offsets.py`: pointing shift of every frame vs the reference
- `group_frames_into_stacks.py`: consecutive groups of ~8 frames
- `standardize_over_night.py`: per-pixel (value − night median) / night scatter
- `mask_bright_pixels.py`: hide what single frames already see
- `make_velocity_grid.py`, `stack_and_find_peaks.py`: GPU shift-and-stack over 4,888 trial motions, best-SNR image, peaks
- `merge_peaks.py`, `measure_velocity_consistency.py`, `cut_stacked_stamps.py`
- `pipeline/run_deep_search.py` (with a **null test** option), `crossmatch/crossmatch_deep_tracklets.py`
- `simulate/plant_fakes_in_exposure.py` (shared with the regular build)
- `link/link_detections_into_tracklets.py`: seed search vectorised
- `configs/deep.yaml`, tests for each, `notebooks/deep_search.ipynb`

---

## The idea
Add up 8 consecutive frames along the motion of a hypothetical asteroid. Its signal adds 8×; noise adds only √8 ≈ 2.8×, a gain of +1.1 mag. We don't know the motion, so we try every one (every direction, up to 120 "/hr, in steps of 3 px/h). Then we **link** the stack detections across the 4.4-hour night (7 stacks): a real object must appear in several independent stacks along one straight line.

---

## The journey: every number below was measured on the 55-frame night (2019-02-08)
| step | peaks per stack | what it taught |
|---|---|---|
| first version, threshold 5σ | 47,550 | the threshold must account for the **number of trials** |
| pure-noise calibration | 6,179 at 5σ → **44 at 6σ** | picking the best of 4,888 motions = look-elsewhere effect; 6σ set from data |
| real data at 6σ | still ~47,000 | real difference images are **not** Gaussian noise |
| + clip each frame at 5σ | 28,000 (on 1/8 of motions) | one bright frame can't dominate, but that wasn't the main problem |
| + static-sky subtraction (night median) | 5,765 | extended static residuals (halos, galaxies) stack at *any* motion inside them |
| + per-pixel standardisation (median **and scatter**) | 391 | residuals that **wobble** with the seeing are turned down too |
| + mask pixels single frames already see | **94** | bright asteroids created **echo** tracklets |

Then the tracklets:
| check | result |
|---|---|
| echo test (before masking) | 151 of 242 unknown tracklets lay within 30 px of a known asteroid (random: ~190 px): ghosts |
| **null test** (frame times shuffled) | 16 false tracklets per night |
| **velocity consistency** (each stack's own best motion vs the night-long motion) | known asteroids 2.4 px/h, null tracklets 61 px/h → cut at 6 px/h |
| after the cut | known 20/20 kept, **null 0**, unknown **6** |

Speed, too: the first linker looped over seed pairs in Python and never finished (CPU at 100% for 20+ minutes). `sparse_distance_matrix` returns all pairs at once: 4,000 peaks × 6 stacks in 26 s.

---

## Depth (300 injected fakes, mag 20–23, same night)
| mag | ≤ 20.9 | 21.1 | 21.4–21.9 | 22.1 | ≥ 22.4 |
|---|---|---|---|---|---|
| completeness | 0% (masked on purpose) | 55% | **94–100%** | 48% | 0% |

- **Deep search: 50% at mag 22.12.** The single-frame v4 pipeline: 21.41 on the same night. The two searches together cover everything to ~22.1, **+0.7 mag**.
- Calibration from 100 found fakes: `mag = 26.25 − 5.23 · log10(mean stack SNR)`, scatter 0.17 mag.

---

## The candidates
6 unknown, velocity-consistent tracklets. All move like main-belt asteroids near opposition (33–40 "/hr, westward):

| # | stacks | span | rate ("/hr) | est. r mag |
|---|---|---|---|---|
| 7 | 7 of 7 | 3.5 h | 40.4 | 21.4 |
| 14 | 6 | 2.9 h | 39.0 | 21.6 |
| 16 | 5 | 3.5 h | 34.8 | 21.9 |
| 18 | 5 | 2.9 h | 38.0 | 21.9 |
| 26 | 4 | 3.5 h | 36.1 | 21.9 |
| 47 | 3 | 1.2 h | 33.5 | 21.9 |

Vetting so far:
- **Stacked stamps:** a compact point source centred in stack after stack, like the known reference asteroids of the same brightness.
- **SkyBoT within 5' (today's catalogue: it lists objects discovered in 2026):** no known object with a large enough ephemeris uncertainty to be one of them. Nearest known objects are 77–288" away with uncertainties ≤ 0.9". Similar motion is expected (all main-belt asteroids near opposition move ~−35 "/hr in RA).
- **Brightness:** r ≈ 21.4–21.9, where ZTF's single frames are weak. Being uncatalogued is plausible.

**What they are not yet:** discoveries. The MPC needs a **second night**. ZTF has no other deep night of this field close in time (one g-band frame 4 days later, far too shallow). An honest status is "**6 single-night candidates, 3.5-hour arcs, passing every test we can apply with this data**". Confirming them needs either another night of stacked ZTF data in the right place, or a follow-up observation.

## Concepts
- **Look-elsewhere effect:** the more trials, the more chance flukes. Set thresholds from simulations with the same number of trials.
- **Null tests:** break the property you're searching for (here, time order) and count what survives. That's your false-alarm rate, measured on real data.
- **Robust statistics:** median and MAD per pixel over time separate "always there" and "always wobbling" from "here once".
- **Echoes and masking:** a strong source leaks into neighbouring hypotheses; remove it from the search it doesn't need.
- **Independent consistency:** each stack measures velocity on its own; agreement across stacks is evidence a fluke rarely fakes.
- **Calibrate with injections:** turn a detection statistic into a physical quantity (magnitude) using fakes passed through the identical pipeline.

## Quiz
1. Why did 5σ produce thousands of peaks on pure noise?
2. Why does the per-pixel *scatter*, not just the median, matter?
3. What's an echo tracklet, and why does masking bright pixels remove it?
4. Explain the null test in two sentences. Why is it better than estimating false alarms from theory?
5. Why are the 6 candidates not yet discoveries? What exactly would confirm one?
