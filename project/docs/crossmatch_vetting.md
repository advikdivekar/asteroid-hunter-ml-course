# Phase: Crossmatch and vetting

## Goal
Run the real search (no fakes) on a validation night, separate known asteroids from unknown tracklets, and learn how to vet a candidate before believing it.

**Files built:**
- `crossmatch/crossmatch_known_asteroids.py`: a tracklet is "known" if ≥ half its detections match the same SkyBoT object (any brightness, 4 px)
- `report/write_candidate_report.py`: HTML page with science / reference / difference stamps per detection
- `report/write_ades_report.py`: a **draft** in the Minor Planet Center's ADES format (mid-exposure times, observatory I41), marked "NOT FOR SUBMISSION"
- `pipeline/run_night_pipeline.py`: detect → CNN → link → features → score → crossmatch → candidates → reports
- tests, `notebooks/crossmatch_vetting.ipynb`

---

## How I thought about it
1. **Known asteroids are free ground truth on real data.** Every tracklet that matches SkyBoT is a real asteroid our pipeline found by itself. That checks the whole system on real objects, not just on our fakes.
2. **Before calling anything new, rule out everything known**, using a looser radius than for labels, because orbit predictions can be off.
3. **Expect bogus survivors.** The look-elsewhere effect says some junk will pass every cut. Estimate how many *before* looking at candidates.
4. **A human makes the final call**, with a written checklist.

---

## Concepts

### Crossmatching tracklets
For every detection, find the nearest SkyBoT prediction in its own frame (within 4 px = 4", covering SkyBoT's ~2.9" systematic offset). If at least half the tracklet's detections match the *same* object, the tracklet is that object. Requiring several matches makes chance coincidences negligible.

### Results on validation night 2023-09-15 (real search, no fakes)
- 31,953 detections → **13 tracklets → 10 known asteroids, 0 unknown candidates**.
- Known asteroids with V ≤ 20 present in ≥ 4 frames: 7, of which **5 recovered**.

### The look-elsewhere effect
We test thousands of seed combinations per night. Even rare accidents happen somewhere. The validation estimate is ~2 bogus tracklets per night above the threshold, from only 9 examples, so "a few". If a night produces about that many unknown candidates, most are probably bogus. That isn't failure; it's how honest statistics look.

### Vetting checklist (for every unknown candidate)
1. **Stamps:** a point source in science *and* difference, *nothing* in reference, in every frame?
2. **Motion:** steady, and plausible? Near opposition, main-belt asteroids move ~25–50 "/hr roughly westward. On the test night the 31 known ones all moved −18° ± 8°.
3. **Brightness:** roughly constant (small `mag_std`)?
4. **Artifacts:** not on a galaxy or bright star, not along a bleed trail, not part of a satellite streak?
5. **Known?** No SkyBoT match, even at a looser radius (we use 15 px as a second check)?
6. **Confirmation:** seen on another night? The MPC needs ≥ 2 nights before an object means anything.

### What vetting found on the test night (full run phase)
The three most interesting unknown tracklets, all below the threshold:
| tracklet | numbers | stamps say | verdict |
|---|---|---|---|
| #45 | 40.7 "/hr, −28°, mag 21.6, steady brightness: *looked* like a main-belt asteroid | sits on an elongated **galaxy** visible in the reference; frame 0 shows nothing | subtraction residual |
| #44 | 72.8 "/hr, −28°, mag 20.4 | long straight **streaks** in science and difference | satellite trails |
| #28 | 8.3 "/hr (right at the limit), SNR 16 | edge of a **saturated star's bleed** | artifact |

The numbers for #45 were convincing; the pictures weren't. **Always look at the stamps.**

### ADES draft
ADES is the MPC's observation format. Our writer:
- uses **mid-exposure** times (OBSJD is shutter-open; +15 s). JPL Horizons confirmed mid-exposure fits our detections better (0.13" vs 0.21" for one asteroid).
- gives each tracklet a temporary designation (`trkSub`),
- writes a header comment: **"DRAFT … NOT FOR SUBMISSION"**.

Real submissions need vetted, multi-night objects, and ZTF already reports its own observations.

---

## Steps, in order
1. Wrote `crossmatch_known_asteroids` (+ test: a tracklet following "Ceres" is named; one elsewhere isn't).
2. Wrote the HTML report and ADES writer (+ tests: page and image exist; header and mid-exposure time correct).
3. Wrote `run_night_pipeline`, which only chains existing functions.
4. Hit a bug where zero fakes produced object-typed empty columns that pandas refused to assign. Wrote a test that reproduces it exactly (it failed before the fix, passes after), then fixed `label_detections`.
5. Ran the search on validation night 2023-09-15; listed known recoveries; compared unknown candidates with the expected bogus count.

## Commands
```bash
pytest tests/crossmatch tests/report tests/label
jupyter lab notebooks/crossmatch_vetting.ipynb
open reports/2023-09-15/index.html
```

## Pitfalls
- **Matching known objects with too tight a radius:** catalogue predictions aren't perfect (SkyBoT: ~2.9").
- **Believing numbers without looking at pixels** (#45).
- **Ignoring the expected number of false survivors.**
- **Submitting single-night tracklets.**
- **A test that passes before the fix doesn't test the bug.**

## Quiz
1. Why do we require at least half of a tracklet's detections to match the same known object?
2. A night gives 3 unknown candidates; validation predicts ~2 bogus per night. What do you conclude?
3. Which checklist items did #45 fail?
4. Why does the ADES writer add 15 s to OBSJD?
5. Why did my first no-fakes test not catch the bug?
