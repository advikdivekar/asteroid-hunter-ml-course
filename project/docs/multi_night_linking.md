# Phase: Multi-night linking (round 2)

## Goal
Connect one-night tracklets of the **same object** across nights days apart. The Minor Planet Center designates a new object only after it has been seen on **at least two nights**. A single-night tracklet is a candidate, never a discovery.

**Files built (`src/asthunt/link/`):**
- `measure_sky_motion.py`: one row per tracklet, giving its position at the middle time and its velocity (v_ra including cos dec, v_dec) in "/hr
- `link_tracklets_across_nights.py`: predict night 2 from night 1, apply a tolerance that grows with distance travelled, require velocity agreement, one partner per tracklet
- tests: a synthetic **curved** path 3 days apart is linked and an unrelated object is not; each tracklet gets at most one partner
- `notebooks/multi_night_linking.ipynb`

## How I thought about it
1. **Physics sets the tolerance.** Over 72 hours a main-belt asteroid moves ~2,700" (0.75°), and its path curves as Earth and the asteroid move along their orbits. A straight-line prediction is off by tens to a hundred arcseconds, so the tolerance has to scale with the distance travelled (15%). A fixed few-pixel radius would miss everything.
2. **Velocity is a second, independent measurement.** Each night measures its own velocity, and those barely change in 3 days (0.5–1.7 "/hr for real pairs).
3. **Check on known asteroids before trusting it.** They are free labelled data: a correct pair links an object to itself.

## Before → After (field 618, CCD 3, quadrant 4, 2019-01-20 → 2019-01-23)
| version | correct pairs (known on both nights) | wrong pairs |
|---|---|---|
| velocity tolerance 25%, every pair kept | 5 / 5 | 1 (2006 FD16 → 1996 AV6, velocity off by 7.7 "/hr) |
| **10% tolerance, best pair only** | **5 / 5** | **0** |

Separations of correct pairs were 10–131": exactly the curvature the growing tolerance allows for.

## Honest limits
- **Quadrant boundaries.** 0.75° in 3 days is almost a whole quadrant (0.87°), so most objects on night 1 sit in a *neighbouring* quadrant on night 2. A production system links across the whole field (64 quadrants) or several fields. The function already works on sky coordinates, so only the data feeding it has to change.
- **The 6 deep candidates** (2019-02-08) can't be linked: ZTF has no deep r-band night of that sky within a few days.
- Two nights still give a short arc. A reliable orbit needs three or more nights; that's the job of orbit-fitting tools (e.g. `find_orb`) and is out of scope.
- The IRSA archive was unreachable during this phase (connection refused). The field-567 nights (2019-01-28/31, 40 and 19 frames) will be downloaded automatically when it is back.

## Concepts
- **Tolerances from physics, then from data:** start from what the motion can do, then tighten on labelled examples.
- **Global assignment:** greedy best-first matching stops one tracklet from claiming two partners.
- **Arc length:** the longer the time span, the better the orbit. One night is minutes to hours, two nights is days, and a secure orbit takes weeks.

## Quiz
1. Why must the position tolerance grow with the time between nights?
2. What gave the wrong pair away, and why is that measurement independent of position?
3. Why does linking across quadrants matter for a 3-day gap?
4. Why is a two-night arc not yet an orbit?
