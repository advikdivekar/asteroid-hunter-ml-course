# Side check: the Kepler post

## Goal
Apply the project's habits (cross-match, quantify, name the decisive test) to the claim that started it: *"a 0.24% dip every 13.77 days on KIC 5689121 is a new planet."*

**Files built:** `notebooks/kepler_post_check.ipynb` (queries the Kepler Input Catalog via VizieR and the NASA Exoplanet Archive).

---

## How I thought about it
1. **Search for the star.** A first web search found nothing about KIC 5689121. It did find KIC 5689351 with a catalogued **13.77-day** candidate (KOI-505.01). Neighbouring catalogue numbers are often neighbours on the sky.
2. **Get exact numbers.** Archive queries gave positions, brightnesses and the confirmed planet's period, depth and transit times.
3. **Test my own first idea.** I'd suggested "blended light from the neighbour". The dilution arithmetic shows simple blending *can't* make a dip that deep. I kept that result instead of hiding it.
4. **Stop at what the data supports.** We don't have the post's light curve, so the verdict is "suspicious, with a decisive test", not "debunked".

---

## Findings
| | KIC 5689121 (post) | KIC 5689351 (KOI-505) |
|---|---|---|
| Kepler magnitude | 16.33 | 14.19 (7.2× brighter) |
| Temperature | 3,786 K (red dwarf ✓) | 4,985 K |
| Separation | 263" = **66 Kepler pixels** | |
| Known planet | none | **Kepler-169 e**, confirmed, P = **13.7671 d**, depth 685 ppm |

- **The period matches** the post's 13.77 d. Chance of an unrelated signal landing within ±0.01 d of one of this system's 5 periods: about **3 × 10⁻³ (1 in 350)**.
- **The dilution limit:** leaked light can only make a dip *shallower* than the source's. The most simple blending could produce is **0.0685%**; the post reports **0.24%**, 3.5× deeper.
- **Possible ways to reconcile both:** electronic **crosstalk** between CCD outputs, or a **baseline error** in custom pixel photometry (over-subtracted background shrinks the faint star's flux and inflates any relative dip).

**The decisive test:** do the post's dips happen at Kepler-169 e's transit times, BKJD = 174.81089 + n × 13.76710 (BKJD = BJD − 2454833)? If they line up, it's contamination. If not, the signal is worth a careful look.

---

## Concepts
- **Ephemeris matching:** two signals with the same period *and* the same transit times almost certainly come from one source. That's how Kepler's own vetting caught contamination.
- **Dilution:** observed depth = true depth × (contaminating flux) / (total flux in the aperture). It can't exceed the true depth.
- **Coincidence probability:** turn "looks similar" into a number before deciding. Here 1 in 350 is suggestive, not proof.
- **Burden of proof:** an extraordinary claim (a new planet) needs the ordinary explanations ruled out first.

## Steps, in order
1. Web search → KOI-505 with a 13.77 d candidate.
2. VizieR KIC lookup for both stars → magnitudes, temperatures, separation.
3. Exoplanet Archive query → KOI-505.01 = Kepler-169 e, confirmed, period and depth.
4. Coincidence estimate, dilution limit, transit ephemeris.

## Quiz
1. Why can't blended light from Kepler-169 e's host make a 0.24% dip on its own?
2. How could bad background subtraction make a dip look deeper?
3. What single comparison would settle whether the post's signal is Kepler-169 e?
4. How does our asteroid pipeline perform the equivalent of this check?
