# Advik: presentation guide

You speak **twice**: the opening (3 min) and the main block at the end (5 min + demo). Total about 8-9 minutes. If the slot is shorter, see the 12-minute plan at the end.

---

## Part A: the opening (3 minutes, 3 slides)

### Slide A1: "What we built" (45 s)
**On the slide:** title; one sentence; the picture `00_Start_Here/11_Diagrams/01_system_overview.svg` is for A2.
**Say:** "Telescopes photograph the same sky again and again. Stars stay still; asteroids move. We built a machine-learning pipeline that finds moving objects in old, public images from the Zwicky Transient Facility, tells real ones from junk, and, most important for this course, measures honestly how well it works."

### Slide A2: "The pipeline and the team" (1 min 30 s)
**On the slide:** `system_overview.svg` (colour = owner).
**Say:** "Images come from the ZTF archive. Aaryan will show how we prepare the data. Paras will show the neural network that decides which blobs are real. Satvik will show how blobs become tracks, how a tree model scores them, and the metrics. I will show how we evaluate everything, a deep search that goes fainter than any single frame, and a live demo. Two models learn: a CNN on pixels and gradient-boosted trees on track numbers. Everything else is physics, search and statistics."

### Slide A3: "The question we ask" (45 s)
**On the slide:** two bullets: "How deep can we see?" and "Can we trust the number?"
**Say:** "Accuracy is useless here: only 3% of blobs are real, so a model that says 'bogus' every time is 96.8% accurate and finds nothing. So we ask two harder questions: how faint an asteroid can we still recover, and how do we know that number is honest? The rest of the talk answers both."
**Transition:** "First, the data. Aaryan."

---

## Part B: the main block (5 minutes + 3 minute demo)

### Slide B1: "How we test: the evaluation protocol" (1 min)
**On the slide:** `07_Diagrams/02_evaluation_protocol.svg`.
**Say:** "We plant fake asteroids of known brightness into the real images, run the same pipeline we would use for the real search, and count what comes out. Thresholds are chosen on validation nights, never on test. One function, the scoreboard, produces every number. And there is a holdout night that only the scoreboard opens. When we compare models we compare at the same false-alarm level, because each model's own threshold lands on a different point of its trade-off curve."

### Slide B2: "What the scoreboard taught us" (1 min 15 s)
**On the slide:** `07_Diagrams/03_scoreboard_versions.svg`.
**Say:** "Version 1 looked perfect: the tracklet classifier had average precision 0.999. The scoreboard said otherwise: depth only about 20.5 magnitudes. Bigger magnitude means fainter. We measured where asteroids were being lost and it was not the classifier, it was linking. Relaxing linking found 11% more asteroids but made about 1,500 chance tracks per night; two physics features controlled them, giving 21.4. A bigger multi-frame network only tied the simple XGBoost, so we kept the simple one. Three times more training data did not help either, because one 30-second frame holds only so many photons."

### Slide B3: "The deep search" (1 min 30 s)
**On the slide:** `07_Diagrams/04_shift_and_stack.svg`.
**Say:** "If you cannot see an asteroid in one frame, add frames. Shift each frame along a guessed motion so the asteroid lands on the same pixel and add them: the signal grows N times, the noise only the square root of N. Eight frames give plus 1.1 magnitudes. We do not know the motion, so we try 4,888 of them on the GPU. That creates a statistics problem: pick the best of thousands of trials and noise alone looks significant. On pure noise we saw 6,179 false peaks at 5 sigma and only 44 at 6 sigma, so we require 6 sigma. A null test, with the frame times shuffled so real movers cannot line up, produced zero false tracks, while 20 of 20 known asteroids survived. Result: 50% completeness at magnitude 22.1, 0.7 magnitudes deeper than our best single-frame system on the same night."

### Demo (3 minutes): the live viewer
See the demo script below.

### Slide B4: "Limits and conclusion" (45 s)
**On the slide:** four bullets: limits and the one-line conclusion.
**Say:** "Limits: the best depth is for one night and one chip quadrant; our real labels come partly from simulation; the test night was looked at while fixing real bugs, which is why we also keep a holdout night; and 'unknown' only means 'not in the catalogues we queried'. We do not claim a discovery. What we do claim is a pipeline whose numbers you can trust, a clear picture of where the limit is, and honest negative results: a bigger model tied a small one, and more data did not help."

---

## Demo script (3 minutes, rehearse twice)
Setup: laptop with the data; `python scripts/live_viewer.py` already running; browser on `http://127.0.0.1:8765`; choose a **training** night (e.g. 2020-09-19, field 618, CCD 3, quadrant 4); fakes = 8; "step by step" ticked.
1. **Start.** "The agent now picks the night." Press Next. *Stage 2:* "It opens four files per exposure. Look at the third panel: the difference image, where stars have cancelled."
2. **Noise.** "The histogram should be a bell curve if there were only noise. The tail is the real blobs and artifacts."
3. **Plant fakes.** "These 8 fakes are our test objects, brightness and speed known."
4. **Find blobs.** "Seven thousand blobs in one frame, nearly all junk. The zoom shows one planted fake (teal ring) among noise."
5. **CNN stage.** "The network gives each blob a probability. The histogram shows junk piled near zero and our fakes near one." Show the "most convinced it is bogus" mosaic quickly.
6. **Link.** "Blobs that line up across frames become tracks; the close-ups show the straight lines."
7. **Result.** "6 of 8 fakes are recovered; the two faint ones (magnitudes 21.5 and 21.7) are missed. That is the depth limit you saw in the scoreboard. And no unexplained candidate: everything found is a known asteroid or a fake."
Switch to automatic (untick step by step) if time is short; the whole run takes about 20 seconds.
**If the demo fails:** show the pictures in `01_Advik/08_Demo_Screenshots/` (already saved: one per stage, plus the finished page) and say "the page runs the real pipeline functions; here is a recording of the run."

---

## Anticipated first questions (prepare 20-second answers)
1. "Did you find a new asteroid?" -> No; we recover known ones to about magnitude 21 and have no discovery claim.
2. "Why not just use accuracy?" -> 3% positives; accuracy paradox.
3. "How do you know the fakes are realistic?" -> AUC 0.47 when a classifier tries to tell them from real asteroids.
4. "Why does more data not help?" -> limited by photons in one frame, not by model size.
5. "What is the look-elsewhere effect?" -> picking the best of many trials; hence 6 sigma.

## The 12-minute plan (if your slot is short)
Opening: 1 slide, 1 minute (pipeline picture + "how deep and can we trust it"). Main block: B2 (1 min), B3 (1.5 min), a 30-second demo clip or two screenshots, B4 (30 s). Skip B1 but mention "we compare at equal false alarms" in B2.

## Delivery tips
- Speak to the diagram, not the screen text. Point at the box you are describing.
- Use "bigger magnitude = fainter" the first time you say a magnitude.
- Give *one* number per sentence. Round (20.5, 21.4, 22.1) and name the source if asked.
- If you do not know: "I did not test that; I would check it by ..."
