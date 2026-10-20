# Aaryan: presentation guide

**Your slot: 4 minutes, 4 slides.** (In the 12-minute plan: 2 minutes, slides 1 and 4 plus one sentence on the split.)
Pictures live in `diagrams/` and `../00_Start_Here/diagrams/`.

---

### Slide 1: "The data and the problem" (1 min)
**On the slide:** `data_flow.svg`; three bullets: ZTF difference images; 5 frames of 30 s each night; 97% of blobs are junk.
**Say:** "The data is public, from the Zwicky Transient Facility at Palomar, served by NASA's IRSA archive. For every 30-second exposure we get four files: the science image, a mask of bad pixels, a difference image and the blur shape. Before downloading anything we asked the archive for its table of exposures: 1,552 for our patch of sky, 16% with quality flags which we drop. Then we chose nights, wrote the split into a config file, and downloaded safely: the downloader can be re-run, writes files only when complete, and waits and retries when the archive is busy, which it was, twice."

### Slide 2: "Difference imaging" (1 min)
**On the slide:** `difference_imaging.svg`.
**Say:** "A telescope picture is mostly stars. We use ZTF's difference image: tonight's picture minus a deep picture of the same sky from long ago. Everything that did not change cancels. What is left is new or moving things, which is what we want, but also leftovers from imperfect subtraction: dipoles from bright stars, cosmic rays, noise. Most of what remains is junk, which is why we need a classifier. Detection itself is simple: smooth with the shape of a star and mark everything two times the noise or more. That gives about 7,800 blobs per frame, only 173 of them strong. We tuned detection to miss nothing, and leave the junk removal to the neural network."

### Slide 3: "Splitting by night and the leakage audit" (1 min 15 s)
**On the slide:** `../00_Start_Here/diagrams/split_by_night.svg` on the left and `leakage_types.svg` (or only its red experiment box) on the right.
**Say:** "We split by night, not by random rows, and wrote the split into a config before any modelling: eight nights to train, two to validate, one best night to test, and one never-touched holdout. Frames from one night share stars, noise and artifacts, so a random split would leak. This mattered, because our very first dataset cheated in four ways: columns that existed only for fake asteroids, all samples from the same four images, bogus examples so easy that one rule separated them, and a test night that was different. We proved it: trained and tested on the easy data the model scored 100%; trained on the easy data and tested on realistic near-misses it scored 50%, a coin flip; trained on realistic data it scored 99%."

### Slide 4: "What we hand to the model" (45 s)
**On the slide:** `preprocessing_funnel.svg`.
**Say:** "After this stage the next person receives clean exposures, a mask, a noise estimate, and a table of blobs on a split that cannot leak. The numbers to remember: 8, 2, 1, 1 nights; about 7,800 blobs per frame; 3% of them real. Over to Paras, who will turn each blob into three small pictures and teach a neural network to judge them."

---

## Delivery tips
- Keep the *why* before the *what*: say why you ask for metadata first, why split by night.
- On slide 3, slow down at 100% -> 50% -> 99%: it is the strongest evidence in your part.
- Do not read numbers from the slide; say them naturally and round.
- If asked about astronomy details you do not know, say what you do know (the idea) and offer the term from the glossary.

## Anticipated first questions
1. "Why a difference image?" -> static things cancel; only new or moving remain. 2. "Why split by night?" -> shared stars/noise/artifacts would leak. 3. "Why 2 sigma?" -> tuned for completeness, measured, later checked with injections. 4. "What is a mask?" -> per-pixel flags; we mask defects but not ghosts. 5. "How do you know the data is good?" -> quality flags dropped, bad pixels masked, split audited.
