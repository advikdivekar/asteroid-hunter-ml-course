# Advik: viva questions and model answers

Answer in your own words; the answers below are the skeleton. Sections: A system, B evaluation, C deep search, D fast/live/viewer, E the whole project and ML in general, F tricky ones.

## A. System and engineering
**1. Why does the scoreboard call `run_night_pipeline` instead of reading saved tracklet files?**
To measure exactly what would be deployed. Notebooks that re-implement steps drift from the real pipeline, and then their numbers describe something nobody runs. One path, used with planted fakes for evaluation and without for the real search.

**2. Why must the thresholds live in the same bundle as the models?**
A threshold is only meaningful for the model that produced the scores. Mixing a new CNN with an old cut is a classic production bug, so a version = CNN + tracklet model + thresholds + commit + nights, never overwritten, so we can roll back.

**3. What does a "bundle" contain and how is a version chosen?**
`models/v4/`: `realbogus_cnn.pt`, `tracklet_classifier.json`, `thresholds.json` (CNN 0.0081, tracklet 0.601, min 3 detections), `bundle.json` (commit, train and validation nights, the 17 feature names). `configs/live.yaml` names the bundle in use (v4).

**4. How is the project reproducible?**
Settings in `configs/`; night split in a config; seeds `[project seed, night date]`; archive answers cached to CSV; known-asteroid lookups cached; `environment.yml`; about 150 offline tests; model names carry the git sha.

**5. What do the tests protect? Give two examples.**
The end-to-end pipeline on tiny fake frames with pointing shifts (fakes must link and never be reported as candidates), the leakage check (no night in two splits), and the viewer test (traced run equals the pipeline). Many tests reproduce a bug we actually had, e.g. the static-neighbour feature counting a slow asteroid's own detections.

## B. Evaluation
**6. What is injection-recovery and why is it the standard method?**
Plant objects of known properties, run the normal pipeline, count what comes out. It gives labels and completeness (recovery vs brightness/speed). The sky is not labelled; our own fakes are. Weakness: fakes are only as realistic as the simulation, so we test the realism (AUC 0.46-0.48 when a classifier tries to separate fakes from real known asteroids).

**7. What does "depth" or "50% completeness at magnitude 20.5" mean?**
Half of the planted fakes at that brightness are recovered. Bigger magnitude = fainter, so deeper means a larger number.

**8. v1 had tracklet AP 0.999. Why is that a poor metric for judging v2?**
It is saturated: 96% of tracklets in that data are real (we plant 300 fakes per night), the opposite of the real sky. A saturated metric cannot show improvement. Depth, known asteroids recovered and unexplained alarms per night have headroom.

**9. What would make the holdout night stop being a holdout?**
Looking at it to choose a setting, a threshold, a feature or a model. It is opened in one place only, to write one number. Honesty: the test night was looked at while fixing real bugs, which is why a separate holdout exists.

**10. v5 (more training data) had depth 20.56 on test vs 21.34 for v4. Is v5 worse?**
Not really. Each model's threshold is chosen by a rule ("95% recall on validation") that lands at a different point on its own trade-off curve; v5's landed at a much stricter point (0.3 alarms per night vs 5). At matched alarms (about 10, 5, 3 per night) v4 was equal or slightly better (21.38 vs 21.28; 21.34 vs 21.25; 21.33 vs 21.09). Compare at the same operating point.

**11. Why did more data not help?**
The system is capped by what a single 30-second frame can show (about 21.4 on a good night), not by the model. More examples cannot reveal photons that are not there. Data helps when the model is the bottleneck; here linking and the frame depth were.

**12. How did you decide the problem was linking, not the classifier?**
Before building a model we measured where planted asteroids were lost: with 4-of-5 linking, 184 of 294 fakes were linked; with 3-of-5, 200; at 99% stage-1 recall, 205 of 294 (but 1,499 bogus tracklets). Relaxed linking finds 11% more.

**13. A multi-frame CNN scored AP 0.9921 and XGBoost 0.9924. What do you do?**
It is a tie, and the 3-detection subset has only 33 real tracklets so its differences are noise. Use the simplest model that ties: XGBoost. (Average of both: 0.9933, a negligible gain.)

**14. What went wrong with v2 and how did you fix it?**
On the dense test night v2 raised 15 unexplained alarms. We looked at all 15 by eye: "artifact walkers" (slow tracks along a faint line present in every frame) and "noise alignments" (three 2-sigma blips with nothing at the predicted spot in other frames). We added one physics feature for each failure mode: static neighbours and forced photometry. Alarms fell to 7 on that night (v4).

**15. You found a bug in your own feature. Tell the story.**
The static-neighbour feature counted detections left behind at a tracklet's own positions in other frames, but slow asteroids leave their own detections nearby, so it flagged 22% of real tracklets. I checked the feature on real examples before trusting the rule, fixed it (0.4%), added a test that reproduces the bug, retired v3 (trained with the buggy feature) and trained v4 correctly. Lesson: check a feature on real examples; do not retrain old versions with new code silently.

## C. Deep search
**16. Explain shift-and-stack and the +1.1 magnitude.**
Shift frames along a trial motion so a moving object lands on one pixel and add. Signal adds N times; independent noise adds sqrt(N) times. N = 8: 8 / 2.83 = 2.83x in SNR. Magnitude gain = 2.5 log10(2.83) = 1.13.

**17. Why 4,888 trial motions?**
We do not know the motion. We cover every direction up to 120 arcsec/hour in steps of 3 px/hour (a half step over a 0.5 h stack is a 0.75 px worst-case smear). The GPU makes it feasible.

**18. What is the look-elsewhere effect and how did you handle it?**
Testing many places or motions makes chance peaks likely (picking the best of 4,888). On pure-noise stacks, 5 sigma gave 6,179 false peaks and 6 sigma only 44, so the threshold is 6 sigma, set from data.

**19. Real data are not Gaussian. What went wrong first and how did you fix it?**
At 6 sigma, real frames still gave about 47,000 peaks. Fixes in order: clip each frame at 5 sigma (no frame dominates); subtract the night's median per pixel (static halos and galaxies no longer stack at any motion) -> 5,765; standardise per pixel by median and scatter (residuals that wobble with the seeing are turned down) -> 391; mask pixels single frames already see (otherwise bright asteroids make echo tracklets) -> 94.

**20. What is a null test and what does it show?**
Repeat the whole search with the frame times shuffled so real movers no longer line up; whatever it finds is a false alarm. We got 16 false tracklets per night before the velocity-consistency cut and 0 after, while 20 of 20 known asteroids were kept. It shows the false-alarm rate of the method on that data; it does not prove any single candidate is real.

**21. What is the velocity-consistency cut?**
Each stack's own best motion must agree with the night-long motion within 6 px/hour. Known asteroids agree to 2.4 px/h; null tracklets differ by 61 px/h.

**22. What does the deep search deliver and what are its limits?**
50% completeness at magnitude 22.12 vs 21.41 for the best single-frame system on the same night (+0.7). Limits: needs at least 20 frames in a night (three stacks); built and measured on one night and quadrant; fakes brighter than about 20.9 are masked on purpose because single frames already see them.

**23. How do you convert stack SNR to magnitude?**
A fitted straight line: mag = 26.25 - 5.23 log10(mean stack SNR), scatter 0.17 mag, from 100 found fakes. (A small regression.)

## D. Fast movers, live system, viewer
**24. Why a separate fast-mover search?**
Near-Earth objects move 300-15,000 arcsec/hour, so they trail (about rate/120 pixels in 30 s) and a round-blob detector under-counts them. A matched filter for trails plus a consistency check (trail points along the motion within 15 degrees; length equals speed x exposure) recovers 97% of planted trails on the tuning nights, 93% on a later night.

**25. What did you learn from real fast tracks?**
Six of seven we inspected ran along CCD columns: bleed-trail residuals. A veto exists but is off by default because it was chosen on the same tracks (optimistic) and costs about 1 point of recovery.

**26. How does the live viewer prove it shows the truth?**
It calls the same functions in the same order as the pipeline. A test on tiny frames, and a run on a real night, show identical detections, scores, tracks and candidates (max difference 0.0).

**27. What would you change to run it in production?**
`run_live_monitor`: poll the archive, wait until a night is complete, run point + fast + deep, store results idempotently, alert on unknown objects. A person reads every alert.

## E. Big picture
**28. What is the single most important lesson of the project?**
Measuring honestly beats modelling cleverly: the scoreboard exposed that a perfect-looking metric hid a depth problem, the simplest model tied the complex one, and the biggest gain came from physics (adding frames), not from more learning.

**29. What would you do with another month?**
Evaluate the deep search on more fields; add the SVM and clustering experiments the syllabus mentions; a learning-curve study; independent confirmation from another survey for any object we cared about.

**30. If we gave you only 60 seconds for the whole project?**
"A pipeline that finds moving objects in telescope images. A CNN removes junk blobs, a line search links the rest into tracks, a tree model scores them, a catalogue check removes the known ones. We measure everything on planted fakes and held-out nights; single frames reach magnitude 21.4, shift-and-stack reaches 22.1, and bigger models or more data did not help. No discovery is claimed."

## F. Tricky ones
**31. "You said honest, but you looked at the test night while fixing bugs."** Yes, and we say so. The fixes were physics corrections (coordinates, units, a minimum speed), not tuning to raise a score. To be clean we added a holdout night that only the scoreboard opens; its numbers match validation, which is the evidence nothing was secretly tuned.

**32. "Isn't planting fakes circular?"** It is circular only if fakes differ from reality. We use the real PSF, noise and frame times, put them on the sky correctly, test realism by adversarial validation (AUC 0.47), and cross-check with real known asteroids: both fade between magnitudes 20.5 and 21.5.

**33. "Your CNN got AP 0.938 on test. Is that good?"** It beat the feature baseline on every split and cuts false alarms 4.6x at the same recall. But AP on a planted-fake-heavy set flatters; the numbers that matter are depth, known asteroids recovered and unexplained alarms per night.

**34. "How much of this did you write yourselves?"** (answer truthfully; see `00_Start_Here/07_AI_Use_and_Integrity.md`). State that the project was built with an AI coding assistant under your direction, say which parts you personally ran, changed or extended, and demonstrate understanding by explaining a part without notes.
