# 06. Whole-project viva: questions every member must answer

How to use: take turns asking each other, answer out loud in 30-60 seconds, then read the model answer. The model answers are short on purpose; add your own example from *your* part.

### The basics
**Q1. What does the project do, in one sentence?**
It is an ML pipeline that searches archival ZTF telescope images for moving objects (asteroids), separates real ones from junk with a CNN and a tree model, checks them against the catalogue of known asteroids, and measures honestly how deep and how reliable the search is.

**Q2. Why difference images?**
Subtracting an old reference of the same sky from tonight's picture cancels stars, galaxies and sky glow, so only new or moving things remain. That shrinks the search from about 1,950 sources to a few hundred strong ones per frame.

**Q3. Why does the project need machine learning at all?**
Most blobs in a difference image are junk, in many shapes (dipoles, cosmic rays, edges, noise); only about 3% are real. Hand-written rules cannot capture all those shapes. A CNN learns them from labelled examples, and gradient-boosted trees combine many weak track-level clues better than a single cut. The rest of the pipeline (matched filtering, linking, shift-and-stack) is physics and search, not learning.

**Q4. What are the two ML models, and what are their inputs and outputs?**
(1) A CNN: input a 3 x 31 x 31 stamp (science, reference, difference), output p(real) for one blob. (2) XGBoost: input 17 numbers describing a track (speed, straightness, brightness steadiness, CNN scores, forced photometry...), output a score for the whole track.

**Q5. What is a unit and what is a tracklet?**
A unit is one field + CCD + quadrant + night. A tracklet is a set of detections in different frames that lie on a straight line at a plausible asteroid speed (at least 3 of 5 frames, within 1.5 px of the line, 8-120 arcsec/hour).

### Data and honesty
**Q6. Why split by night and not randomly?**
Frames from one night share the same stars, noise and artifacts. A random split would put near-copies in train and test and inflate scores. Splitting by night, decided in a config file before modelling, keeps the test honest.

**Q7. Give an example of data leakage from your own project.**
Our first dataset had `mag` and `true_rate` columns that existed only for planted fakes, so "is mag missing?" equalled the label. Also, the random bogus tracklets were 222+ pixels off a line while real ones were within 2.6 px, so a one-line rule scored 100%. Trained on that data, a model got 50% (a coin flip) on realistic near-miss tracklets and 99% when trained on realistic data.

**Q8. How do you get labels when the sky is not labelled?**
Three sources: planted fake asteroids (perfect labels, but only as real as the simulation), known asteroids from catalogues (real, but fewer, positions from JPL Horizons), and everything else called bogus (with a little label noise, because it includes undiscovered real objects).

**Q9. How do you know your fakes look like real asteroids?**
A classifier trained to tell fakes from real known asteroids (matched in brightness, cross-validated by night) scored AUC 0.46-0.48, where 0.5 means indistinguishable. Also real asteroids and fakes both fade out between magnitudes 20.5 and 21.5.

**Q10. What was the biggest bug and how was it found?**
The telescope pointing shifts between exposures (up to 12 px). We linked blobs in each frame's own pixel coordinates, so real asteroids did not line up while fakes (planted straight in pixel coordinates) did. Real asteroids all showed the same residual pattern, which pointed at a per-frame error. Fix: put every blob on frame 1's sky grid before linking; real residuals fell from about 2.5 px to 0.05-0.47 px. Lesson: test the simulation against real objects, not only against itself.

### Models and metrics
**Q11. Why not just report accuracy?**
Only about 3% of blobs are real, so "always bogus" scores 96.8% accuracy while finding nothing. We report recall (how many real objects we keep), precision (how pure the flagged set is), false alarms per frame and average precision.

**Q12. Why is a precision-recall curve better than ROC here?**
ROC's x-axis divides by all bogus (65,000), so hundreds of false alarms look like a tiny step: ROC AUC was 0.95 while precision at useful recall was under 20%. Precision divides by the number of alerts, so it shows directly how polluted the output is.

**Q13. How did you choose the thresholds?**
On the validation nights only, never on test: the CNN cut keeps 95% of real detections; the tracklet cut keeps 95% of real tracklets. Fixing recall first and then counting false alarms follows the cost argument: a missed asteroid is a lost discovery, a false alarm costs one minute of a person's time.

**Q14. What is cross-fitting and why did you need it?**
The tracklet model uses the CNN's scores as features. A CNN scoring nights it trained on is over-confident, and the tracklet model would learn that. So training nights are scored by CNNs trained on other nights (4 folds of 2 nights), and validation/test nights by the final CNN. It is stacking without leakage.

**Q15. How do you prevent overfitting?**
CNN: dropout 0.3, data augmentation (rotations and mirrors), early stopping on validation AP with patience 4. XGBoost: shallow trees (depth 3), small learning rate (0.05), 80% row and column subsampling. Everything is judged on nights the model never saw.

**Q16. The tracklet classifier scores AP 0.999. Is it great?**
No. That number is saturated: 96% of tracklets in that data are real because we plant 300 fakes per night, the opposite of the real sky. It cannot show improvement. What matters is depth, known asteroids recovered, and bogus tracklets per night on held-out nights.

### Results and evaluation
**Q17. What does "50% completeness at magnitude 20.5" mean?**
Plant fake asteroids of known brightness; at magnitude 20.5 the system recovers half of them. Bigger magnitude is fainter, so a deeper system has a bigger number.

**Q18. How did the system improve from v1 to v4, and why?**
v1 reached depth about 20.5. We measured where asteroids were lost: at linking, not at the classifier. Relaxing linking found 11% more but created about 1,500 chance tracks per night, which became hard negatives; adding two physics features (static neighbours, forced photometry) controlled them. Depth on the test night rose to 21.4. A multi-frame CNN only tied the simple XGBoost, so we kept XGBoost.

**Q19. Why did 3x more training data (v5) not help?**
At equal false-alarm levels v4 was equal or slightly better. The system is limited by how many photons one 30-second frame holds, so more examples cannot reveal what is not in the image. That is why the deep search, which adds frames, gained +0.7 magnitude, more than any model change.

**Q20. What is the deep search and why does it work?**
Shift the frames along a trial motion so a moving object lands on the same pixel and add them. The signal adds N times but the noise only sqrt(N) times, so 8 frames gain about 2.8x, +1.1 magnitudes. We try 4,888 motions on the GPU. Because we pick the best of thousands of trials, noise alone reaches 5 sigma (look-elsewhere effect), so we require 6 sigma, and a null test with shuffled frame times produced 0 false tracks.

**Q21. How do you know the deep search is not finding noise?**
Calibrate the threshold on pure noise; a null test (shuffle the frame times so real movers no longer line up) gives 0 false tracks after the velocity-consistency cut; the cut also kept 20 of 20 known asteroids. Injection-recovery gives 94-100% recovery at magnitude 21.4-21.9.

**Q22. Why compare models at the same false-alarm level?**
A rule like "95% recall on validation" lands on different points of each model's trade-off curve. v5 looked 0.8 mag worse at its own threshold, but at matched alarms it tied v4. Fair comparison means the same operating point.

### Limits and ethics
**Q23. Did you discover an asteroid?**
No. The system recovers known asteroids reliably to about magnitude 21. One faint object in a related two-night study passes every check we have (an orbit fit, noise tests, no catalogue match), but that is not a discovery: it needs a human to read the stamps and the Minor Planet Center's process. Nothing has been reported.

**Q24. What are the main limitations?**
Depth is capped by single frames (the deep search helps on one quadrant of one night); the training data covers one field mostly; "real" labels come partly from simulation; catalogue positions have errors; the test night is no longer perfectly untouched because real bugs were fixed after looking (so a separate holdout night exists); and "unknown" only means "not in the catalogue we queried".

**Q25. What would you do with more time?**
Evaluate on more fields and nights; try an SVM and clustering of the "bogus zoo" for the course syllabus; add a learning curve; run the deep search on more nights; get an independent confirmation from another survey.

**Q26. How reproducible is it?**
Configs hold every setting; the night split is in a config; random seeds are `[project seed, night date]`; model versions are bundles with the commit and nights used; archive answers and known-asteroid lookups are cached; about 150 offline tests; `environment.yml` pins the software.

**Q27. How did you use AI in this project?** (answer truthfully; see `07_AI_Use_and_Integrity.md`)
The project was developed with an AI coding assistant (Claude) under Advik's direction, including code, documentation and diagrams; each of us then re-ran and studied our part. We declare this as the course policy requires. Be ready to show that you understand your own part without notes.

### Where the course topics are
**Q28. Which syllabus topics does the project use?**
Logistic regression (baseline), gradient descent and backpropagation (from scratch), neural networks/CNN, regularisation (L2, dropout, early stopping), train/validation/test, error analysis and skewed-class metrics, decision trees, bagging, random forests and boosting, ROC. Not used: SVM, PCA, clustering, normal equation (we say so openly).

**Q29. If I gave you 3 minutes and a whiteboard, how would you draw the pipeline?**
Data (IRSA, split by night) -> difference image -> detect blobs -> stamps -> CNN -> keep blobs above a cut -> link into tracks on the sky grid -> 17 features -> XGBoost -> cross-match with known asteroids -> unknown candidates -> a person. Underneath: planted fakes and held-out nights measure every stage; the deep search adds frames along trial motions for fainter objects.

**Q30. What did you personally learn that surprised you?** (prepare your own answer)
Good examples: a model's most confident "mistakes" revealed a labelling error; a near-perfect AP meant nothing; adding data did not help; the evaluation design mattered more than the model.
