# 08. Glossary

| Term | Meaning (plain words) |
|---|---|
| **Accuracy** | fraction of all predictions that are right; misleading when one class is rare |
| **Adam** | a gradient-descent variant that adapts each weight's step size; our optimiser |
| **Adversarial validation** | train a classifier to tell two datasets apart; if it cannot (AUC 0.5) they are alike |
| **Anomaly detection** | finding items that do not fit the known pattern; our "unknown" tracks are anomalies by catalogue lookup |
| **Arcsecond ("")** | 1/3600 of a degree; one ZTF pixel is about 1.01 arcsec |
| **Asteroid (main belt)** | rocky body orbiting the Sun, mostly between Mars and Jupiter |
| **Augmentation** | making extra training examples by rotating/mirroring; sky has no "up" |
| **Average precision (AP)** | area under the precision-recall curve; a single summary for rare positives |
| **Backpropagation** | chain rule from output back to every weight; gives all gradients in one sweep |
| **Bagging** | train many models on random samples and average them (random forest) |
| **Base rate** | how common the positive class really is (about 3% real) |
| **Bias / variance** | systematic error vs sensitivity to the training sample; underfit vs overfit |
| **Bleed trail** | charge from a saturated star spilling along a CCD column; causes false fast tracks |
| **Boosting** | trees built one after another, each fixing the previous errors (XGBoost) |
| **Bogus** | a detection that is not a real moving object (artifact, noise...) |
| **Bundle (model)** | a saved version: stage-1 model + tracklet model + thresholds + provenance (`models/v6` is deployed: classical ML; `models/v4` has the CNN) |
| **Calibration** | making scores mean real probabilities (reliability plot, isotonic regression) |
| **CCD / quadrant** | a camera chip / one quarter of a chip (about 3072 x 3080 px) |
| **Class imbalance** | one class is much rarer than the other |
| **CNN** | convolutional neural network: filters slide over an image to find local patterns (we keep it only as a comparison) |
| **Feature engineering** | measuring numbers with a meaning from raw data (29 per stamp here) |
| **PCA** | principal component analysis: keeps the few directions in which the data vary most (675 pixels -> 20 numbers) |
| **k-means** | clustering: groups points around k centres, without labels (6 kinds of junk) |
| **SVM** | support vector machine: the widest-margin boundary between classes |
| **Common grid** | frame 1's pixel grid; every blob is projected onto it before linking |
| **Completeness** | fraction of planted fakes recovered (recall by brightness) |
| **Confusion matrix** | counts of true/false positives and negatives |
| **Cross-fitting** | scoring each fold with a model that did not train on it, so stage 2 sees honest scores |
| **Cross-validation (GroupKFold)** | repeated train/test by groups (nights), never by random rows |
| **Deep search** | shift-and-stack along trial motions to reach fainter than one frame |
| **Depth** | the brightness (magnitude) where 50% of fakes are recovered |
| **Difference image** | science minus reference: only new things remain |
| **Dipole** | bright half beside dark half: imperfect subtraction of a bright star |
| **Dropout** | randomly switching off neurons during training to reduce overfitting |
| **Early stopping** | stop training when validation performance stops improving; keep the best epoch |
| **Ephemeris** | a table of where an object should be at given times |
| **FITS** | the standard astronomical image file format |
| **Forced photometry** | measuring flux at a predicted position even where nothing was detected |
| **Gradient descent** | repeatedly move weights against the gradient of the loss |
| **Holdout night** | a night never used for choosing anything; opened only to write one number |
| **IRSA** | NASA/IPAC archive that serves the ZTF data |
| **Injection-recovery** | plant fakes, run the pipeline, count what comes out |
| **Isotonic regression** | monotone curve used to calibrate scores |
| **KD-tree** | data structure for fast nearest-neighbour lookups |
| **Leakage** | information at training time that will not exist at use time (target, group, shortcut, shift) |
| **Linking** | joining blobs from different frames into one straight-line track |
| **Log-loss (cross-entropy)** | the loss for probabilities: confident and wrong costs a lot |
| **Look-elsewhere effect** | testing many places makes chance peaks likely; raise the threshold |
| **Magnitude** | log brightness scale: bigger is fainter; 5 mag = 100x |
| **MAD** | median absolute deviation; robust spread: sigma = 1.4826 x MAD |
| **Mask** | per-pixel flags (saturated, dead, cosmic ray, ghost...) |
| **Matched filter** | smoothing with the expected source shape to lift faint sources out of noise |
| **MPC** | Minor Planet Center: the official keeper of asteroid observations |
| **Normalisation** | putting stamps in units of their own noise: (x - median) / (1.4826 MAD) |
| **Null test** | repeat the search on data where there is nothing to find (shuffled times) to measure false alarms |
| **Opposition** | asteroid opposite the Sun: brightest, most numerous |
| **Overfitting** | learning the noise of the training set |
| **Precision / recall / FPR** | purity of what is flagged / share of real objects kept / share of bogus wrongly flagged |
| **PSF** | point spread function: the blur shape of a point source |
| **Reference image** | deep picture of the same sky built from old exposures |
| **Regularisation** | any device that discourages overfitting (L2, dropout, early stopping, subsampling) |
| **ROC / PR curve** | recall vs false-positive rate / precision vs recall |
| **Scoreboard** | one function that computes every headline number from the deployed pipeline |
| **Sigma** | the typical size of the noise; "5 sigma" = 5 times it |
| **SkyBoT / JPL Horizons** | services that predict where known asteroids are (Horizons is more accurate) |
| **SNR** | signal-to-noise ratio: flux divided by its uncertainty |
| **Stamp (triplet)** | 3 x 31 x 31 cut-out: science, reference, difference |
| **Shift-and-stack** | shift frames to follow a trial motion, then add them |
| **Tracklet** | a set of detections that line up as one moving object |
| **Unit** | one field + CCD + quadrant + night |
| **WCS** | the mapping between pixels and sky coordinates (RA, Dec) |
| **XGBoost** | a fast gradient-boosted tree library |
| **ZTF** | Zwicky Transient Facility, a survey telescope at Palomar Observatory |
