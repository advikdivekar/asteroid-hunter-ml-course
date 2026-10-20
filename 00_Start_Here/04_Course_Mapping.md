# 04. The course policy document and what we have done against it

Source: *Course Policy, Machine Learning (702DB0C015)*, B.Tech CSE (Data Science), Semester V, 2026-27, MPSTME NMIMS. Coordinator: Prof. Abhay Kolhe.

## 1. What the document asks of the project
- **Weight:** the project is 15 marks of the 50-mark internal assessment (marked at the 4th and 11th week).
- **Team size:** "Group of 2-3 students". **We are four.** Ask the coordinator to approve a group of four (or decide how to present it) *before* the presentation.
- **Topic approval:** "the faculty member should approve the topic and title of the project by the end of 3rd week". Confirm that approval exists in writing; if it does not, get it now.
- **What to do:** "Identify and implement a journal/IEEE conference research paper based on some topic relevant to the course... Read, understand, summarize and implement the paper(s)."
- **Marking scheme:** Problem Understanding and Formulation (3), Data Preprocessing and Exploration (3), Model Development (4), Model Evaluation and Performance (3), Documentation and Presentation (2).
- **Course outcomes the project maps to:** CO1 (describe requirements of ML algorithms), CO2 (select and regularise linear and nonlinear models), CO3 (design a machine-learning system), CO4 (classification and clustering to extract insight).
- **GenAI policy (read `07_AI_Use_and_Integrity.md`):** AI may help with planning, research, learning new concepts and deployment, but students must not use it "to develop entire project; the core implementation and critical thinking aspects of the project must be the students' own work", and "students need to cite the references used".

## 2. Which research paper(s) does this project implement?
The document requires a paper. Our project was inspired by a post, not by a single paper, so we map it to the closest published work. **Confirm the final choice with your coordinator.** All four were checked to exist:

| Role | Paper | How our project relates |
|---|---|---|
| **Primary** (CNN part) | Duev, D. A. et al. (2019), *Real-bogus classification for the Zwicky Transient Facility using deep learning*, MNRAS 489, 3582 (arXiv:1907.11259) | Same task family: a CNN that separates real detections from false positives in ZTF data. Differences to state openly: ours scores moving-object blobs, uses 31x31 pixel science/reference/difference stamps, and gets labels from planted fakes plus known asteroids. Read the paper and write down exactly what you replicated and what you changed. |
| Supporting (ML in ZTF) | Mahabal, A. et al. (2019), *Machine Learning for the Zwicky Transient Facility*, PASP 131, 038002 (arXiv:1902.01936) | Overview of ML at ZTF: real/bogus, star/galaxy, asteroid detection, training samples. Good for the problem formulation and related work. |
| Supporting (deep search) | Whidden, P. J. et al. (2019), *Fast algorithms for slow moving asteroids: constraints on the distribution of Kuiper Belt Objects*, AJ 157, 119 (arXiv:1901.02492) | GPU shift-and-stack over a huge number of trial motions (their KBMOD software). Our deep search is the same idea at a smaller scale (4,888 motions). |
| Supporting (data) | Masci, F. J. et al. (2019), *The Zwicky Transient Facility: Data Processing, Products, and Archive*, PASP 131, 018003 (arXiv:1902.01872) | Describes the difference images and the archive we use. |
| Inspiration | Shallue, C. J. & Vanderburg, A. (2018), *Identifying exoplanets with deep learning...*, AJ 155, 94 (arXiv:1712.05044) | The "AI on old Kepler data" idea the project started from. |
| Methods | Chen, T. & Guestrin, C. (2016), *XGBoost: a scalable tree boosting system*, KDD; Kingma, D. & Ba, J. (2015), *Adam: a method for stochastic optimization*, ICLR | The boosting library and the optimiser we use. |

*(Only the abstracts and bibliographic details were verified online; you must read the papers yourselves before citing details.)*

## 3. The marking scheme against what we have (evidence and owner)
| Criterion (marks) | What we have | Evidence in the folder | Owner |
|---|---|---|---|
| Problem understanding and formulation (3) | The problem, why ML, why difference images, success metrics (recall at a fixed false-alarm level, depth), what is out of scope | `01_Project_Overview`, `02_Domain_Primer`, `project/docs/astronomy_data_primer.md`, `project/PLAN.md` | **Aaryan** |
| Data preprocessing and exploration (3) | Metadata-first acquisition, split by night, masks, noise estimation, EDA (thresholds, SNR histograms, stamp zoo), leakage audit, normalisation, augmentation, balancing | `project/docs/data_acquisition.md`, `difference_imaging.md`, `leakage_audit.md`, `dataset_building.md` + notebooks | **Aaryan** (data) and **Paras** (dataset building) |
| Model development (4) | CNN from scratch (backprop notebook first), XGBoost with explained settings, logistic-regression baseline, model comparison (tree, forest, boosting), a multi-frame model that tied | `project/docs/realbogus_cnn.md`, `tracklet_classifier.md`, `multiframe_model.md`, `ml_basics.md`, `neural_networks.md` | **Paras** (CNN) and **Satvik** (classical models) |
| Model evaluation and performance (3) | Metrics for skewed classes, PR vs ROC, calibration, completeness curves, scoreboard on held-out nights, comparison at equal false alarms, negative results | `project/docs/metrics_evaluation.md`, `scoreboard.md`, `more_data.md`, `reports/scoreboard.csv` | **Advik** (with Satvik on metrics) |
| Documentation and presentation (2) | One lesson per phase with a quiz, notebooks, architecture diagrams, this folder, live viewer demo | `project/docs/`, `project/docs/architecture/`, every `PRESENTATION_GUIDE.md` | **Advik** (lead) and everyone |

## 4. Syllabus coverage, honestly
"Used" = the topic is really in the project (with the file). "Partly" = touched but not a main method. "Not used" = a gap.

| Syllabus topic (session) | Status | Where / note |
|---|---|---|
| Supervised vs unsupervised learning | Supervised: used. Unsupervised: not used | CNN, XGBoost are supervised |
| Linear regression, multiple, polynomial | Partly | one linear fit (magnitude vs log SNR) in the deep search; no regression model as such |
| Logistic regression | **Used** | baseline (`models/train_baseline_classifier.py`, `docs/ml_basics.md`) |
| Hypothesis, gradient descent, cost function | **Used** | from-scratch numpy network in `notebooks/neural_networks.ipynb` |
| Normal equation | Not used | |
| Regularisation | **Used** | L2 in logistic regression, dropout, early stopping, XGBoost subsampling, augmentation |
| Neural networks: hypothesis, cost, backprop, activation | **Used** | `docs/neural_networks.md`, CNN |
| Evaluating a hypothesis, train/val/test | **Used** | split by night; leakage audit |
| Bias and variance curves | Partly | overfitting demo with train/validation loss; no formal learning-curve study |
| Error analysis, metrics for skewed classes, precision/recall trade-off | **Used** | `docs/metrics_evaluation.md`, error analysis in the CNN and tracklet docs |
| Decision trees, classification trees, trees vs linear models | **Used** | `docs/tracklet_classifier.md` (depth-3 tree vs linear) |
| Regression trees | Not used | |
| Bagging, random forests, boosting | **Used** | forest and XGBoost cross-validated |
| SVM (maximal margin, SVC, multiclass, relation to logistic regression) | **Not used** | gap |
| ROC curves | **Used** | ROC vs precision-recall |
| PCA, clustering, k-means, hierarchical clustering | **Not used** | gap (CO4 asks for classification **and clustering**) |
| Anomaly detection and large-scale ML | Partly | "unknown tracks" is an anomaly-style filter by catalogue lookup; GPU shift-and-stack, float16 stamps, streaming pipeline for scale |

### How to close the gaps if your faculty expects them (small, honest extras, not required by the policy text)
1. **SVM on the tracklet features** next to XGBoost (same cross-validation by night): covers SVM and its relation to logistic regression (CO3).
2. **PCA, then k-means on the CNN's stamp embeddings** (or on raw normalised stamps) to explore the "bogus zoo" (dipoles, spikes, edges): covers PCA and clustering (CO4) and strengthens the exploration marks.
3. A **learning curve** (validation score vs number of training nights): covers bias/variance curves.
Each is a day of work for one person and is a *new* experiment you must run and understand yourselves. Ask Advik if you want the starting code reviewed.

## 5. What we have done, in one paragraph (for the report's abstract)
We implemented an end-to-end machine-learning pipeline for detecting moving objects in archival ZTF difference images. We built reproducible data acquisition split by night, generated labels by injection-recovery and catalogue matching and audited their quality, trained a convolutional real/bogus classifier (average precision 0.899 on validation, 0.938 on the test night, 4.6x fewer false alarms than a logistic-regression baseline), linked detections into tracks and scored them with gradient-boosted trees, and evaluated the whole system with a scoreboard on held-out nights using planted asteroids (depth of about 20.5 magnitudes for single frames, 21.4 after improving linking, 22.1 with a shift-and-stack deep search). We report negative results honestly (a multi-frame CNN tied XGBoost; three times more training data did not help) and make no discovery claim.

## 6. Things to decide with your coordinator this week
- [ ] Is a group of four acceptable, and how will individual contribution be assessed?
- [ ] Is the topic approved (title, paper choice)? Get it in writing.
- [ ] How should AI assistance be declared (see `07_AI_Use_and_Integrity.md`)?
- [ ] Is a live demo allowed/expected? (The live viewer needs the data on a laptop.)
- [ ] Will the missing syllabus topics (SVM, clustering) be expected?
