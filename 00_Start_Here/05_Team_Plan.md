# 05. Team plan: who owns what

## The split (by pipeline stage, matched to the marking scheme)

| | Owner | One-line title | Rubric marks it carries |
|---|---|---|---|
| 1 | **Advik** (lead) | **System architecture, evaluation, deep search, live viewer** | Model evaluation (3), documentation and presentation (2), plus the integration of everyone's work |
| 2 | **Aaryan** | **Problem, domain, data and preprocessing** | Problem understanding (3), data preprocessing and exploration (3, shared) |
| 3 | **Paras** | **Labels, dataset building and the neural network (CNN)** | Model development (4, shared), data preparation (shared) |
| 4 | **Satvik** | **Linking, the tracklet classifier (trees) and metrics** | Model development (4, shared), evaluation basics (shared) |

**Why Advik has the most important part:** it is the part that decides whether the numbers can be believed (the evaluation protocol, the scoreboard, the held-out nights, fair model comparison), it contains the main new method (the deep search, +0.7 magnitude) and it connects all other parts into one running system, plus the live viewer used in the demo. If this part is wrong, every other part's numbers are meaningless. Advik also opens and closes the presentation.

## How the parts connect (the "contract" between owners)
```
Aaryan  ->  units of images, split by night, cleaned (masks, noise, sky coordinates), blobs detected
Paras   ->  (3x31x31 stamps + labels) -> trained CNN -> p(real) for every blob
Satvik  ->  tracks from the CNN-kept blobs -> 17 features -> XGBoost score -> known/unknown
Advik   ->  runs it all as one deployed path, measures it on planted fakes and held-out nights, adds the deep search, shows it live
```
| Handover | What is handed over | File to look at |
|---|---|---|
| Aaryan -> Paras | `load_exposure` output, the blobs table from `extract_sources` (x, y, flux, SNR, shape), the nights split, `configs/acquire.yaml`, `detect.yaml` | `clean/load_exposure.py`, `detect/extract_sources.py`, `acquire/` |
| Paras -> Satvik | `detections` table with a `score` column (cross-fitted for training nights) and the CNN threshold | `models/predict_realbogus.py`, `models/cross_fit_realbogus_scores.py` |
| Satvik -> Advik | tracklets with `tracklet_score` and `known_name` | `pipeline/run_night_pipeline.py` |
| Advik -> all | the scoreboard: the one place numbers come from | `evaluate/build_scoreboard.py`, `reports/scoreboard.csv` |

Rule: **every number in your slides must be traceable to a file** (a lesson document, a notebook output or `scoreboard.csv`). If you cannot point at the file, do not say the number.

## Presentation order (adjust to your time slot; default 20 minutes + questions)
| Order | Who | Time | Content |
|---|---|---|---|
| 1 | Advik | 3 min | The problem, the idea, the whole pipeline in one picture, who does what |
| 2 | Aaryan | 4 min | The data, difference imaging, preprocessing, splitting by night, leakage |
| 3 | Paras | 5 min | Labels (injection-recovery), stamps, the CNN, training, results, the label-bug story |
| 4 | Satvik | 5 min | Linking, the 17 features, trees and boosting, metrics, why accuracy fails |
| 5 | Advik | 5 min | Evaluation protocol, scoreboard v1 to v4, deep search, **live demo** (3 min), limits, conclusion |
| | everyone | | questions: each answers about their own part first; anyone may add |

If you only get **12 minutes**: 2 + 2 + 3 + 3 + 2 (skip the live demo; show one screenshot per stage).

## Timeline suggestion (about 2 weeks; compress if needed)
| Days | Everyone | Advik | Aaryan | Paras | Satvik |
|---|---|---|---|---|---|
| 1-2 | Read `00_Start_Here`; run the tests; open the viewer once | prepare the demo; check numbers against `scoreboard.csv` | read `02_Knowledge_Pack.md`; re-run `data_acquisition` + `difference_imaging` notebooks | re-run `neural_networks` notebook; read `realbogus_cnn.md` | re-run `ml_basics` + `metrics_evaluation` notebooks |
| 3-6 | Study your pack; write your own summary in your own words | scoreboard + deep search notebooks; draft the report's evaluation chapter | leakage audit notebook; EDA figures for the slides | dataset building + CNN notebooks; error-analysis story | linking + tracklet classifier notebooks; metrics exercises |
| 7-9 | Slides from `03_Presentation_Guide.md`; practise 3 times with a timer | integrate slides; rehearse the demo | | | |
| 10-12 | Cross-viva each other with `04_Viva_QA.md` and `06_Whole_Project_Viva_QA.md` | | | | |
| 13-14 | Dress rehearsal, fix weak answers, finalise the report | | | | |

## Rules for the team
1. **Everyone must be able to answer questions on the whole pipeline at the level of `06_Whole_Project_Viva_QA.md`**, not only their own part.
2. **Run things yourself.** Re-run your notebooks, change one setting, see what changes, and write what you saw. This is what makes the explanation yours (see `07_AI_Use_and_Integrity.md`).
3. **One source of numbers.** If a number changes, tell Advik; slides and the report are updated from one list.
4. **Say "I don't know" correctly:** "I did not test that; my guess is ... and I would check by ...". That is a good viva answer. A made-up number is not.
5. **Do not claim a discovery.** Ever.

## Report structure (suggested; one section per owner)
1. Introduction and problem formulation (Aaryan) · 2. Related work and the paper(s) implemented (Aaryan + Paras) · 3. Data and preprocessing, exploration (Aaryan)
4. Labels and dataset (Paras) · 5. Models: CNN (Paras), classical models (Satvik) · 6. Evaluation methodology and results (Advik, with Satvik's metrics)
7. Deep search and live system (Advik) · 8. Limitations and negative results (Advik) · 9. Conclusion · References · AI-use declaration (everyone, see file 07)
