# Satvik: Linking, tracklet classifier and metrics

**How I did my part, step by step** (the same steps are drawn in `00_Process_Diagram_Satvik.svg`).

| Step | What | How, in one line | Main code |
|---|---|---|---|
| 1 | **Keep the model's blobs** | drop blobs under the stage-1 cut (set low, so almost no real blob is lost) | `project/src/asthunt/models/predict_realbogus.py` |
| 2 | **One shared sky grid** | the telescope points slightly differently each time: correct for it | `project/src/asthunt/link/measure_sky_motion.py` |
| 3 | **Link into tracks** | pairs + KD-tree look-up + straight-line fit: 52,000 -> 3,500 seeds | `project/src/asthunt/link/link_detections_into_tracklets.py` |
| 4 | **17 numbers per track** | speed, straightness, steady brightness, stage-1 score, forced photometry | `project/src/asthunt/features/compute_tracklet_features.py` |
| 5 | **Compare 4 models** | logistic regression, tree, forest, XGBoost; cross-validated by night | `project/src/asthunt/models/train_tracklet_classifier.py` |
| 6 | **Pick the threshold** | on validation nights only, at a fixed recall | `project/src/asthunt/evaluate/choose_threshold.py` |
| 7 | **Use the right metrics** | accuracy lies (96.8% for 'always bogus'); use precision, recall, AP | `project/src/asthunt/evaluate/compute_classification_metrics.py` |
| 8 | **Known or unknown?** | match tracks to the asteroid catalogue; checklist for a human | `project/src/asthunt/crossmatch/crossmatch_known_asteroids.py` |

**Output:** Hands Advik scored tracks, each marked known or unknown.

## What is in this folder (read in this order)
| File | What it is |
|---|---|
| `00_Process_Diagram_Satvik.svg` | the steps above as one picture (open in a browser) |
| `01_Part_Brief.md` | what is mine, what I hand over, where the code is, numbers to know |
| `02_Knowledge_Pack.md` | everything I must understand, in detail |
| `03_Presentation_Guide.md` | slide by slide: what to say and for how long |
| `04_Viva_QA.md` | questions the examiner may ask, with model answers |
| `05_Handbook_Satvik.html` / `.pdf` | files 01 to 04 plus all diagrams in one printable file |
| `06_Report_Satvik.docx` | my part written up from start to end in simple words, with results |
| `07_Diagrams/` | pictures for my slides, numbered in the order of the steps |

Start with `../00_Start_Here/` (the whole project), then this folder. The runnable code is in `../project/`; the web demo is in `../web/`.
