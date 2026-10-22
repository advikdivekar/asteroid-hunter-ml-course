# Paras: Labels, dataset and the ML model

**How I did my part, step by step** (the same steps are drawn in `00_Process_Diagram_Paras.svg`).

| Step | What | How, in one line | Main code |
|---|---|---|---|
| 1 | **Plant fake asteroids** | draw fakes with ZTF's own blur shape: 300 per night | `project/src/asthunt/simulate/plant_fakes_in_exposure.py` |
| 2 | **Label every blob** | fake or known asteroid within 2 px = real; the rest = bogus | `project/src/asthunt/label/label_detections.py` |
| 3 | **Cut 3 x 31 x 31 stamps** | science, reference and difference around each blob | `project/src/asthunt/detect/cut_triplet_stamps.py` |
| 4 | **Prepare the data** | normalise and balance: 53,178 training stamps, 16.7% real | `project/src/asthunt/dataset/assemble_stamp_dataset.py` |
| 5 | **Measure features + PCA** | 29 numbers with a meaning per stamp, plus 20 PCA numbers | `project/src/asthunt/features/compute_stamp_features.py` |
| 6 | **Compare 5 ML models** | LR, SVM, forest, XGBoost; XGBoost + PCA wins: AP 0.902 | `project/scripts/compare_realbogus_models.py` |
| 7 | **Cluster the junk** | k-means on PCA: 6 kinds of junk; CNN only as a comparison | `project/scripts/compare_realbogus_models.py` |
| 8 | **Cross-fit the scores** | each night scored by a model that never saw it | `project/scripts/train_classical_bundle.py` |

**Output:** Hands Satvik every blob with a model score and the cut.

## What is in this folder (read in this order)
| File | What it is |
|---|---|
| `00_Process_Diagram_Paras.svg` | the steps above as one picture (open in a browser) |
| `01_Part_Brief.md` | what is mine, what I hand over, where the code is, numbers to know |
| `02_Knowledge_Pack.md` | everything I must understand, in detail |
| `03_Presentation_Guide.md` | slide by slide: what to say and for how long |
| `04_Viva_QA.md` | questions the examiner may ask, with model answers |
| `05_Handbook_Paras.html` / `.pdf` | files 01 to 04 plus all diagrams in one printable file |
| `06_Report_Paras.docx` | my part written up from start to end in simple words, with results |
| `07_Diagrams/` | pictures for my slides, numbered in the order of the steps |

Start with `../00_Start_Here/` (the whole project), then this folder. The runnable code is in `../project/`; the web demo is in `../web/`.
