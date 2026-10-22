# Advik: System, evaluation, deep search and live viewer (lead)

**How I did my part, step by step** (the same steps are drawn in `00_Process_Diagram_Advik.svg`).

| Step | What | How, in one line | Main code |
|---|---|---|---|
| 1 | **Design the system** | split into 4 stages with clear hand-overs; one deployed code path | `project/src/asthunt/pipeline/run_night_pipeline.py` |
| 2 | **Set the test rules** | plant fakes on nights nobody tuned on; compare at equal false alarms | `project/src/asthunt/evaluate/measure_completeness.py` |
| 3 | **Build the scoreboard** | one table for every version v1 -> v6; every number comes from here | `project/src/asthunt/evaluate/build_scoreboard.py` |
| 4 | **Find where we lose** | asteroids were lost at linking, not the classifier: fixed (v2, v4) | `project/docs/scoreboard.md` |
| 5 | **Classical ML only (v6)** | replaced the CNN with XGBoost: same known asteroids, depth within 0.07 | `project/scripts/evaluate_bundle.py` |
| 6 | **Deep search** | add frames along 4,888 trial motions; 6 sigma; null test 0 false: 22.1 mag | `project/src/asthunt/deep/stack_and_find_peaks.py` |
| 7 | **Live viewer + demo** | runs the real pipeline stage by stage in the browser | `project/scripts/live_viewer.py` |
| 8 | **Make it reproducible** | versioned model bundles, configs, 171 offline tests | `project/src/asthunt/models/save_model_bundle.py` |

**Output:** Result: depth 20.5 -> 21.3 -> 22.1 mag with classical ML; 40 of 44 known asteroids; no discovery claimed.

## What is in this folder (read in this order)
| File | What it is |
|---|---|
| `00_Process_Diagram_Advik.svg` | the steps above as one picture (open in a browser) |
| `01_Part_Brief.md` | what is mine, what I hand over, where the code is, numbers to know |
| `02_Knowledge_Pack.md` | everything I must understand, in detail |
| `03_Presentation_Guide.md` | slide by slide: what to say and for how long |
| `04_Viva_QA.md` | questions the examiner may ask, with model answers |
| `05_Handbook_Advik.html` / `.pdf` | files 01 to 04 plus all diagrams in one printable file |
| `06_Report_Advik.docx` | my part written up from start to end in simple words, with results |
| `07_Diagrams/` | pictures for my slides, numbered in the order of the steps |
| `08_Demo_Screenshots/` | one saved picture per live-viewer stage, in case the live demo fails |

Start with `../00_Start_Here/` (the whole project), then this folder. The runnable code is in `../project/`; the web demo is in `../web/`.
