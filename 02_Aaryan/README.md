# Aaryan: Problem, data and preprocessing

**How I did my part, step by step** (the same steps are drawn in `00_Process_Diagram_Aaryan.svg`).

| Step | What | How, in one line | Main code |
|---|---|---|---|
| 1 | **Define the problem** | find moving dots in ZTF difference images; success = depth + known asteroids found | `project/docs/project_setup.md` |
| 2 | **Ask the archive first** | metadata query: 1,552 exposures, 16.4% flagged, 40 usable nights | `project/src/asthunt/acquire/query_ztf_metadata.py` |
| 3 | **Split by night, early** | 8 train / 2 validation / 1 test / 1 holdout night, fixed in a config | `project/configs/acquire.yaml` |
| 4 | **Download 4 files each** | science, mask, difference, PSF: 220 files, 3.55 GB | `project/src/asthunt/acquire/download_ztf_images.py` |
| 5 | **Clean every picture** | bad-pixel mask, reference = science - difference, noise level | `project/src/asthunt/clean/load_exposure.py` |
| 6 | **Find the blobs** | 2 sigma, 5 px minimum: about 7,800 blobs per frame | `project/src/asthunt/detect/extract_sources.py` |
| 7 | **Audit for leakage** | the first dataset cheated: 100% -> 50% -> 99%; rebuilt | `project/src/asthunt/evaluate/check_split_leakage.py` |

**Output:** Hands Paras a clean blob table and a leak-free split by night.

## What is in this folder (read in this order)
| File | What it is |
|---|---|
| `00_Process_Diagram_Aaryan.svg` | the steps above as one picture (open in a browser) |
| `01_Part_Brief.md` | what is mine, what I hand over, where the code is, numbers to know |
| `02_Knowledge_Pack.md` | everything I must understand, in detail |
| `03_Presentation_Guide.md` | slide by slide: what to say and for how long |
| `04_Viva_QA.md` | questions the examiner may ask, with model answers |
| `05_Handbook_Aaryan.html` / `.pdf` | files 01 to 04 plus all diagrams in one printable file |
| `06_Report_Aaryan.docx` | my part written up from start to end in simple words, with results |
| `07_Diagrams/` | pictures for my slides, numbered in the order of the steps |

Start with `../00_Start_Here/` (the whole project), then this folder. The runnable code is in `../project/`; the web demo is in `../web/`.
