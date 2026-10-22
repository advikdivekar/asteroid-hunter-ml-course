# Asteroid Hunter: the Machine Learning course project (team folder)

Course: **Machine Learning (702DB0C015)**, B.Tech CSE (Data Science), Semester V, 2026-27, MPSTME NMIMS.
Team: **Advik, Aaryan, Paras, Satvik**.

**One sentence:** we built a classical machine-learning pipeline that looks through old, public images from the Zwicky Transient Facility (a telescope at Palomar) for moving dots (asteroids), and we measured honestly how well it works.

## How this folder is organised

```
asteroid_ml_project/
  README.md                    <- you are here
  00_Start_Here/               <- EVERYONE reads this first (about 1 hour)
      00_How_Our_Model_Works.svg  the deployed ML model (classical: 49 measured numbers + 600 decision trees), with a real stamp
      00b_CNN_Comparison.svg      the deep-learning CNN we tried first, kept only as a comparison
      01_Project_Overview.md      what the project is, the results, the pipeline
      02_Domain_Primer.md         the astronomy you need, in plain words
      03_ML_Primer.md             the ML ideas we used, tied to the course syllabus
      04_Course_Mapping.md        the course policy document -> what we did, what is missing
      05_Team_Plan.md             who owns what, how the parts connect, presentation order
      06_Whole_Project_Viva_QA.md questions EVERY member must be able to answer
      07_AI_Use_and_Integrity.md  how to be honest about AI help (the course has a GenAI policy)
      08_Glossary.md              every term in one place
      09_Handbook_Start_Here.html / .pdf   files 01-08 in one printable file
      10_Group_Report_Asteroid_Hunter.docx the whole project from start to end, with results
      11_Diagrams/                01 system overview, 02 night split
  01_Advik/    <- lead: system, evaluation, deep search, live viewer
  02_Aaryan/   <- problem, domain, data, preprocessing, exploration
  03_Paras/    <- labels, dataset, the classical ML model (features, PCA, XGBoost; SVM/forest/CNN compared; k-means)
  04_Satvik/   <- linking, tracklet classifier (trees), metrics
        each person's folder has (numbered in reading order):
            README.md                      my part step by step + what each file is
            00_Process_Diagram_<Name>.svg  how I did my part, as one picture
            01_Part_Brief.md               what is yours, what to hand over, where the code is
            02_Knowledge_Pack.md           everything you must understand (your part, then the other parts in brief)
            03_Presentation_Guide.md       slide by slide, what to say, timing, demo cues
            04_Viva_QA.md                  questions and model answers
            05_Handbook_<Name>.html / .pdf files 01-04 plus the diagrams in one printable file
            06_Report_<Name>.docx          my part from start to end in simple words, with results
            07_Diagrams/                   the pictures for your slides, numbered in step order
            08_Demo_Screenshots/           (Advik only) a saved picture of every live-viewer stage
  web/         <- the Next.js web app: the pipeline step by step on real data, the classical ML model running in the browser, results
  project/     <- the full runnable code (src, tests, configs, models, notebooks, docs). No raw data.
  tools/       <- make_diagrams.py, make_project_figures.py, build_handbooks.py, export_web_data.py, make_reports.js
```

## Run it (5 minutes)

```bash
cd project
conda env create -f environment.yml        # once
conda activate asthunt
pip install -e .                           # once: makes `import asthunt` work
pytest -q                                  # 171 tests, all offline, ~45 s
```

The live viewer needs real images on disk. The raw data (about 50 GB) is **not** in this folder. On Advik's computer it is in
`~/Desktop/asteroid/data`; to use the viewer from this folder, either run the viewer from the original project
(`cd ~/Desktop/asteroid && python scripts/live_viewer.py`) or copy/link that `data` folder into `project/` (do NOT zip a link: it would try to include 50 GB).

## See it work (web app)

```bash
cd web
npm install      # once
npm run dev      # open http://localhost:3000
```
No Python and no raw data needed: the app ships a small slice of the real test night and the trained model (600 trees; the CNN too, for comparison), made by `tools/export_web_data.py`.

## Read in this order
1. `00_Start_Here/01_Project_Overview.md`, then `05_Team_Plan.md`, then `04_Course_Mapping.md`.
2. Your own folder: `README.md` -> `01_Part_Brief.md` -> `02_Knowledge_Pack.md` -> `03_Presentation_Guide.md` -> `04_Viva_QA.md`.
3. `06_Whole_Project_Viva_QA.md` together, out loud, taking turns.

## What this project is NOT
It is **not** a claimed asteroid discovery. The system finds known asteroids reliably down to about magnitude 21 and has one faint
object in a related study that passes every check so far; nothing has been reported anywhere. In the viva, say exactly that.
