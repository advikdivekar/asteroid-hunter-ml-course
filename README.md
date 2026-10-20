# Asteroid Hunter: the Machine Learning course project (team folder)

Course: **Machine Learning (702DB0C015)**, B.Tech CSE (Data Science), Semester V, 2026-27, MPSTME NMIMS.
Team: **Advik, Aaryan, Paras, Satvik**.

**One sentence:** we built a machine-learning pipeline that looks through old, public images from the Zwicky Transient Facility (a telescope at Palomar) for moving dots (asteroids), and we measured honestly how well it works.

## How this folder is organised

```
asteroid_ml_project/
  README.md                    <- you are here
  00_Start_Here/               <- EVERYONE reads this first (about 1 hour)
      01_Project_Overview.md      what the project is, the results, the pipeline
      02_Domain_Primer.md         the astronomy you need, in plain words
      03_ML_Primer.md             the ML ideas we used, tied to the course syllabus
      04_Course_Mapping.md        the course policy document -> what we did, what is missing
      05_Team_Plan.md             who owns what, how the parts connect, presentation order
      06_Whole_Project_Viva_QA.md questions EVERY member must be able to answer
      07_AI_Use_and_Integrity.md  how to be honest about AI help (the course has a GenAI policy)
      08_Glossary.md              every term in one place
      diagrams/                   system overview, night split
  01_Advik/    <- lead: system, evaluation, deep search, live viewer
  02_Aaryan/   <- problem, domain, data, preprocessing, exploration
  03_Paras/    <- labels, dataset, the neural network (CNN)
  04_Satvik/   <- linking, tracklet classifier (trees), metrics
        each person's folder has:
            PART_BRIEF.md          what is yours, what to hand over, where the code is
            KNOWLEDGE_PACK.md      everything you must understand (your part, then the other parts in brief)
            PRESENTATION_GUIDE.md  slide by slide, what to say, timing, demo cues
            VIVA_QA.md             questions and model answers
            diagrams/              the pictures for your slides
            Handbook_<Name>.html / .pdf   the four documents above in one printable file
  project/     <- the full runnable code (src, tests, configs, models, notebooks, docs). No raw data.
  tools/       <- make_diagrams.py, build_handbooks.py (regenerate the diagrams and handbooks)
```

## Run it (5 minutes)

```bash
cd project
conda env create -f environment.yml        # once
conda activate asthunt
pip install -e .                           # once: makes `import asthunt` work
pytest -q                                  # 166 tests, all offline, ~40 s
```

The live viewer needs real images on disk. The raw data (about 50 GB) is **not** in this folder. On Advik's computer it is in
`~/Desktop/asteroid/data`; to use the viewer from this folder, either run the viewer from the original project
(`cd ~/Desktop/asteroid && python scripts/live_viewer.py`) or copy/link that `data` folder into `project/` (do NOT zip a link: it would try to include 50 GB).

## Read in this order
1. `00_Start_Here/01_Project_Overview.md`, then `05_Team_Plan.md`, then `04_Course_Mapping.md`.
2. Your own folder: `PART_BRIEF.md` -> `KNOWLEDGE_PACK.md` -> `PRESENTATION_GUIDE.md` -> `VIVA_QA.md`.
3. `06_Whole_Project_Viva_QA.md` together, out loud, taking turns.

## What this project is NOT
It is **not** a claimed asteroid discovery. The system finds known asteroids reliably down to about magnitude 21 and has one faint
object in a related study that passes every check so far; nothing has been reported anywhere. In the viva, say exactly that.
