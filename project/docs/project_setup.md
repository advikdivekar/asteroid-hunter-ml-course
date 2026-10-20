# Phase: Project setup

## Goal
Before any ML, build a workshop where:
- anyone (including future you) can recreate the exact same software with one command,
- every file has one job and its name says what that job is,
- data, code and results live in separate, predictable places,
- changes are recorded, so you can always go back.

ML projects rarely fail because the model was bad. Much more often they fail because nobody can reproduce what was run, which data was used, or which version of the code produced a result. This phase fixes that up front.

---

## How I thought about it

1. **Look at what already exists.** I read every script and checked the data before planning anything. I found:
   - 5 working scripts with the same functions copy-pasted between them (`load_image`, `inject_asteroid`, `detect_sources`, `linearity_residual`),
   - absolute paths like `/Users/advikdivekar/Desktop/asteroid/...` written into the code,
   - `sys.path.insert(...)` hacks so one script can import another,
   - libraries installed in the base conda Python 3.14, with no record of which versions,
   - no version control (not a git repository).
2. **Don't destroy working code.** The old scripts work and contain good ideas, so I moved them to `legacy/` unchanged. We will rebuild each idea properly, one stage at a time, and compare against the old version.
3. **Pick boring, standard tools.** conda for the environment, a `src/` package layout, YAML for settings, pytest for tests, git for history. Nothing exotic, so every tutorial and Stack Overflow answer applies.
4. **Python 3.12, not 3.14.** Scientific libraries (PyTorch especially) support new Python versions months after release. 3.12 is the safe choice.
5. **Only build what this phase needs.** The plan has ~35 files. Right now I created only the folders and one real file (`load_config.py`), because it's the first thing every other stage will use.

---

## Concepts (with small examples)

### 1. Environment = a recipe for your software
Your laptop had at least 6 different Pythons (Homebrew, python.org 3.12 and 3.14, system, miniconda base and 3 conda envs). Running `python script.py` used whichever came first on your `PATH`.

`environment.yml` is a **recipe**: it lists the packages, and `conda env create -f environment.yml` cooks a fresh, isolated Python called `asthunt`.

*Example of why it matters:* numpy 2.0 changed some behaviour. If your teammate has numpy 1.26 and you have 2.1, the same code can give different answers, and nobody knows why.

*The nuance:* our recipe lists packages without exact versions, so a rebuild next year may get newer ones. Once everything works, we'll **lock** it with `conda env export > environment.lock.yml`, which pins exact versions.

### 2. The `src/` package layout
```
src/asthunt/detect/extract_sources.py   →   from asthunt.detect.extract_sources import extract_sources
```
`pip install -e .` (the `-e .` line in `environment.yml`) tells Python "the package `asthunt` lives in this folder". The `-e` means **editable**: change the code and the import sees the change immediately, with no reinstall. This replaces all the `sys.path.insert` hacks.

### 3. One purpose per file
- `extract_sources.py` contains exactly one public function, `extract_sources()`.
- Folder = pipeline stage, so `detect/extract_sources.py` reads as "the detect stage, extract sources step".

**Why:** when something breaks, the file name tells you where to look. When you want to test one step, there's exactly one file to test. When you swap a method (SEP → another detector), you replace one file.

### 4. Configs instead of hard-coded numbers
Old code: `DETECT_THRESH = 6.0` in one script, `3.0` in another. Which one produced `tracklets.csv`? You can't tell.

New code: settings live in `configs/<stage>.yaml`, and `load_config("project")` reads them. Later, every output file gets saved next to the config that produced it, so you always know.

`load_config` also turns relative paths (`data/raw`) into absolute ones rooted at the project. Move the whole folder somewhere else and nothing breaks.

### 5. Separate code, data and results
```
src/        code: small, goes into git
configs/    settings: small, goes into git
data/       raw → interim → processed: big, NOT in git
models/     trained weights: big, NOT in git
reports/    outputs: regenerable, NOT in git
```
**Rule: `data/raw` is read-only.** Every stage reads files and writes *new* files. If a step is wrong, delete its output and re-run. The raw data is never damaged.

Git stores text history well. It handles 40 MB FITS files badly, which is why `.gitignore` excludes `data/`, `*.fits` and the 155 MB Miniconda installer.

### 6. Tests
A test is a small piece of code that checks another piece of code does what it promises.
```python
def test_missing_config_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="no_such_stage"):
        load_config("no_such_stage")
```
This may look trivial, but tests become essential later. Example: "an injected mag-18 asteroid is detected." If a future change breaks detection, the test fails immediately, instead of you finding out after training a model for an hour on broken data.

### 7. Version control (git)
git takes snapshots ("commits") of your code. Benefits:
- you can return to any earlier state,
- every model file will be stamped with the commit it came from (`realbogus_cnn__20261013__a1b2c3d.pt`), so results are traceable to exact code.

---

## Steps, in the order I did them
1. **Read** every existing script, the data README, and `tracklets.csv`. Checked which Pythons and libraries exist.
2. **Wrote the plan** (`PLAN.md`) and got your approval.
3. **Wrote `environment.yml`**: Python 3.12 plus the astro, ML, notebook and test libraries from conda-forge; `-e .` to install our package. *(Later correction: I first installed PyTorch from pip. That build carries its own copy of the OpenMP library, which clashed with conda's and crashed the process (`OMP: Error #15`). PyTorch now comes from conda-forge too, and the Apple GPU still works. Lesson: mix conda and pip as little as possible.)*
4. **Wrote `pyproject.toml`**, which tells pip that `asthunt` lives in `src/` and pytest that tests live in `tests/`.
5. **Wrote `.gitignore`**: data, models, reports, FITS files, caches, the installer.
6. **Moved old files** to `legacy/` (scripts, `tracklets.csv`, PNGs) and `ztf_test/` to `data/raw/ztf_test/`. Deleted `__pycache__/` (it's an auto-generated cache).
7. **Created stage folders** in `src/asthunt/` and matching `tests/` folders. Each package folder has an `__init__.py`, which marks it as a Python package.
8. **Wrote `configs/project.yaml`**: seed, paths and our ZTF field.
9. **Wrote `src/asthunt/config/load_config.py` and its tests.**
10. **Created the `asthunt` conda env**, registered it as a Jupyter kernel, and ran the tests.
11. **Wrote the notebook** `notebooks/project_setup.ipynb` and ran it end to end.
12. **`git init`** and made the first commit.

## Commands to reproduce
```bash
cd ~/Desktop/asteroid
conda env create -f environment.yml          # build the environment (several minutes)
conda activate asthunt
python -m ipykernel install --user --name asthunt   # make it selectable in Jupyter
pytest                                         # all tests should pass
jupyter lab notebooks/project_setup.ipynb      # run every cell
```

## Pitfalls
- **Mixing pip and conda for compiled libraries.** Two copies of a low-level library (here OpenMP) in one process can crash it. Prefer one package source.
- **The wrong Python.** If `import sep` fails, run `which python`. It must point inside `.../envs/asthunt/`. In Jupyter, pick the kernel named **asthunt**.
- **The legacy scripts no longer run as-is.** They contain absolute paths to `ztf_test/` (now moved to `data/raw/ztf_test/`). That's intentional: they are reference material. We'll rebuild each piece properly.
- **Never commit data.** If `git status` ever shows a `.fits` file, stop and fix `.gitignore`.
- **Editing files in `data/raw/`** breaks reproducibility. Treat it as read-only.

## Quiz
1. Why did we pick Python 3.12 instead of the 3.14 you already had?
2. What does the `-e` in `pip install -e .` do, and which old hack does it replace?
3. Name two reasons why FITS files must not go into git.
4. The old code had `DETECT_THRESH = 6.0` in one file and `3.0` in another. Which project rule prevents that now?
5. In which folder and file would you expect the code that matches injected asteroids to detections? (Use the naming rule.)
