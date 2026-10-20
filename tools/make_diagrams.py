"""Draw every diagram of the course folder as a clean SVG (white background, so it prints and goes into slides).

    python3 tools/make_diagrams.py

One colour per owner: Advik violet, Aaryan teal, Paras blue, Satvik amber, shared grey. Numbers in the diagrams come from the
project's own results (docs/ and reports/scoreboard.csv in project/); if a result changes, change it here and re-run.
"""

import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
FONT = "Helvetica, Arial, sans-serif"
INK, MUTED = "#1f2937", "#6b7280"
COLOURS = {                                   # (strong, light)
    "advik": ("#6d28d9", "#ede9fe"), "aaryan": ("#0f766e", "#ccfbf1"), "paras": ("#1d4ed8", "#dbeafe"),
    "satvik": ("#b45309", "#fef3c7"), "grey": ("#4b5563", "#f3f4f6"), "red": ("#b91c1c", "#fee2e2"), "green": ("#15803d", "#dcfce7"),
}


class Svg:
    def __init__(self, width, height, title):
        self.width, self.height, self.parts = width, height, []
        self.title = title
        self.text(width / 2, 30, title, size=20, weight="bold")

    def rect(self, x, y, w, h, who="grey", rx=10, dashed=False, fill=None):
        strong, light = COLOURS[who]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or light}" stroke="{strong}" stroke-width="1.6"{dash}/>')

    def text(self, x, y, s, size=14, weight="normal", fill=INK, anchor="middle", italic=False):
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{style}>{escape(str(s))}</text>')

    def box(self, x, y, w, h, lines, who="grey", title=None, size=13, dashed=False, title_size=14):
        """A rounded box; `title` in bold on top, then the lines, all centred. Long lines wrap by words; a box with a title and no
        lines is a container (its title sits at the top so smaller boxes can be drawn inside it)."""
        self.rect(x, y, w, h, who, dashed=dashed)
        strong = COLOURS[who][0]
        if title and not lines:
            self.text(x + w / 2, y + 26, title, size=title_size, weight="bold", fill=strong)
            return
        width_chars = max(8, int((w - 16) / (size * 0.56)))
        wrapped = []
        for line in lines:
            wrapped += textwrap.wrap(line, width_chars) or [""]
        rows = ([("t", title)] if title else []) + [("l", line) for line in wrapped]
        step = size + 6
        top = y + h / 2 - (len(rows) - 1) * step / 2 + size / 3
        for k, (kind, line) in enumerate(rows):
            if kind == "t":
                self.text(x + w / 2, top + k * step, line, size=title_size, weight="bold", fill=strong)
            else:
                self.text(x + w / 2, top + k * step, line, size=size)

    def arrow(self, x1, y1, x2, y2, label=None, colour=MUTED, label_dy=-6, label_dx=0):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="2" marker-end="url(#head)"/>')
        if label:
            self.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy, label, size=11, fill=MUTED)

    def line(self, x1, y1, x2, y2, colour=MUTED, width=1.5, dashed=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="{width}"{dash}/>')

    def circle(self, x, y, r, colour, fill=None):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or "none"}" stroke="{colour}" stroke-width="1.6"/>')

    def note(self, x, y, s, size=12, anchor="middle", fill=MUTED):
        self.text(x, y, s, size=size, anchor=anchor, fill=fill, italic=True)

    def save(self, path):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}">'
               f'<defs><marker id="head" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="{MUTED}"/></marker></defs>'
               f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)


def flow(svg, y, h, items, x0=40, gap=34, w=None, who="grey", size=13):
    """A left-to-right row of boxes joined by arrows. items = [(title, [lines]), ...]; returns the x of each box."""
    count = len(items)
    w = w or (svg.width - 2 * x0 - gap * (count - 1)) / count
    xs = []
    for k, (title, lines) in enumerate(items):
        x = x0 + k * (w + gap)
        svg.box(x, y, w, h, lines, who, title=title, size=size)
        xs.append(x)
        if k:
            svg.arrow(x - gap + 2, y + h / 2, x - 2, y + h / 2)
    return xs


# ---------------------------------------------------------------- shared

def system_overview():
    s = Svg(1240, 700, "The whole system: from telescope images to candidate asteroids")
    s.box(30, 60, 150, 230, ["ZTF archive", "(IRSA)", "", "science image", "mask", "difference image", "PSF"], "grey", title="Data source")
    s.box(215, 60, 245, 230, ["query metadata, pick nights", "download 4 files per exposure", "load, mask bad pixels", "estimate background & noise", "detect blobs (2-sigma, SEP)", "split train / val / test BY NIGHT"],
          "aaryan", title="1. Data & preprocessing (Aaryan)")
    s.box(495, 60, 260, 230, ["cut 3 x 31 x 31 stamps per blob", "labels: planted fakes + known", "asteroids, rest = bogus", "normalise, augment, balance", "CNN scores p(real) per blob"],
          "paras", title="2. Labels, stamps, CNN (Paras)")
    s.box(790, 60, 260, 230, ["link blobs into straight tracks", "17 numbers per track", "XGBoost scores each track", "cross-match known asteroids"],
          "satvik", title="3. Linking & classifier (Satvik)")
    s.box(1085, 60, 130, 230, ["unknown tracks", "", "a person looks", "at the stamps"], "grey", title="Candidates")
    for x1, x2 in [(180, 215), (460, 495), (755, 790), (1050, 1085)]:
        s.arrow(x1, 175, x2, 175)
    s.box(215, 350, 835, 190, [], "advik", title="4. Evaluation & discovery engine (Advik)")
    s.box(235, 400, 190, 120, ["plant fakes, measure", "what is recovered", "(completeness)"], "advik", title="Scoreboard", size=12)
    s.box(445, 400, 190, 120, ["train / val / test", "+ untouched holdout", "versions v1..v5"], "advik", title="Honest testing", size=12)
    s.box(655, 400, 190, 120, ["add frames along a", "trial motion: +0.7 mag", "fainter than 1 frame"], "advik", title="Deep search", size=12)
    s.box(865, 400, 170, 120, ["watch every stage", "run live in a", "browser"], "advik", title="Live viewer", size=12)
    s.arrow(340, 350, 340, 292, "measures the stages above", label_dy=-4)
    s.arrow(920, 350, 920, 292)
    s.box(30, 350, 150, 190, ["archive + metadata", "cached on disk", "so every run is", "repeatable"], "grey", title="Provenance", size=12)
    s.note(620, 600, "Two ML models do the learning: a CNN on pixels (stage 2) and gradient-boosted trees on track numbers (stage 3).", size=13)
    s.note(620, 625, "Everything else is physics, search and statistics. Nothing is reported as a discovery without a person checking it.", size=13)
    s.note(620, 665, "Colour = owner of that part of the work.", size=12)
    return s


def split_by_night():
    s = Svg(1100, 470, "Splitting by NIGHT, decided before any modelling")
    s.text(550, 62, "Every detection, fake and stamp from one night lives in exactly one group (configs/acquire.yaml).", size=13, fill=MUTED)
    train = ["2020-09-19", "2020-09-21", "2020-09-23", "2021-09-11", "2022-09-22", "2022-09-24", "2023-09-20", "2023-09-25"]
    s.text(40, 110, "TRAIN (8 nights): the models learn from these", size=14, weight="bold", fill=COLOURS["aaryan"][0], anchor="start")
    for k, n in enumerate(train):
        s.box(40 + k * 128, 125, 118, 50, [n], "aaryan", size=12)
    s.text(40, 215, "VALIDATION (2 nights): choose settings and thresholds here", size=14, weight="bold", fill=COLOURS["paras"][0], anchor="start")
    for k, n in enumerate(["2021-09-12", "2023-09-15"]):
        s.box(40 + k * 128, 230, 118, 50, [n], "paras", size=12)
    s.text(40, 320, "TEST (1 night): opened once for the final number", size=14, weight="bold", fill=COLOURS["satvik"][0], anchor="start")
    s.box(40, 335, 160, 50, ["2019-02-08"], "satvik", size=12)
    s.note(230, 365, "the best night: sharp seeing, field at opposition, the most asteroids", anchor="start")
    s.text(640, 320, "HOLDOUT (1 night): never touched", size=14, weight="bold", fill=COLOURS["advik"][0], anchor="start")
    s.box(640, 335, 160, 50, ["2022-06-17"], "advik", size=12)
    s.note(550, 430, "Why by night? Frames of one night share stars, noise and artifacts. Random row splits would leak that into the test.", size=12)
    return s


# ---------------------------------------------------------------- Aaryan

def data_flow():
    s = Svg(1240, 420, "Aaryan: from the archive to files on disk")
    flow(s, 70, 130, [
        ("Ask first", ["metadata query:", "1,552 exposures of", "our quadrant", "(a few seconds)"]),
        ("Choose nights", ["only nights with", ">= 4 good r-band frames", "quality flag = 0", "write split to config"]),
        ("Download", ["4 files / exposure:", "science 38 MB, mask 19 MB,", "difference 7 MB, PSF 6 KB", "retry, resume, atomic"]),
        ("Define a UNIT", ["field + CCD + quadrant", "+ night", "e.g. 000618_c03_q4", "_2019-02-08"]),
        ("Load + clean", ["read FITS, mask bad pixels,", "reference = science - diff", "background + noise"]),
    ], who="aaryan", size=12, w=215)
    s.box(40, 250, 560, 120, ["220 files, 3.55 GB for the first 11 nights, 0 missing", "re-running skips files that exist (idempotent)", "a file appears only when complete: .part -> rename (atomic)",
                              "waits 20 s, 40 s, 60 s when the server says 503 (busy)"], "grey", title="Engineering rules for the downloader", size=12)
    s.box(640, 250, 560, 120, ["data/raw: never edited", "data/interim: cached known-asteroid lookups", "data/processed: detections, stamps (float16)", "configs/*.yaml: every setting, no numbers hidden in code"],
          "grey", title="Where things live", size=12)
    return s


def difference_imaging():
    s = Svg(1140, 470, "Aaryan: difference imaging, why only new things remain")
    s.box(40, 70, 250, 130, ["tonight's picture", "(stars + sky + asteroid)"], "aaryan", title="SCIENCE")
    s.text(318, 142, "-", size=44, weight="bold")
    s.box(345, 70, 250, 130, ["the same sky long ago", "(stars + sky, no asteroid)", "blurred to match tonight"], "grey", title="REFERENCE")
    s.text(623, 142, "=", size=44, weight="bold")
    s.box(650, 70, 250, 130, ["stars and sky cancel", "what is left is NEW"], "paras", title="DIFFERENCE")
    s.arrow(900, 135, 940, 135)
    s.box(940, 70, 170, 130, ["blobs to examine"], "satvik", title="Detect")
    s.box(40, 240, 1070, 190, [], "grey", title="What a difference image contains")
    rows = [("new or moving thing", "star-shaped POSITIVE blob (what we want)", "green"), ("faded or departed thing", "NEGATIVE blob (we ignore these)", "grey"),
            ("bright star, imperfect subtraction", "dipole: a bright half and a dark half side by side", "red"), ("cosmic ray", "sharp spike, narrower than a star", "red"),
            ("pure noise", "random speckle; 2-sigma blips pass the threshold by chance", "red")]
    for k, (a, b, who) in enumerate(rows):
        s.text(70, 300 + k * 26, a, size=13, anchor="start", weight="bold", fill=COLOURS[who][0])
        s.text(420, 300 + k * 26, b, size=13, anchor="start")
    return s


def preprocessing_funnel():
    s = Svg(1100, 470, "Aaryan: how many 'things' are left at each step (one training frame)")
    steps = [("1,950", "sources in the SCIENCE image", "grey", 880), ("334", "sources in the DIFFERENCE image at 5-sigma", "aaryan", 520),
             ("7,764", "blobs at 2-sigma (lenient on purpose: do not miss anything)", "paras", 880), ("173", "of those have SNR > 5", "paras", 420),
             ("~1,000", "real detections per night (planted fakes + known asteroids) among ~36,000 blobs", "satvik", 880)]
    for k, (n, label, who, w) in enumerate(steps):
        y = 65 + k * 70
        s.rect(40, y, w, 52, who)
        s.text(60, y + 33, n, size=22, weight="bold", anchor="start", fill=COLOURS[who][0])
        s.text(170, y + 32, label, size=13, anchor="start")
    s.note(550, 440, "Detection is tuned for COMPLETENESS (miss nothing). Classification, next, is tuned for PURITY (throw junk away).", size=13)
    return s


def leakage_types():
    s = Svg(1180, 560, "Aaryan: four ways a model can cheat, and what we found in our own first data")
    cols = [("1. Target leakage", "a feature contains the answer", "mag, true_rate existed only for rows WE made: 'is mag missing?' = the label", "feature must be measurable on new data"),
            ("2. Group leakage", "train and test share a source", "all tracklets came from the same 4 images", "split by NIGHT; check_split_leakage must be empty"),
            ("3. Shortcut learning", "classes differ in an easy, unreal way", "random bogus tracklets were >= 222 px off a line; real ones <= 2.6 px", "build hard negatives (near misses)"),
            ("4. Distribution shift", "deployment data differs from training", "test night is sharper and deeper than training nights", "expect and measure it; keep a holdout")]
    for k, (title, what, ours, rule) in enumerate(cols):
        x = 30 + k * 285
        s.box(x, 65, 270, 70, [what], "aaryan", title=title, size=12)
        s.box(x, 150, 270, 100, [ours], "grey", title="ours", size=11)
        s.box(x, 265, 270, 80, [rule], "green", title="rule", size=11)
    s.box(120, 380, 940, 150, [], "red", title="The experiment that proved it (same model, different test data)")
    for k, (a, b) in enumerate([("trained on easy legacy data -> tested on easy legacy data", "100%"), ("trained on easy legacy data -> tested on REAL near misses", "50%  (a coin flip)"),
                                ("trained on real + near misses -> tested on real + near misses", "99%")]):
        s.text(150, 440 + k * 30, a, size=13, anchor="start")
        s.text(1030, 440 + k * 30, b, size=14, weight="bold", anchor="end", fill=COLOURS["red"][0])
    return s


# ---------------------------------------------------------------- Paras

def label_sources():
    s = Svg(1180, 470, "Paras: where the labels come from (the sky is not labelled)")
    s.box(40, 70, 340, 190, ["plant fake asteroids with known", "brightness, speed and direction,", "using ZTF's own blur shape (PSF)", "added to science AND difference", "", "PERFECT labels; but only as real as the simulation"],
          "paras", title="1. Injection-recovery", size=12)
    s.box(420, 70, 340, 190, ["look up catalogued asteroids", "(JPL Horizons positions,", "SkyBoT as fallback)", "match to detections within 2 px", "", "REAL labels; fewer; catalogues miss faint ones"],
          "satvik", title="2. Known asteroids", size=12)
    s.box(800, 70, 340, 190, ["everything not matched", "", "includes variable stars, artifacts,", "and the rare real object nobody", "has catalogued yet", "", "a little LABEL NOISE we accept"], "grey", title="3. Bogus = the rest", size=12)
    s.arrow(210, 262, 210, 300)
    s.arrow(590, 262, 590, 300)
    s.arrow(970, 262, 970, 300)
    s.box(40, 300, 1100, 70, ["one training night: 1,059 real vs 36,161 bogus   (1 : 34)   -> a strongly imbalanced problem"], "paras", title="Result", size=13)
    s.box(40, 385, 1100, 70, ["Checks: chance-match rate estimated (1.15% per point -> 1.3 false matches per frame);  fakes vs real asteroids cannot be told apart (AUC 0.46-0.48)"],
          "green", title="Quality checks", size=12)
    return s


def triplet_stamp():
    s = Svg(1180, 430, "Paras: the CNN's input is three pictures of the same 31 x 31 pixels")
    s.text(430, 70, "science", size=14, weight="bold")
    s.text(560, 70, "reference", size=14, weight="bold")
    s.text(690, 70, "difference", size=14, weight="bold")
    rows = [("Real new object (asteroid)", "blob", "nothing", "blob", "green"), ("Bad subtraction of a bright star", "bright star", "bright star", "dipole / ring", "red"),
            ("Cosmic ray", "sharp spike", "nothing", "sharp spike", "red"), ("Noise", "nothing", "nothing", "faint speck", "red"), ("Edge of a masked area", "step / stripe", "step / stripe", "sharp edge", "red")]
    for k, (name, a, b, c, who) in enumerate(rows):
        y = 85 + k * 52
        s.text(40, y + 28, name, size=13, anchor="start", weight="bold", fill=COLOURS[who][0])
        for x, t in [(375, a), (505, b), (635, c)]:
            s.box(x, y, 110, 42, [t], "grey", size=11)
    s.box(850, 85, 300, 260, ["3 channels x 31 x 31 pixels", "", "each channel normalised:", "(value - median)", "/ (1.4826 x MAD)", "then clipped to [-10, 50]", "", "so '5' means 5 sigma", "on ANY night"], "paras", title="Input to the CNN", size=12)
    s.note(590, 395, "A person tells these apart by comparing the three panels. The CNN learns to do the same from examples.", size=13)
    return s


def cnn_architecture():
    s = Svg(1240, 520, "Paras: the real/bogus CNN (72,145 learnable numbers)")
    blocks = [("Input", ["3 x 31 x 31"], "grey", 0), ("Conv 3x3", ["16 filters", "+ ReLU", "448 params"], "paras", 0), ("Conv 3x3", ["16 filters", "+ ReLU", "2,320"], "paras", 0), ("MaxPool", ["31 -> 15"], "grey", 0),
              ("Conv 3x3 x2", ["32 filters", "+ ReLU", "4,640 + 9,248"], "paras", 0), ("MaxPool", ["15 -> 7"], "grey", 0), ("Conv 3x3", ["64 filters", "+ ReLU", "18,496"], "paras", 0), ("MaxPool", ["7 -> 3"], "grey", 0)]
    x = 30
    for k, (title, lines, who, _) in enumerate(blocks):
        s.box(x, 70, 130, 120, lines, who, title=title, size=11, title_size=13)
        if k:
            s.arrow(x - 18, 130, x - 2, 130)
        x += 148
    s.line(1190, 190, 1190, 250)
    s.line(1190, 250, 185, 250)
    s.arrow(185, 250, 185, 288)
    head = [("Flatten", ["64 x 3 x 3", "= 576"], "grey"), ("Dropout 0.3", ["switch off 30%", "of neurons while", "training"], "grey"), ("Dense 64", ["+ ReLU", "36,928 params"], "paras"), ("Dense 1", ["65 params", "-> one logit"], "paras"),
            ("Sigmoid", ["logit -> p(real)", "0 to 1"], "satvik")]
    x = 90
    for k, (title, lines, who) in enumerate(head):
        s.box(x, 290, 190, 110, lines, who, title=title, size=12)
        if k:
            s.arrow(x - 18, 345, x - 2, 345)
        x += 210
    s.box(90, 430, 1050, 70, ["Loss: binary cross-entropy (BCEWithLogits)   |   Optimiser: Adam, learning rate 0.001, batch 256   |   Augmentation: random 90-degree rotations + mirror",
                              "Early stopping: stop when validation average precision has not improved for 4 epochs; keep the best epoch (best: AP 0.899)"], "paras", title="Training recipe", size=11)
    return s


def cross_fitting():
    s = Svg(1180, 500, "Paras: cross-fitting, so the second model never sees over-confident scores")
    s.text(590, 62, "Problem: the tracklet model uses the CNN's scores as features. A CNN scoring its OWN training nights is over-confident.", size=13, fill=MUTED)
    for fold in range(4):
        y = 90 + fold * 62
        s.text(30, y + 30, f"fold {fold + 1}", size=13, weight="bold", anchor="start")
        for k in range(4):
            held = k == fold
            s.box(110 + k * 190, y, 175, 46, ["2 nights: SCORED" if held else "2 nights: train"], "satvik" if held else "paras", size=12)
        s.text(900, y + 30, "train CNN on the other 6 nights," if fold == 0 else "", size=12, anchor="start", fill=MUTED)
        s.text(900, y + 48, "score the 2 held-out nights" if fold == 0 else "", size=12, anchor="start", fill=MUTED)
    s.box(110, 350, 760, 56, ["validation and test nights are scored by the FINAL CNN (it never trained on them)"], "green", size=12)
    s.box(110, 425, 760, 56, ["result: every night's scores come from a model that never saw that night -> honest features for stage 3"], "green", size=12)
    s.note(1010, 400, "Folds split by NIGHT, never by detection.", size=12)
    return s


# ---------------------------------------------------------------- Satvik

def linking():
    s = Svg(1180, 500, "Satvik: linking = a smart search, not learning")
    s.text(590, 62, "5 frames x ~8,000 blobs: trying every combination = about 2 x 10^19 tracks. Impossible. So: seed, predict, check.", size=13, fill=MUTED)
    xs = [120, 290, 460, 630, 800]
    for k, x in enumerate(xs):
        s.rect(x - 60, 90, 120, 250, "grey", rx=6)
        s.text(x, 112, f"frame {k + 1}", size=12, weight="bold")
    pts = [(120, 300), (290, 262), (460, 224), (630, 186), (800, 148)]
    s.line(60, 322, 860, 124, COLOURS["satvik"][0], width=2, dashed=True)
    for k, (x, y) in enumerate(pts):
        s.circle(x, y, 7, COLOURS["satvik"][0], COLOURS["satvik"][1])
    for x, y in [(90, 160), (150, 230), (260, 300), (320, 150), (430, 290), (490, 130), (600, 250), (660, 300), (770, 220), (830, 270), (140, 140), (300, 205)]:
        s.circle(x, y, 4, MUTED)
    s.box(900, 90, 250, 250, ["1. SEED: two blobs in two", "frames at a plausible speed", "(8-120 arcsec/hour)", "", "2. PREDICT: that fixes a line,", "so look UP where the object", "must be in the other frames", "(KD-tree, within 2 px)", "", "3. KEEP: >= 3 of 5 frames,", "within 1.5 px of the line"], "satvik", title="How", size=11)
    s.box(40, 370, 540, 110, ["the CNN removes ~76% of blobs first: seed pairs 52,000 -> 3,500", "linking one night takes about 0.2 seconds", "minimum speed 8 arcsec/hour: slower 'movers' are just wobbling", "stationary artifacts"], "grey", title="Why it is fast", size=11)
    s.box(600, 370, 540, 110, ["the telescope POINTING SHIFTS between exposures (up to 12 px).", "Fix: put every blob on frame 1's sky grid (x_common, y_common)", "before linking. Real asteroids' straight-line error fell from", "~2.5 px to 0.05-0.47 px."], "red", title="The biggest bug we found", size=11)
    return s


def boosting():
    s = Svg(1180, 470, "Satvik: from one decision tree to gradient boosting (XGBoost)")
    s.box(40, 70, 320, 150, ["a single tree: a few yes/no", "questions, e.g. 'is the", "brightness steady?'", "", "readable, but one tree", "over- or under-fits"], "satvik", title="Decision tree", size=12)
    s.box(430, 70, 320, 150, ["hundreds of trees, each on a", "random sample of rows and", "features; AVERAGE their votes", "", "lower variance"], "satvik", title="Random forest (bagging)", size=12)
    s.box(820, 70, 320, 150, ["trees built ONE AFTER ANOTHER;", "each fits the ERRORS of the", "trees before it, scaled by a", "small learning rate", "", "usually the best on tables"], "satvik", title="Gradient boosting", size=12)
    s.arrow(362, 145, 428, 145)
    s.arrow(752, 145, 818, 145)
    s.box(40, 250, 1100, 90, ["300 trees, depth 3, learning rate 0.05, each tree sees 80% of rows and 80% of columns.   Settings are modest on purpose: only ~1,600 tracklets to learn from."],
          "grey", title="Our model (train_tracklet_classifier.py)", size=12)
    s.box(40, 360, 1100, 90, ["cross-validated by NIGHT (GroupKFold): logistic regression 0.999, tree 0.983, forest 0.999, XGBoost 0.999 (average precision).", "They TIE, so the choice is practical (handles missing values, interactions). A simpler model would also be defensible."],
          "green", title="Honest result", size=12)
    return s


def confusion_metrics():
    s = Svg(1180, 500, "Satvik: reading a confusion matrix (the logistic-regression baseline, validation nights)")
    s.box(60, 110, 200, 70, ["called real"], "grey", size=13)
    s.box(270, 110, 200, 70, ["called bogus"], "grey", size=13)
    s.text(40, 150, "", size=1)
    s.box(60, 190, 200, 90, ["1,547", "true positives"], "green", size=14)
    s.box(270, 190, 200, 90, ["582", "false negatives (missed)"], "red", size=13)
    s.box(60, 290, 200, 90, ["493", "false positives (alarms)"], "red", size=13)
    s.box(270, 290, 200, 90, ["64,419", "true negatives"], "green", size=14)
    s.text(40, 235, "truly real", size=12, anchor="end")
    s.text(40, 335, "truly bogus", size=12, anchor="end")
    f = ["recall = 1,547 / 2,129 = 72.7%", "(of all real objects, how many kept)", "", "precision = 1,547 / 2,040 = 75.8%", "(of everything flagged, how many real)", "",
         "false-positive rate = 493 / 64,912 = 0.76%", "", "accuracy = (1,547 + 64,419) / 67,041 = 98.4%"]
    for k, line in enumerate(f):
        s.text(520, 130 + k * 25, line, size=13, anchor="start")
    s.box(60, 410, 1060, 70, ["Accuracy paradox: 'always say bogus' scores 96.8% accuracy and finds nothing (recall 0%). With rare positives, accuracy only measures the common class.", "Use recall, precision and the precision-recall curve."],
          "satvik", title="Why we do not report accuracy alone", size=12)
    return s


def operating_point():
    s = Svg(1180, 470, "Satvik: choosing the operating point (fix recall first, then count the false alarms)")
    s.box(40, 70, 330, 140, ["a missed asteroid = a lost discovery", "a false alarm = one minute of a", "person's time", "", "so: target 90% recall, then", "minimise false alarms"], "satvik", title="The cost argument", size=12)
    s.box(410, 70, 330, 140, ["threshold chosen on the VALIDATION", "nights (never on test):", "'the cut that keeps 90% of real", "objects'"], "satvik", title="How we choose", size=12)
    s.box(780, 70, 360, 140, ["raw scores are inflated because", "training was balanced (16.7% real,", "truth 3.1%). Base-rate adjustment or", "isotonic calibration fixes the meaning,", "not the ranking"], "satvik", title="Do scores mean probabilities?", size=11)
    s.text(590, 250, "False alarms per frame at 90% recall (validation nights, 67,041 detections, 3.2% real)", size=14, weight="bold")
    for k, (label, n, who) in enumerate([("logistic regression on 6 numbers", 813, "grey"), ("CNN on pixels", 176, "paras")]):
        y = 275 + k * 70
        s.text(40, y + 36, label, size=13, anchor="start")
        s.rect(330, y, n / 813 * 640, 54, who)
        s.text(330 + n / 813 * 640 + 12, y + 35, str(n), size=18, weight="bold", anchor="start", fill=COLOURS[who][0])
    s.box(330, 415, 640, 40, ["4.6 times fewer false alarms at the same completeness  (AP 0.780 -> 0.899)"], "green", size=13)
    return s


# ---------------------------------------------------------------- Advik

def architecture_full():
    s = Svg(1300, 760, "Advik: the architecture in code (one purpose per file)")
    cols = [("acquire/ clean/", "aaryan", ["query_ztf_metadata", "download_ztf_images", "load_exposure", "estimate_background"]),
            ("detect/ dataset/ label/ simulate/", "paras", ["extract_sources", "cut_triplet_stamps", "plan_injections", "label_detections"]),
            ("models/", "paras", ["define_realbogus_cnn", "train_realbogus_cnn", "cross_fit_realbogus_scores", "save/load_model_bundle"]),
            ("link/ features/ crossmatch/", "satvik", ["link_detections_into_tracklets", "compute_tracklet_features", "train_tracklet_classifier", "crossmatch_known_asteroids"]),
            ("deep/ fast/", "advik", ["build_snr_image", "stack_and_find_peaks (GPU)", "find_streaks", "check_streak_consistency"]),
            ("evaluate/ pipeline/", "advik", ["build_scoreboard", "measure_completeness", "run_night_pipeline", "run_full_search"])]
    for k, (title, who, lines) in enumerate(cols):
        s.box(30 + k * 208, 70, 196, 200, lines, who, title=title, size=10, title_size=12)
    s.box(30, 300, 1240, 80, ["run_night_pipeline calls the stages in order: detect -> CNN -> link -> features -> classifier -> cross-match -> candidates.  The SAME function is used for evaluation (with planted fakes) and for the real search."],
          "advik", title="One deployed path", size=12)
    s.box(30, 400, 400, 130, ["configs/*.yaml: every setting", "the night split, thresholds, sizes", "tests/: 150+ tests, all offline", "environment.yml: exact software"], "grey", title="Reproducibility", size=12)
    s.box(450, 400, 400, 130, ["models/v1 ... v5: a version =", "CNN + tracklet model + thresholds", "+ the commit and nights used,", "never overwritten"], "grey", title="Model bundles (a mini registry)", size=12)
    s.box(870, 400, 400, 130, ["scripts/live_viewer.py runs the", "real functions one stage at a time", "and shows the pictures in a browser", "(src/asthunt/viewer/)"], "advik", title="Live viewer", size=12)
    s.box(30, 555, 1240, 150, ["1. All numbers in the README come from reports/scoreboard.csv, produced by ONE function (no notebook computes its own).",
                               "2. The holdout night is opened in one place only, to write one number.",
                               "3. A model is compared at the SAME false-alarm level, never at its own chosen threshold.",
                               "4. Quick fixes were refused when a measurement pointed to a different cause (e.g. a linking bug, not a model problem)."], "advik", title="Engineering habits that make the results trustworthy", size=12)
    return s


def shift_and_stack():
    s = Svg(1240, 520, "Advik: shift-and-stack, how we see what no single frame shows")
    s.text(620, 62, "A faint asteroid is too weak in any one 30-second frame. If we shift the frames so the asteroid sits on the same spot and ADD them, it adds up; the noise does not.", size=12, fill=MUTED)
    for k in range(8):
        x = 40 + k * 100
        s.rect(x, 90, 86, 86, "grey", rx=6)
        s.circle(x + 14 + k * 8, 150 - k * 8, 5, COLOURS["advik"][0], COLOURS["advik"][1])
        s.text(x + 43, 192, f"frame {k + 1}", size=11)
    s.arrow(430, 205, 430, 245, "shift each frame by the trial motion", label_dy=4, label_dx=130)
    for k in range(8):
        x = 40 + k * 100
        s.rect(x, 250, 86, 86, "grey", rx=6)
        s.circle(x + 43, 293, 5, COLOURS["advik"][0], COLOURS["advik"][1])
    s.arrow(430, 345, 430, 385, "add them up", label_dy=4, label_dx=60)
    s.box(250, 390, 360, 100, ["signal adds N times, noise only sqrt(N) times", "8 frames: gain 8 / 2.8 = 2.8x = +1.1 magnitudes"], "advik", title="Result", size=12)
    s.box(850, 90, 340, 190, ["We do not know the motion, so we", "TRY 4,888 trial motions", "(every direction, up to 120 arcsec/hour)", "on the GPU and keep the best", "", "Picking the best of 4,888 = the", "LOOK-ELSEWHERE effect: noise alone", "reaches 5 sigma, so the cut is 6 sigma"], "advik", title="Search over motions", size=11)
    s.box(850, 300, 340, 190, ["real difference images are NOT Gaussian:", "clip each frame at 5 sigma; remove the", "night's median; standardise every pixel;", "mask what single frames already see", "", "peaks per stack: 47,550 -> 94", "null test (times shuffled): 0 false tracks"], "green", title="Making it honest", size=11)
    return s


def scoreboard_versions():
    s = Svg(1180, 520, "Advik: how deep can we see? (50%-completeness magnitude on the test night 2019-02-08)")
    s.text(590, 62, "Larger magnitude = fainter. Completeness = fraction of planted fakes recovered.", size=12, fill=MUTED)
    bars = [("v1: single frames, first system", 20.51, "grey"), ("v4: relaxed linking + physics features", 21.41, "paras"), ("deep search (shift-and-stack)", 22.12, "advik")]
    base, scale = 19.0, 330.0
    for k, (label, mag, who) in enumerate(bars):
        y = 100 + k * 90
        s.text(40, y + 30, label, size=14, anchor="start")
        s.rect(430, y, (mag - base) * scale / 3.2 * 1.0, 56, who)
        s.text(430 + (mag - base) * scale / 3.2 + 12, y + 36, f"{mag:.2f}", size=20, weight="bold", anchor="start", fill=COLOURS[who][0])
    s.text(430, 82, "axis starts at magnitude 19", size=11, anchor="start", fill=MUTED)
    s.box(40, 380, 520, 110, ["v1 -> v4: found where asteroids were lost (linking), not by a bigger model.", "A multi-frame CNN TIED XGBoost: kept the simpler model."], "grey", title="Lesson 1", size=12)
    s.box(600, 380, 540, 110, ["v5 (3x more training data) did NOT help at equal false alarms.", "The limit is the photons in one 30 s frame, not the model."], "grey", title="Lesson 2", size=12)
    return s


def evaluation_protocol():
    s = Svg(1180, 500, "Advik: the evaluation protocol (what makes a number believable)")
    flow(s, 70, 140, [("1. Freeze", ["models, thresholds, configs", "saved as a version bundle"]), ("2. Choose on VALIDATION", ["thresholds are set here,", "never on test"]),
                      ("3. Plant fakes", ["known truth: 300 per night,", "real PSF, real noise,", "moving on the SKY"]), ("4. Run the DEPLOYED path", ["run_night_pipeline,", "not a notebook copy"]),
                      ("5. One function", ["build_scoreboard", "-> reports/scoreboard.csv"])], who="advik", size=12, w=205)
    s.box(40, 250, 540, 120, ["compare models at the SAME false-alarm level", "(v5 looked 0.8 mag worse at its own threshold, but tied at equal alarms)",
                              "use metrics with headroom: tracklet AP 0.999 cannot show improvement; depth and known-recovery can"], "advik", title="Comparing fairly", size=11)
    s.box(600, 250, 540, 120, ["real asteroids and planted fakes both fade out between 20.5 and 21.5", "-> the simulation is realistic", "fake-vs-real classifier: AUC 0.47 (0.5 = cannot tell apart)"],
          "green", title="Checking the simulation against reality", size=11)
    s.box(40, 390, 1100, 90, ["Honesty note: we looked at the test night while fixing real bugs (pointing shift, units). Fixes were physics, not score-tuning, but strictly the test night is no longer untouched.", "That is why a separate HOLDOUT night (2022-06-17) is opened only by the scoreboard."], "red", title="Limits we state openly", size=11)
    return s


def live_viewer_flow():
    s = Svg(1180, 430, "Advik: how the live viewer works")
    flow(s, 70, 150, [("Browser page", ["pick a night, press Start", "polls the state every 0.8 s", "shows pictures + numbers", "Next / Stop buttons"]),
                      ("Small web server", ["serve_live_viewer.py", "(Python standard library)", "/api/state, /img/..", "/api/start, /api/next"]),
                      ("Shared state", ["live_state.py", "stages, numbers, pictures", "a gate that waits for", "'Next' in step mode"]),
                      ("Traced run", ["run_traced_night.py", "calls the REAL functions", "in the real order and", "reports each stage"])], who="advik", size=12, w=245)
    s.box(40, 250, 1100, 80, ["Tested: on a real night the traced run gives the SAME detections, scores, tracks and candidates as run_night_pipeline (maximum difference 0.0).",
                              "'fakes' plants fake asteroids first, so you can watch them found (bright) or missed (too faint for a single frame)."], "green", title="Why you can trust what it shows", size=11)
    s.box(40, 345, 1100, 70, ["12 stages: pick night -> open files -> noise -> plant fakes -> find blobs -> stamps -> CNN -> link -> features -> classifier -> known? -> result"], "grey", size=12)
    return s


def cnn_results():
    s = Svg(1180, 430, "Paras: CNN versus the simple baseline")
    s.text(590, 62, "Average precision (AP): one number for rare positives; 1.0 is perfect. Same 67,041 validation detections (3.2% real).", size=12, fill=MUTED)
    bars = [("logistic regression, 6 numbers (validation)", 0.780, "grey"), ("CNN on pixels (validation)", 0.899, "paras"), ("CNN on the untouched test night", 0.938, "satvik")]
    for k, (label, ap, who) in enumerate(bars):
        y = 90 + k * 70
        s.text(40, y + 34, label, size=13, anchor="start")
        s.rect(430, y, ap * 600, 52, who)
        s.text(430 + ap * 600 + 12, y + 34, f"{ap:.3f}", size=20, weight="bold", anchor="start", fill=COLOURS[who][0])
    s.box(40, 320, 540, 90, ["at 90% recall: 813 -> 176 false alarms per frame (4.6x fewer)", "precision at 90% recall: 0.191 -> 0.522"], "paras", title="Same completeness, far fewer false alarms", size=12)
    s.box(600, 320, 540, 90, ["the CNN's gain is PURITY: only ~24% of detections remain,", "so linking seed pairs fall from 52,000 to 3,500"], "green", title="Why it matters for the next stage", size=12)
    return s


def build_all():
    out = ROOT
    diagrams = {
        "00_Start_Here/diagrams/system_overview.svg": system_overview, "00_Start_Here/diagrams/split_by_night.svg": split_by_night,
        "02_Aaryan/diagrams/data_flow.svg": data_flow, "02_Aaryan/diagrams/difference_imaging.svg": difference_imaging,
        "02_Aaryan/diagrams/preprocessing_funnel.svg": preprocessing_funnel, "02_Aaryan/diagrams/leakage_types.svg": leakage_types,
        "03_Paras/diagrams/label_sources.svg": label_sources, "03_Paras/diagrams/triplet_stamp.svg": triplet_stamp,
        "03_Paras/diagrams/cnn_architecture.svg": cnn_architecture, "03_Paras/diagrams/cnn_results.svg": cnn_results, "03_Paras/diagrams/cross_fitting.svg": cross_fitting,
        "04_Satvik/diagrams/linking.svg": linking, "04_Satvik/diagrams/boosting.svg": boosting,
        "04_Satvik/diagrams/confusion_metrics.svg": confusion_metrics, "04_Satvik/diagrams/operating_point.svg": operating_point,
        "01_Advik/diagrams/architecture_full.svg": architecture_full, "01_Advik/diagrams/shift_and_stack.svg": shift_and_stack,
        "01_Advik/diagrams/scoreboard_versions.svg": scoreboard_versions, "01_Advik/diagrams/evaluation_protocol.svg": evaluation_protocol,
        "01_Advik/diagrams/live_viewer_flow.svg": live_viewer_flow,
    }
    for path, make in diagrams.items():
        make().save(out / path)
        print("wrote", path)


if __name__ == "__main__":
    build_all()
