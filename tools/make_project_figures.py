"""Draw the project's headline pictures as SVG (and PNG copies for the Word reports).

    00_Start_Here/00_How_Our_Model_Works.svg     the CNN, drawn like the classic "CNN" poster, with real stamps
    0X_<Name>/00_Process_Diagram_<Name>.svg      how each person did their part, step by step

Needs: numpy, Pillow (any env) and the exported web data (tools/export_web_data.py) for the real stamps.
PNG copies (these two plus every team diagram) are rendered with headless Google Chrome into tools/figures_png/,
which tools/make_reports.js puts into the Word reports.

    python tools/make_project_figures.py
"""

import base64
import io
import json
import subprocess
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "public" / "data"
PNG_DIR = ROOT / "tools" / "figures_png"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAPER, INK, MUTED, LINE = "#f1efea", "#0e0e10", "#6b6a66", "#c9c5bb"
EMBER, STEEL, GOOD, BLUE = "#ff5a1f", "#16202c", "#0f9d6b", "#2f6fd6"
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', Menlo, monospace"


def text(x, y, s, size=14, weight=400, fill=INK, anchor="start", family=FONT, extra=""):
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'


def stamp_png(channel):
    """One normalised stamp channel -> base64 PNG (grey, sky dark, objects bright)."""
    scaled = np.clip((channel + 3) / 18, 0, 1)
    image = Image.fromarray((255 * np.sqrt(scaled)).astype(np.uint8)).resize((124, 124), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_stamps():
    meta = json.loads((DATA / "stamps.json").read_text())
    values = np.fromfile(DATA / "stamps.bin", dtype=np.float32).reshape(len(meta), 3, 31, 31)
    return meta, values


def slab_stack(x, y, w, h, n, dx, dy, fill, stroke=INK):
    """n stacked feature maps, drawn back to front."""
    out = []
    for i in reversed(range(n)):
        out.append(f'<rect x="{x + i * dx}" y="{y - i * dy}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    return "".join(out)


def bracket(x1, x2, y, label):
    return (f'<path d="M{x1} {y - 10} V{y} H{x2} V{y - 10}" fill="none" stroke="{INK}" stroke-width="1.2"/>'
            + text((x1 + x2) / 2, y + 26, label, 17, 600, anchor="middle"))


def model_diagram():
    meta, values = load_stamps()
    real = next(i for i, m in enumerate(meta) if m["kind"] == "known" and m["score_python"] > 0.999)
    W, H = 1680, 1010
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
         text(W / 2, 58, "How our model works: the real/bogus CNN", 34, 800, anchor="middle"),
         text(W / 2, 90, "Asteroid Hunter · ZTF difference images · one small convolutional neural network decides if a blob is a real moving object", 16, 400, MUTED, "middle")]

    # Input: three stacked stamps (real pixels from the test night)
    s.append(text(70, 150, "Input", 24, 700))
    s.append(text(70, 172, "one blob = 3 pictures, 31 x 31 px", 13, 400, MUTED))
    names = ["science (tonight)", "reference (old sky)", "difference (new - old)"]
    for c in reversed(range(3)):
        x, y = 60 + c * 50, 200 + c * 50
        s.append(f'<image href="{stamp_png(values[real][c])}" x="{x}" y="{y}" width="140" height="140" style="image-rendering:pixelated"/>')
        s.append(f'<rect x="{x}" y="{y}" width="140" height="140" fill="none" stroke="{INK}" stroke-width="1.5"/>')
        s.append(text(x + 144, y + 12, f"ch{c + 1}", 11, 700, EMBER, family=MONO))
    for c, name in enumerate(names):
        s.append(text(60, 590 + c * 20, f"ch{c + 1}  {name}", 12.5, 500, INK, family=MONO))
    s.append(f'<rect x="{60 + 100 + 30}" y="{300 + 30}" width="24" height="24" fill="none" stroke="#f4c20d" stroke-width="3"/>')
    s.append(f'<path d="M{60 + 100 + 42} {300 + 140 + 2} V500" stroke="{INK}" stroke-width="1.4" marker-end="url(#arrow)"/>')
    s.append(text(60 + 100 + 42, 525, "Kernel 3 x 3", 16, 600, anchor="middle"))
    s.append(f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>')

    # Feature extraction: three conv blocks
    blocks = [(330, 220, 150, 6, "16 maps 31x31", "Conv 3x3 x2 + ReLU", "Pooling -> 15x15"),
              (560, 250, 110, 8, "32 maps 15x15", "Conv 3x3 x2 + ReLU", "Pooling -> 7x7"),
              (760, 275, 80, 11, "64 maps 7x7", "Conv 3x3 + ReLU", "Pooling -> 3x3")]
    fills = ["#d9c7a8", "#d3bf9c", "#ccb690"]
    for (x, y, size, n, maps, conv, pool), fill in zip(blocks, fills):
        s.append(text(x + size / 2 + n * 3, y - 40 - n * 4, pool, 15, 600, anchor="middle"))
        s.append(slab_stack(x, y, size, size, n, 6, 6, fill))
        s.append(f'<rect x="{x + size * 0.45}" y="{y + size * 0.3}" width="{size * 0.16}" height="{size * 0.16}" fill="none" stroke="#f4c20d" stroke-width="2.5"/>')
        s.append(text(x + size / 2, y + size + 40, conv.split(" + ")[0], 15.5, 600, anchor="middle"))
        s.append(text(x + size / 2, y + size + 60, "+ ReLU", 15.5, 600, anchor="middle"))
        s.append(text(x + size / 2, y + size + 82, maps, 12.5, 400, MUTED, "middle", MONO))
    s.append(slab_stack(930, 300, 40, 40, 14, 4, 4, "#c4ac82"))
    s.append(text(980, 395, "64 x 3 x 3", 12.5, 400, MUTED, "middle", MONO))
    for (x1, y1), (x2, y2) in [((60 + 100 + 66, 300 + 42), (330 + 60, 220 + 50)), ((330 + 80, 220 + 60), (560 + 50, 250 + 40)),
                               ((560 + 60, 250 + 50), (760 + 35, 275 + 30)), ((760 + 45, 275 + 40), (930 + 20, 300 + 10))]:
        s.append(f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{INK}" stroke-dasharray="4 4" stroke-width="1"/>')
    s.append(f'<path d="M330 640 H950" stroke="{INK}" stroke-width="1.2" marker-end="url(#arrow)" marker-start="url(#arrow)"/>')
    s.append(f'<rect x="560" y="628" width="160" height="24" fill="{PAPER}"/>' + text(640, 646, "Feature Maps", 17, 500, anchor="middle"))

    # Flatten + dense layers
    fx = 1040
    for i in range(14):
        s.append(f'<rect x="{fx}" y="{200 + i * 22}" width="26" height="22" fill="#d9c7a8" stroke="{INK}"/>')
    s.append(f'<path d="M{930 + 96} {300 - 50} L{fx} 200 M{930 + 96} {340} L{fx} {200 + 14 * 22}" stroke="{INK}" stroke-dasharray="4 4"/>')
    s.append(text(fx + 13, 540, "Flatten", 15.5, 600, anchor="middle"))
    s.append(text(fx + 13, 560, "576 numbers", 12.5, 400, MUTED, "middle", MONO))
    s.append(text(fx + 13, 580, "Dropout 0.3", 12.5, 400, EMBER, "middle", MONO))
    layer_a = [(1130, 190 + i * 30) for i in range(12)]
    layer_b = [(1250, 235 + i * 30) for i in range(9)]
    out_node = (1360, 355)
    lines = []
    for a in layer_a:
        lines += [f'<line x1="{fx + 26}" y1="{y}" x2="{a[0]}" y2="{a[1]}" stroke="{INK}" stroke-opacity="0.12"/>' for y in range(211, 500, 44)]
        lines += [f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{INK}" stroke-opacity="0.35"/>' for b in layer_b]
    lines += [f'<line x1="{b[0]}" y1="{b[1]}" x2="{out_node[0]}" y2="{out_node[1]}" stroke="{INK}" stroke-opacity="0.6"/>' for b in layer_b]
    s += lines
    s += [f'<circle cx="{x}" cy="{y}" r="9" fill="#7fb6e6"/>' for x, y in layer_a]
    s += [f'<circle cx="{x}" cy="{y}" r="9" fill="#3c86c9"/>' for x, y in layer_b]
    s.append(f'<circle cx="{out_node[0]}" cy="{out_node[1]}" r="11" fill="{STEEL}"/>')
    s.append(text(1190, 560, "Fully Connected", 15.5, 600, anchor="middle"))
    s.append(text(1190, 580, "64 neurons + ReLU -> 1", 12.5, 400, MUTED, "middle", MONO))

    # Output
    score = meta[real]["score_python"]
    s.append(text(1530, 150, "Output", 24, 700, anchor="middle"))
    s.append(f'<rect x="1430" y="290" width="200" height="130" fill="{EMBER}" rx="3"/>')
    s.append(text(1530, 330, f"Real  {score:.2f}", 22, 700, "#fff", "middle", MONO))
    s.append(text(1530, 370, f"Bogus {1 - score:.2f}", 22, 700, "#fff", "middle", MONO))
    s.append(text(1530, 402, f"known asteroid {meta[real]['name'] or ''}", 11, 500, "#fff", "middle", MONO))
    s.append(f'<path d="M{out_node[0] + 11} {out_node[1]} H1428" stroke="{INK}" stroke-dasharray="4 4"/>')
    s.append(text(1530, 455, "Sigmoid", 16, 600, anchor="middle"))
    s.append(text(1530, 476, "turns the score into", 13, 400, MUTED, "middle"))
    s.append(text(1530, 494, "p(real) from 0 to 1", 13, 400, MUTED, "middle"))

    s.append(bracket(40, 1010, 700, "Feature Extraction"))
    s.append(bracket(1030, 1390, 700, "Classification"))
    s.append(bracket(1420, 1640, 700, "Probability"))

    # Where the CNN sits in the whole pipeline
    s.append(f'<rect x="40" y="770" width="{W - 80}" height="200" fill="#fff" stroke="{LINE}"/>')
    s.append(text(64, 802, "WHERE THIS MODEL SITS IN THE WHOLE PIPELINE", 13, 700, MUTED, extra='letter-spacing="1.5"'))
    steps = [("1", "Photos", "5 frames of one patch", "Aaryan"), ("2", "Difference", "new minus old sky", "Aaryan"),
             ("3", "Blobs", "41,522 on the test night", "Aaryan"), ("4", "CNN", "p(real) for every blob", "Paras"),
             ("5", "Link", "straight-line tracks", "Satvik"), ("6", "XGBoost", "17 numbers -> track score", "Satvik"),
             ("7", "Known?", "catalogue check", "Satvik"), ("8", "Measure", "scoreboard, deep search", "Advik")]
    bw, gap = 172, 22
    for i, (n, title, sub, who) in enumerate(steps):
        x = 64 + i * (bw + gap)
        hot = title == "CNN"
        s.append(f'<rect x="{x}" y="822" width="{bw}" height="118" fill="{EMBER if hot else PAPER}" stroke="{EMBER if hot else LINE}"/>')
        c = "#fff" if hot else INK
        s.append(text(x + 14, 850, n, 13, 700, c if hot else EMBER, family=MONO))
        s.append(text(x + 14, 880, title, 20, 700, c))
        s.append(text(x + 14, 904, sub, 12, 400, c if hot else MUTED))
        s.append(text(x + 14, 926, who, 11.5, 600, c if hot else MUTED, family=MONO))
        if i < len(steps) - 1:
            s.append(f'<path d="M{x + bw + 3} 881 H{x + bw + gap - 3}" stroke="{INK}" stroke-width="1.3" marker-end="url(#arrow)"/>')
    s.append(text(W - 40, 995, "72,145 learnable weights · trained on 53,178 stamps · average precision 0.899 (validation) and 0.938 (untouched test night) · baseline 0.780",
                  12.5, 500, MUTED, "end", MONO))
    s.append("</svg>")
    return "\n".join(s)


PEOPLE = {
    "Aaryan": ("02_Aaryan", "Problem, data and preprocessing", [
        ("Define the problem", "find moving dots in ZTF difference images; success = depth + known asteroids found", "docs/project_setup.md"),
        ("Ask the archive first", "metadata query: 1,552 exposures, 16.4% flagged, 40 usable nights", "acquire/query_ztf_metadata.py"),
        ("Split by night, early", "8 train / 2 validation / 1 test / 1 holdout night, fixed in a config", "configs/acquire.yaml"),
        ("Download 4 files each", "science, mask, difference, PSF: 220 files, 3.55 GB", "acquire/download_ztf_images.py"),
        ("Clean every picture", "bad-pixel mask, reference = science - difference, noise level", "clean/load_exposure.py"),
        ("Find the blobs", "2 sigma, 5 px minimum: about 7,800 blobs per frame", "detect/extract_sources.py"),
        ("Audit for leakage", "the first dataset cheated: 100% -> 50% -> 99%; rebuilt", "evaluate/check_split_leakage.py"),
    ], "Hands Paras a clean blob table and a leak-free split by night"),
    "Paras": ("03_Paras", "Labels, dataset and the CNN", [
        ("Plant fake asteroids", "draw fakes with ZTF's own blur shape: 300 per night", "simulate/plant_fakes_in_exposure.py"),
        ("Label every blob", "fake or known asteroid within 2 px = real; the rest = bogus", "label/label_detections.py"),
        ("Cut 3 x 31 x 31 stamps", "science, reference and difference around each blob", "detect/cut_triplet_stamps.py"),
        ("Prepare the data", "normalise, flip/rotate, balance: 53,178 stamps, 16.7% real", "dataset/assemble_stamp_dataset.py"),
        ("Build + train the CNN", "72,145 weights, Adam lr 0.001, batch 256, early stopping", "models/train_realbogus_cnn.py"),
        ("Measure it", "AP 0.899 vs 0.780 baseline; 176 vs 813 false alarms per frame", "docs/realbogus_cnn.md"),
        ("Learn from mistakes", "confident errors were real asteroids: catalogue 2.9\" off -> Horizons", "label/refine_known_positions.py"),
        ("Cross-fit the scores", "each night scored by a model that never saw it", "models/cross_fit_realbogus_scores.py"),
    ], "Hands Satvik every blob with a CNN score and a threshold"),
    "Satvik": ("04_Satvik", "Linking, tracklet classifier and metrics", [
        ("Keep the CNN's blobs", "drop blobs under the CNN cut (set low, so almost no real blob is lost)", "models/predict_realbogus.py"),
        ("One shared sky grid", "the telescope points slightly differently each time: correct for it", "link/measure_sky_motion.py"),
        ("Link into tracks", "pairs + KD-tree look-up + straight-line fit: 52,000 -> 3,500 seeds", "link/link_detections_into_tracklets.py"),
        ("17 numbers per track", "speed, straightness, steady brightness, CNN opinion, forced photometry", "features/compute_tracklet_features.py"),
        ("Compare 4 models", "logistic regression, tree, forest, XGBoost; cross-validated by night", "models/train_tracklet_classifier.py"),
        ("Pick the threshold", "on validation nights only, at a fixed recall", "evaluate/choose_threshold.py"),
        ("Use the right metrics", "accuracy lies (96.8% for 'always bogus'); use precision, recall, AP", "evaluate/compute_classification_metrics.py"),
        ("Known or unknown?", "match tracks to the asteroid catalogue; checklist for a human", "crossmatch/crossmatch_known_asteroids.py"),
    ], "Hands Advik scored tracks, each marked known or unknown"),
    "Advik": ("01_Advik", "System, evaluation, deep search and live viewer (lead)", [
        ("Design the system", "split into 4 stages with clear hand-overs; one deployed code path", "pipeline/run_night_pipeline.py"),
        ("Set the test rules", "plant fakes on nights nobody tuned on; compare at equal false alarms", "evaluate/measure_completeness.py"),
        ("Build the scoreboard", "one table for every version v1 -> v5; every number comes from here", "evaluate/build_scoreboard.py"),
        ("Find where we lose", "asteroids were lost at linking, not the classifier: fixed (v2, v4)", "docs/scoreboard.md"),
        ("Test bigger ideas fairly", "multi-frame CNN tied XGBoost (0.993 vs 0.992); 3x data did not help", "docs/multiframe_model.md"),
        ("Deep search", "add frames along 4,888 trial motions; 6 sigma; null test 0 false: 22.1 mag", "deep/stack_and_find_peaks.py"),
        ("Live viewer + demo", "runs the real pipeline stage by stage in the browser", "scripts/live_viewer.py"),
        ("Make it reproducible", "versioned model bundles, configs, 166 offline tests", "models/save_model_bundle.py"),
    ], "Result: depth 20.5 -> 21.4 -> 22.1 mag; 31 known asteroids recovered; no false discovery claimed"),
}


def process_diagram(name):
    folder, role, steps, handover = PEOPLE[name]
    per_row = 4
    rows = (len(steps) + per_row - 1) // per_row
    bw, bh, gx, gy = 330, 168, 40, 56
    W = 60 + per_row * bw + (per_row - 1) * gx + 60
    H = 170 + rows * bh + (rows - 1) * gy + 150
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
         f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{INK}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
         text(60, 64, f"{name}: how I did my part", 32, 800),
         text(60, 96, role + " · follow the arrows", 16, 400, MUTED)]
    centres = []
    for i, (title, body, code) in enumerate(steps):
        r, c = divmod(i, per_row)
        if r % 2:                                                    # snake: second row runs right to left
            c = per_row - 1 - c
        x, y = 60 + c * (bw + gx), 140 + r * (bh + gy)
        centres.append((x, y))
        s.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="#fff" stroke="{LINE}"/>')
        s.append(f'<rect x="{x}" y="{y}" width="6" height="{bh}" fill="{EMBER}"/>')
        s.append(text(x + 24, y + 36, f"STEP {i + 1:02d}", 12, 700, EMBER, family=MONO, extra='letter-spacing="1.5"'))
        s.append(text(x + 24, y + 66, title, 20, 700))
        words, line, ly = body.split(), "", y + 94
        for word in words:
            if len(line + " " + word) > 38:
                s.append(text(x + 24, ly, line.strip(), 14, 400, "#3d3d39"))
                line, ly = "", ly + 20
            line += " " + word
        s.append(text(x + 24, ly, line.strip(), 14, 400, "#3d3d39"))
        s.append(text(x + 24, y + bh - 16, code, 11.5, 500, MUTED, family=MONO))
    for i in range(len(centres) - 1):
        (x1, y1), (x2, y2) = centres[i], centres[i + 1]
        if y1 == y2:
            a, b = (x1 + bw, x2) if x2 > x1 else (x1, x2 + bw)
            s.append(f'<path d="M{a + 4} {y1 + bh / 2} H{b - 4}" stroke="{INK}" stroke-width="1.5" marker-end="url(#arrow)"/>')
        else:
            s.append(f'<path d="M{x1 + bw / 2} {y1 + bh + 4} V{y2 - 4}" stroke="{INK}" stroke-width="1.5" marker-end="url(#arrow)"/>')
    hy = 140 + rows * bh + (rows - 1) * gy + 40
    s.append(f'<rect x="60" y="{hy}" width="{W - 120}" height="70" fill="{STEEL}"/>')
    s.append(text(84, hy + 30, "OUTPUT", 12, 700, EMBER, family=MONO, extra='letter-spacing="1.5"'))
    s.append(text(84, hy + 54, handover, 18, 600, "#fff"))
    s.append("</svg>")
    return "\n".join(s), W, H


def write_readme(name):
    """Each person's folder gets a README: the steps of their part, then what every file in the folder is."""
    folder, role, steps, handover = PEOPLE[name]
    rows = "\n".join(f"| {i} | **{title}** | {body} | `project/src/asthunt/{code}` |".replace("`project/src/asthunt/docs/", "`project/docs/")
                      .replace("`project/src/asthunt/configs/", "`project/configs/").replace("`project/src/asthunt/scripts/", "`project/scripts/")
                      for i, (title, body, code) in enumerate(steps, 1))
    extra = "| `08_Demo_Screenshots/` | one saved picture per live-viewer stage, in case the live demo fails |\n" if name == "Advik" else ""
    return f"""# {name}: {role}

**How I did my part, step by step** (the same steps are drawn in `00_Process_Diagram_{name}.svg`).

| Step | What | How, in one line | Main code |
|---|---|---|---|
{rows}

**Output:** {handover}.

## What is in this folder (read in this order)
| File | What it is |
|---|---|
| `00_Process_Diagram_{name}.svg` | the steps above as one picture (open in a browser) |
| `01_Part_Brief.md` | what is mine, what I hand over, where the code is, numbers to know |
| `02_Knowledge_Pack.md` | everything I must understand, in detail |
| `03_Presentation_Guide.md` | slide by slide: what to say and for how long |
| `04_Viva_QA.md` | questions the examiner may ask, with model answers |
| `05_Handbook_{name}.html` / `.pdf` | files 01 to 04 plus all diagrams in one printable file |
| `06_Report_{name}.docx` | my part written up from start to end in simple words, with results |
| `07_Diagrams/` | pictures for my slides, numbered in the order of the steps |
{extra}
Start with `../00_Start_Here/` (the whole project), then this folder. The runnable code is in `../project/`; the web demo is in `../web/`.
"""


def to_png(svg_path, png_path, width, height):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={width},{height}",
                    "--force-device-scale-factor=2", f"--screenshot={png_path}", svg_path.as_uri()],
                   check=True, capture_output=True)


def render_team_diagrams():
    """PNG copies of every diagram in the 07_Diagrams / 11_Diagrams folders, for the Word reports."""
    import re
    for svg in sorted(ROOT.glob("0*/*_Diagrams/*.svg")):
        size = re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"', svg.read_text()[:600])
        w, h = (int(float(size.group(1))), int(float(size.group(2)))) if size else (1200, 700)
        page = PNG_DIR / "page.html"
        page.write_text(f'<html><body style="margin:0;background:#fff"><img src="{svg.resolve().as_uri()}" width="{w}" height="{h}"></body></html>')
        to_png(page, PNG_DIR / f"{svg.stem}.png", w, h)
    (PNG_DIR / "page.html").unlink(missing_ok=True)


def main():
    PNG_DIR.mkdir(exist_ok=True)
    model = ROOT / "00_Start_Here" / "00_How_Our_Model_Works.svg"
    model.write_text(model_diagram())
    to_png(model, PNG_DIR / "model.png", 1680, 1010)
    for name, (folder, *_rest) in PEOPLE.items():
        svg, w, h = process_diagram(name)
        path = ROOT / folder / f"00_Process_Diagram_{name}.svg"
        path.write_text(svg)
        to_png(path, PNG_DIR / f"process_{name}.png", w, h)
        (ROOT / folder / "README.md").write_text(write_readme(name))
    render_team_diagrams()
    print("written", model, "and 4 process diagrams; PNGs in", PNG_DIR)


if __name__ == "__main__":
    main()
