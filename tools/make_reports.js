// Build the five Word reports: one for the whole group and one per person.
//
//   cd tools && npm install && node make_reports.js
//
// Needs the PNG figures from `python tools/make_project_figures.py` (tools/figures_png/).
// Every number here comes from the project's own documents (00_Start_Here, the knowledge packs,
// project/docs) or from project/reports/scoreboard.csv; the comment next to a table says which.

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, ImageRun, LevelFormat, Packer, PageBreak,
  PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(__dirname, "figures_png");
const INK = "0E0E10", MUTED = "6B6A66", EMBER = "E2430B", PAPER = "F1EFEA", RULE = "D6D2C8";
const CONTENT = 9746; // A4 width minus 0.75" margins, in DXA
const FONT = "Arial";

// ---------- small building blocks ----------
let numberedLists = 0;

function runs(text, extra = {}) {
  // **bold** inside a string becomes a bold run
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((part) =>
    part.startsWith("**") ? new TextRun({ text: part.slice(2, -2), bold: true, font: FONT, ...extra }) : new TextRun({ text: part, font: FONT, ...extra }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 140, line: 300 }, ...opts });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)], pageBreakBefore: true });
const H1keep = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const bullets = (items) => items.map((t) => new Paragraph({ children: runs(t), numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 290 } }));
function numbered(items) {
  const reference = `num-${numberedLists++}`;
  return items.map((t) => new Paragraph({ children: runs(t), numbering: { reference, level: 0 }, spacing: { after: 80, line: 290 } }));
}
const callout = (text, color = EMBER, fill = "FFF0E8") => new Paragraph({
  children: runs(text),
  shading: { type: ShadingType.CLEAR, fill, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 24, color, space: 8 } },
  spacing: { before: 120, after: 200, line: 290 },
  indent: { left: 160, right: 160 },
});

function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    shading: head ? { type: ShadingType.CLEAR, fill: PAPER, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ children: runs(String(text), { size: 19, bold: head || undefined }), spacing: { after: 0, line: 260 } })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 120 } });

function figure(file, caption, widthPx = 620) {
  const data = fs.readFileSync(path.join(FIG, file));
  const w = data.readUInt32BE(16), h = data.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round((widthPx * h) / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 }, children: [new TextRun({ text: caption, font: FONT, italics: true, font: FONT, size: 18, color: MUTED })] }),
  ];
}

function cover(title, subtitle, lines) {
  return [
    new Paragraph({ spacing: { before: 2400, after: 120 }, children: [new TextRun({ text: "ASTEROID HUNTER", font: FONT, bold: true, color: EMBER, size: 22, characterSpacing: 60 })] }),
    new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: title, font: FONT, bold: true, size: 56, color: INK })] }),
    new Paragraph({ spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: EMBER, space: 12 } },
      children: [new TextRun({ text: subtitle, font: FONT, size: 28, color: MUTED })] }),
    ...lines.map((l) => new Paragraph({ spacing: { after: 80 }, children: runs(l, { size: 22 }) })),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function contents(items) {
  return [H1keep("Contents"), ...numbered(items), new Paragraph({ children: [new PageBreak()] })];
}

function makeDoc(shortTitle, children) {
  const refs = Array.from({ length: numberedLists }, (_, i) => ({
    reference: `num-${i}`,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }],
  }));
  return new Document({
    creator: "Asteroid Hunter team",
    title: shortTitle,
    styles: {
      default: { document: { run: { font: FONT, size: 21, color: INK } } },
      paragraphStyles: [
        { id: "Normal", name: "Normal", run: { font: FONT, size: 21, color: INK } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 34, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 120, after: 220 }, outlineLevel: 0,
            border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: EMBER, space: 6 } } } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 27, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 23, bold: true, font: FONT, color: EMBER }, paragraph: { spacing: { before: 220, after: 80 }, outlineLevel: 2, keepNext: true } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
        ...refs,
      ],
    },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: `Asteroid Hunter · ${shortTitle}`, font: FONT, size: 16, color: MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Page ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED }),
        new TextRun({ text: " of ", size: 16, color: MUTED }), new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: MUTED })] })] }) },
      children,
    }],
  });
}

async function save(doc, file) {
  fs.writeFileSync(file, await Packer.toBuffer(doc));
  console.log("wrote", path.relative(ROOT, file));
  numberedLists = 0;
}

// ---------- shared facts ----------
// Blob-level metrics on the test night, from the same export the web app uses (tools/export_web_data.py).
const NIGHT = JSON.parse(fs.readFileSync(path.join(ROOT, "web", "public", "data", "night.json"), "utf8"));
const M = NIGHT.metrics;
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const n0 = (x) => Math.round(x).toLocaleString("en-US");
const CUTS = [["Stage-1 cut (deployed)", M.deployed], ["Cut for 95% recall", M.at_recall["0.95"]], ["Cut for 90% recall", M.at_recall["0.90"]]];
const CUT_TABLE = [`Our CNN on the whole test night (${n0(NIGHT.blobs)} blobs, ${pct(NIGHT.blobs_real / NIGHT.blobs)} real): recall and precision together`,
  ["Cut", "Recall", "Precision", "F1", "False alarms per photo"],
  CUTS.map(([l, o]) => [l, pct(o.recall), pct(o.precision), o.f1.toFixed(2), n0(o.false_alarms_per_photo)]), [3200, 1500, 1500, 1200, 2346]];
const CUT_NOTE = `Recall alone can look too good: the deployed stage-1 cut keeps ${pct(M.deployed.recall)} of real blobs, but only ${pct(M.deployed.precision)} of the blobs it keeps are real ` +
  `(${n0(M.deployed.false_alarms_per_photo)} junk blobs per photo). That is a deliberate choice: an asteroid dropped here is lost for good, while junk is removed later by linking ` +
  `and the track model. Asking for 90% recall instead gives ${pct(M.at_recall["0.90"].precision)} precision. Average precision over all cuts is ${M.average_precision.toFixed(3)} ` +
  `(random guessing would score ${(NIGHT.blobs_real / NIGHT.blobs).toFixed(3)}). The whole pipeline is judged at the end by depth, known asteroids recovered and unexplained alarms per night.`;
const COURSE = "Machine Learning (702DB0C015) · B.Tech CSE (Data Science), Semester V · MPSTME, NMIMS · 2026-27";
const TEAM = "Advik (lead), Aaryan, Paras, Satvik";
const AI_NOTE = "**About this document.** It was drafted with the help of an AI assistant (Claude, by Anthropic) from the project's own files, " +
  "as the course's GenAI policy allows for planning and learning. Before you submit it, read it fully, change anything that is not true for you, " +
  "and rewrite the parts you present in your own words (see 00_Start_Here/07_AI_Use_and_Integrity.md).";
const HEADLINE = [
  ["CNN vs simple model (validation nights, 67,041 blobs, 3.2% real)", "average precision 0.899 vs 0.780", "00_Start_Here/01_Project_Overview.md"],
  ["False alarms per photo when 90% of real blobs are found", "176 (CNN) vs 813 (simple model): 4.6x fewer", "project/docs/realbogus_cnn.md"],
  ["CNN on the untouched test night", "average precision 0.938", "project/docs/realbogus_cnn.md"],
  ["CNN at the deployed stage-1 cut (test night, one quadrant)", `recall ${pct(M.deployed.recall)} but precision ${pct(M.deployed.precision)}: tuned to miss nothing; junk is removed later`, "web/public/data/night.json"],
  ["Track classifier (4 models, cross-validated by night)", "about 0.999 AP for 3 of 4 models: saturated (96% of tracks are easy fakes), so not used to judge", "project/docs/tracklet_classifier.md"],
  ["Depth: magnitude where half the planted fakes are found (test night)", "v1 20.5 → v4 21.4 → deep search 22.1", "reports/scoreboard.csv, docs/deep_search.md"],
  ["Real search of the test night (no fakes)", "31 known asteroids recovered (16.2 to 21.8 mag)", "00_Start_Here/01_Project_Overview.md"],
  ["Fast (trailed) objects", "97% of planted trails recovered (93% on a later night)", "project/docs/fast_movers.md"],
  ["3x more training data (v5)", "did not help at equal false alarms", "project/docs/more_data.md"],
];
const SCOREBOARD = [ // project/reports/scoreboard.csv (v3 left out: one quadrant only)
  ["test", "v1", "0.942", "97.4%", "1,245", "0.998", "20.53", "39/44", "0.0"],
  ["test", "v2", "0.942", "99.2%", "4,293", "0.992", "21.39", "40/44", "13.3"],
  ["test", "v4", "0.942", "99.2%", "4,293", "0.993", "21.34", "40/44", "5.0"],
  ["test", "v5", "0.940", "99.2%", "3,857", "0.993", "20.56", "38/44", "0.3"],
  ["validation", "v1", "0.919", "96.0%", "1,184", "1.000", "20.18", "21/28", "0.0"],
  ["validation", "v4", "0.919", "99.2%", "3,901", "0.998", "20.49", "24/28", "0.0"],
  ["holdout", "v1", "0.905", "95.4%", "1,468", "1.000", "20.14", "16/18", "0.0"],
  ["holdout", "v4", "0.905", "98.5%", "4,716", "0.998", "20.47", "17/18", "0.3"],
];
const SB_HEAD = ["nights", "version", "blob AP", "stage-1 recall", "junk kept per photo", "track AP (saturated)", "depth (mag)", "known found", "unexplained per night"];
const SB_W = [1150, 850, 900, 1000, 1150, 900, 1050, 1000, 1196];
const GLOSSARY = [
  ["Asteroid", "a small rocky body orbiting the Sun; in our photos, a dot that moves in a straight line"],
  ["ZTF", "Zwicky Transient Facility: a telescope camera at Palomar Observatory that photographs the whole northern sky every few nights"],
  ["Exposure / photo / frame", "one 30-second picture of one patch of sky"],
  ["Science / reference / difference image", "tonight's photo / an old, very deep photo of the same sky / tonight minus old: only what changed is left"],
  ["Blob (detection)", "a spot in the difference image brighter than the noise"],
  ["Stamp", "a small 31 x 31 pixel cut-out around a blob, from all three images"],
  ["Real / bogus", "a real object (asteroid, planted fake) / junk (noise, bad subtraction, cosmic ray)"],
  ["Planted fake (injection)", "an artificial asteroid we paint into the images so we know the right answer"],
  ["Tracklet (track)", "blobs from different photos that line up as one moving object"],
  ["Magnitude", "brightness scale where a bigger number means fainter; each +1 is about 2.5x fainter"],
  ["Depth", "the magnitude where half the planted fakes are found; bigger is better"],
  ["CNN", "convolutional neural network: learns patterns in pictures with small sliding filters"],
  ["XGBoost", "a model made of many small decision trees, each one fixing the mistakes of the ones before"],
  ["Precision / recall", "of what we flagged, how much was real / of what was real, how much we flagged"],
  ["Average precision (AP)", "one number summarising precision at every recall; 1.0 is perfect; good for rare positives"],
  ["Leakage", "when the test data secretly gives away the answer, so the score is too good to be true"],
  ["Overfitting", "learning the training examples by heart instead of the general pattern"],
  ["Sigma (σ)", "the typical size of the noise; '5 sigma' means 5 times bigger than typical noise"],
];

// ---------- the group report ----------
function groupReport() {
  const c = [];
  c.push(...cover("The whole project, start to end", "Group report: how we built it, how it works, and how well it works",
    [`**Course:** ${COURSE}`, `**Team:** ${TEAM}`, "**Data:** public ZTF images from the IRSA archive", "**Date:** October 2026"]));
  c.push(...contents(["The project in one page", "The problem", "The data", "The pipeline, step by step", "Our ML model: the CNN",
    "The second model: scoring tracks", "How we tested it", "Results (evaluation metrics)", "How it got better: the story v1 to v5",
    "The deep search", "What did not work, limits, and what we do not claim", "Who did what", "Tools, code, and how to run it",
    "References", "Use of AI (declaration)", "Glossary"]));

  c.push(H1keep("1. The project in one page"));
  c.push(callout(AI_NOTE));
  c.push(P("**In one sentence:** we built a machine-learning pipeline that looks through old, public telescope images from the Zwicky Transient Facility (ZTF) for small moving dots, which are asteroids, and we measured honestly how well it works."));
  c.push(P("Telescopes photograph the same patch of sky again and again. Stars stay in the same place; asteroids move a little between photos. If we subtract an old photo from tonight's, the stars cancel out and only things that changed are left. The problem is that almost everything left over is junk. On our test night, five photos of one patch of sky gave 41,522 blobs, and only about 3% of them were real. So we:"));
  c.push(...numbered([
    "find every blob in the difference images,",
    "use a **neural network (CNN)** to decide which blobs look like a real object and which are junk,",
    "**link** blobs from different photos into straight-line tracks,",
    "use a second model (**XGBoost**, made of decision trees) to score each track,",
    "compare the tracks with the catalogue of known asteroids, and keep anything unknown for a person to check,",
    "**measure** everything on nights the models never saw, using fake asteroids we planted ourselves so we know the right answer.",
  ]));
  c.push(H2("Headline results"));
  c.push(table(["What we measured", "Result", "Where the number comes from"], HEADLINE, [3800, 3300, 2646]));
  c.push(gap());
  c.push(callout("**What we do not claim:** no new asteroid has been discovered or reported. The pipeline reliably finds known asteroids down to about magnitude 21, and one faint object in a related two-night study passes every check we have, but it still needs a human review and the Minor Planet Center's process.", "16202C", "EEF2F6"));
  c.push(...figure("model.png", "Figure 1. The one ML architecture of the project: the real/bogus CNN, and where it sits in the pipeline (00_Start_Here/00_How_Our_Model_Works.svg)."));

  c.push(H1("2. The problem"));
  c.push(H2("What we are solving"));
  c.push(P("**Input:** a few 30-second photos of the same patch of sky, taken over one night. **Output:** a ranked list of moving objects (tracks), each with a score and a flag saying whether it is a known asteroid."));
  c.push(H2("Why it is hard"));
  c.push(...bullets([
    "**Most blobs are junk.** About 97% of blobs are noise, leftovers of bright stars, cosmic-ray hits or edges of bad pixels.",
    "**The interesting asteroids are faint,** barely brighter than the noise.",
    "**Nobody labels the sky.** There is no list of 'every asteroid in this picture', so we had to create our own labels.",
    "**The telescope moves a little between photos,** so the same star is not on the same pixel each time.",
    "**Catalogues have errors too;** one of them was about 2.9 arcseconds off, which we only found because our model disagreed with it.",
  ]));
  c.push(H2("Why machine learning is needed (and where it is not)"));
  c.push(P("Junk comes in many shapes, and writing a hand-made rule for each one fails. A CNN can **learn** what real and junk look like from labelled examples. A tree model can combine many weak clues about a track better than any single threshold. But some steps are not learning at all: finding blobs, joining them along straight lines and adding photos together are physics and statistics. Knowing which part needs learning and which does not is part of good ML design."));
  c.push(H2("How we measure success"));
  c.push(P("Not with accuracy: a model that always says 'junk' would be 96.8% accurate and find nothing. Instead we use **precision, recall and average precision**, the number of **false alarms per photo** at a fixed recall, the **depth** (how faint we can go), the number of **known asteroids recovered**, and **unexplained alarms per night**. A simple baseline (logistic regression on six numbers per blob, AP 0.780) is the bar every model must beat."));
  c.push(P("The idea came from an online post where someone found a planet candidate in old Kepler data with AI-written code. We checked that post first: the signal belongs to an already confirmed planet (Kepler-169 e). That taught us the first lesson of the project: check before you claim."));

  c.push(H1("3. The data"));
  c.push(P("All data is public: ZTF images served by NASA/IPAC's IRSA archive. For every exposure there are four files: the **science** image (38 MB), the **mask** of bad pixels (19 MB), the **difference** image (7 MB) and the **PSF**, the shape a point of light makes in that photo (6 KB). The reference image is not downloaded separately: reference = science − difference."));
  c.push(table(["Fact", "Value"], [
    ["Exposures of our patch of sky in the archive", "1,552 (898 r-band, 478 g-band, 176 i-band)"],
    ["Exposures with quality warnings (removed)", "16.4%"],
    ["Nights with at least 4 good photos", "40"],
    ["Downloaded for the first 11 nights", "220 files, 3.55 GB, nothing missing"],
    ["One photo (one CCD quadrant)", "3072 x 3080 pixels, 1.01 arcsec per pixel, 30 seconds"],
    ["Raw data on disk by the end", "about 50 GB (not in the project folder)"],
  ], [4600, 5146]));
  c.push(H2("Split by night, decided before any modelling"));
  c.push(P("We split the data **by night**, never by random rows, and wrote the split into a configuration file (configs/acquire.yaml) before training anything, so it could not quietly change after a disappointing result."));
  c.push(table(["Split", "Nights", "Used for"], [
    ["Train", "8", "learning the models"], ["Validation", "2", "choosing settings and thresholds"],
    ["Test", "1 (2019-02-08)", "the honest final check; the sharpest, deepest night"], ["Holdout", "1 (2022-06-17)", "opened only by the scoreboard"],
  ], [2200, 2400, 5146]));
  c.push(...figure("02_split_by_night.png", "Figure 2. Splitting by night (00_Start_Here/11_Diagrams/02_split_by_night.svg)."));
  c.push(H2("Cleaning"));
  c.push(...bullets([
    "Bad pixels (saturated, dead, noisy, satellite trails, empty areas) are masked; about 6-12% of a difference image is empty (NaN).",
    "The noise level is measured robustly: 1.4826 x the median absolute deviation, so bright stars do not inflate it.",
    "Pixel positions are turned into sky coordinates, because the telescope points slightly differently each time.",
  ]));

  c.push(H1("4. The pipeline, step by step"));
  c.push(P("The same code path is used to evaluate the system (with planted fakes) and to run the real search (without). What we test is exactly what we deploy."));
  c.push(...figure("01_system_overview.png", "Figure 3. The whole system (00_Start_Here/11_Diagrams/01_system_overview.svg)."));
  const steps = [
    ["Pick the photos", "Take 5 photos of one patch of sky from one night, in time order.", "Aaryan"],
    ["Open and clean", "Read the four files, build the bad-pixel mask, compute the reference, measure the noise.", "Aaryan"],
    ["Plant fakes (test mode)", "Paint 300 fake asteroids into the photos with ZTF's own blur shape, so we know the right answer.", "Paras"],
    ["Find blobs", "Circle every spot at least 2x brighter than the noise and at least 5 pixels big: about 7,800 per photo.", "Aaryan"],
    ["Cut stamps", "Cut a 31 x 31 pixel square around each blob from science, reference and difference.", "Paras"],
    ["CNN score", "The neural network gives every blob a score from 0 (junk) to 1 (real). Low scores are dropped.", "Paras"],
    ["Link", "Join kept blobs from different photos that lie on a straight line at a steady speed.", "Satvik"],
    ["Describe tracks", "Give every track 17 numbers: speed, straightness, steady brightness, CNN opinion and more.", "Satvik"],
    ["Score tracks", "XGBoost turns the 17 numbers into one track score.", "Satvik"],
    ["Known or unknown?", "Compare with the catalogue of known asteroids (positions from JPL Horizons).", "Satvik"],
    ["Measure", "Count planted fakes found by brightness, known asteroids found, and unexplained alarms.", "Advik"],
    ["Show it", "The live viewer and the web app show every stage on real images.", "Advik"],
  ];
  c.push(table(["#", "Step", "What happens, in plain words", "Owner"], steps.map((s, i) => [String(i + 1), ...s]), [500, 2000, 6046, 1200]));

  c.push(H1("5. Our ML model: the CNN"));
  c.push(P("We use **one** neural network architecture: a small convolutional neural network (CNN) that looks at the three stamps of a blob and outputs one number, p(real). Small was a deliberate choice: the stamps are only 31 x 31 pixels and the patterns are local, so 72,145 weights are enough, train fast on a laptop, and are hard to overfit."));
  c.push(H2("Why three pictures"));
  c.push(P("A real new object shows up in the **science** and **difference** images but **not** in the **reference**. A badly subtracted star shows up in science **and** reference, with a messy pattern in the difference. A cosmic ray is a sharp spike, narrower than a star. Seeing all three at once lets the filters learn these differences directly."));
  c.push(...figure("02_triplet_stamp.png", "Figure 4. What the three channels look like for real objects and junk (03_Paras/07_Diagrams/02_triplet_stamp.svg)."));
  c.push(H2("The layers"));
  c.push(table(["Layer", "Output size", "What it does", "Weights"], [
    ["Input", "3 x 31 x 31", "science, reference, difference", "0"],
    ["2 x Conv 3x3 (16 filters) + ReLU", "16 x 31 x 31", "finds small shapes: dots, edges, rings", "2,768"],
    ["Max-pool 2x2", "16 x 15 x 15", "keeps the strongest value; halves the size", "0"],
    ["2 x Conv 3x3 (32 filters) + ReLU", "32 x 15 x 15", "combines shapes into bigger patterns", "13,888"],
    ["Max-pool 2x2", "32 x 7 x 7", "", "0"],
    ["Conv 3x3 (64 filters) + ReLU", "64 x 7 x 7", "patterns over the whole blob", "18,496"],
    ["Max-pool 2x2", "64 x 3 x 3", "", "0"],
    ["Flatten + Dropout 0.3", "576", "one long list; dropout switches off 30% while training", "0"],
    ["Fully connected + ReLU", "64", "weighs all the evidence", "36,928"],
    ["Fully connected + Sigmoid", "1", "p(real) from 0 to 1", "65"],
    ["**Total**", "", "", "**72,145**"],
  ], [3100, 1600, 3846, 1200]));
  c.push(H2("How it was trained"));
  c.push(...bullets([
    "**Labels:** planted fakes and known asteroids within 2 pixels of a blob are 'real'; everything else is 'bogus'.",
    "**Normalising:** every stamp is put on the same scale ((value − median) / noise), so the model compares shapes, not brightness.",
    "**Augmenting:** random 90° rotations and mirror images (the sky has no 'up'); training data only.",
    "**Balancing:** all 8,863 real stamps plus 5 bogus per real = 53,178 training stamps (16.7% real). Validation keeps the true mix: 67,041 stamps, 3.2% real.",
    "**Settings:** Adam optimiser, learning rate 0.001, batch 256, up to 25 epochs, early stopping after 4 epochs without improvement; binary cross-entropy loss. Best validation AP 0.899 at epoch 21.",
    "**Cross-fitting:** the scores passed to the next model come from CNNs that never saw that night, so the next model does not learn from over-confident scores.",
  ]));
  c.push(P("The web app (web/) runs this exact network, with the trained weights, inside the browser; its scores match PyTorch to within 0.00001 on 48 real stamps."));

  c.push(H1("6. The second model: scoring tracks"));
  c.push(P("After linking, each track is described by **17 numbers**. Each has a physical reason: an asteroid moves at a steady speed in a straight line, keeps nearly the same brightness for half an hour, stays a round point, leaves nothing behind at its old positions, and is faintly visible at its predicted position even in photos where no blob was found."));
  c.push(table(["Group", "Numbers", "Idea"], [
    ["Motion", "photos it appears in, speed, distance off a straight line", "steady straight motion"],
    ["Brightness", "mean and spread of magnitude", "asteroids keep their brightness; noise does not"],
    ["Strength", "mean and lowest SNR", "one weak point makes a weak track"],
    ["CNN opinion", "mean and lowest CNN score", "what the network thought of each blob"],
    ["Shape", "size spread, elongation", "a point source stays the same shape"],
    ["Static neighbours (v4)", "blobs left behind at the track's positions", "a mover leaves none; an artifact leaves many"],
    ["Forced photometry (v4)", "signal at the predicted spot in every photo", "real objects leave a weak signal even when missed"],
  ], [2300, 3700, 3746]));
  c.push(P("We compared four models, always cross-validated **by night** (GroupKFold): logistic regression (0.999 AP), a single decision tree (0.983), a random forest (0.999) and XGBoost (0.999; 300 trees, depth 3, learning rate 0.05). Three of them tie. We use XGBoost because it handles missing values and interactions without extra work, but we say openly that logistic regression would be a defensible simpler choice."));
  c.push(...figure("02_boosting.png", "Figure 5. From one decision tree to a forest to boosting (04_Satvik/07_Diagrams/02_boosting.svg).", 560));

  c.push(H1("7. How we tested it"));
  c.push(...bullets([
    "**Planted fakes (injection-recovery).** We cannot know every real asteroid in an image, so we plant 300 fakes per night (magnitude 17 to 22, speed 5 to 100 arcsec/hour, any direction), run the normal pipeline and count how many come out.",
    "**Are the fakes realistic?** A classifier asked to tell fakes from real known asteroids could not (AUC 0.46-0.48, where 0.5 means 'cannot tell').",
    "**Held-out nights.** Numbers only count when they come from nights nobody tuned on.",
    "**One scoreboard.** A single function computes every headline number for every model version and writes reports/scoreboard.csv.",
    "**Fair comparisons.** Models are compared at the same false-alarm level, not each at its own threshold.",
    "**Statistics.** When we try thousands of possibilities (the deep search), chance alone produces strong-looking signals, so the thresholds are set from pure-noise tests.",
  ]));
  c.push(...figure("02_evaluation_protocol.png", "Figure 6. The evaluation protocol (01_Advik/07_Diagrams/02_evaluation_protocol.svg)."));
  c.push(H2("The metrics in plain words"));
  c.push(table(["Metric", "Question it answers"], [
    ["Recall", "Of all the real objects, how many did we find?"],
    ["Precision", "Of everything we flagged, how much was real?"],
    ["Average precision (AP)", "How good is the model across all thresholds, for a rare class? (1.0 = perfect)"],
    ["False alarms per photo at 90% recall", "If we insist on finding 90% of real objects, how much junk do we have to look at?"],
    ["Depth (magnitude at 50% completeness)", "How faint can an asteroid be and still be found half the time?"],
    ["Known asteroids recovered", "Does it find real asteroids on its own, without our fakes?"],
    ["Unexplained alarms per night", "How many high-scoring tracks are neither fakes nor known asteroids?"],
  ], [3600, 6146]));

  c.push(H1("8. Results (evaluation metrics)"));
  c.push(H2("The CNN against the simple baseline"));
  c.push(table(["Model (validation nights)", "Average precision", "Precision at 90% recall", "False alarms per photo at 90% recall"], [
    ["Logistic regression on 6 numbers (baseline)", "0.780", "0.191", "813"],
    ["**Our CNN**", "**0.899**", "**0.522**", "**176**"],
  ], [3900, 1800, 2000, 2046]));
  c.push(P("At the same recall the CNN lets through **4.6x fewer false alarms**. On the untouched test night its average precision is **0.938**."));
  c.push(...figure("04_cnn_results.png", "Figure 7. CNN results (03_Paras/07_Diagrams/04_cnn_results.svg)."));
  c.push(H2("The CNN on the whole test night"));
  c.push(table(CUT_TABLE[1], CUT_TABLE[2], CUT_TABLE[3]));
  c.push(P(CUT_NOTE));
  c.push(table(["At the deployed cut", "Kept (predicted real)", "Dropped (predicted junk)"], [
    ["Really real", `${n0(M.deployed.tp)} true positives`, `${n0(M.deployed.fn)} missed`],
    ["Really junk", `${n0(M.deployed.fp)} false alarms`, `${n0(M.deployed.tn)} true negatives`],
  ], [3200, 3273, 3273]));
  c.push(gap());
  c.push(H2("Why accuracy is the wrong number"));
  c.push(table(["", "Predicted real", "Predicted junk"], [
    ["Really real (2,129)", "1,547 true positives", "582 missed"],
    ["Really junk (64,912)", "493 false alarms", "64,419 true negatives"],
  ], [3200, 3273, 3273]));
  c.push(P("This is the baseline at threshold 0.5 on the validation nights. Accuracy is 98.4%, but a model that always says 'junk' scores 96.8% and finds nothing. ROC AUC looked excellent (0.95) while average precision was only 0.78. For rare positives, precision and recall tell the truth."));
  c.push(H2("The scoreboard: every version, measured the same way"));
  c.push(table(SB_HEAD, SCOREBOARD, SB_W));
  c.push(P("From project/reports/scoreboard.csv. Planted-fake runs over 3 quadrant-nights for test and holdout and 6 for validation. On the single test-night quadrant the summary numbers are 20.5 (v1) and 21.4 (v4); with the deep search, **22.1**."));
  c.push(...figure("03_scoreboard_versions.png", "Figure 8. How the versions compare (01_Advik/07_Diagrams/03_scoreboard_versions.svg)."));
  c.push(H2("The real search"));
  c.push(P("Without any fakes, the pipeline found **31 known asteroids** on the test night by itself, from magnitude 16.2 to 21.8, with no unexplained candidates above the threshold. On the test night, all 9 tracks that our labels called 'false positives' turned out to be real known asteroids that the labels had missed."));

  c.push(H1("9. How it got better: the story v1 to v5"));
  c.push(...numbered([
    "**v1:** CNN + XGBoost on single photos. Track AP 0.999 looked perfect, but the scoreboard showed depth only about 20.5. The right metrics were depth and known asteroids, not AP.",
    "**We measured where asteroids were lost:** at linking, not at the classifier. Relaxing linking (3 of 5 photos instead of 4) found more asteroids but created about 1,500 chance tracks per night, which became useful hard examples (**v2**).",
    "**Looking at false alarms by eye** showed two kinds (artifacts that 'walk' and noise that happens to line up). One new feature group for each gave **v4** with 17 features.",
    "**A bigger multi-frame CNN** tied the simple XGBoost (AP 0.9933 vs 0.9924), so we kept the simple one.",
    "**Three times more training data (v5)** did not help at equal false alarms: one 30-second photo only holds so much light.",
    "So we **added photos together** (the deep search): +0.7 magnitude, to 22.1.",
  ]));
  c.push(H2("Real bugs found by checking against reality"));
  c.push(...bullets([
    "**Telescope pointing shifts** (up to 12 pixels between photos): real asteroids failed to link while fakes linked perfectly. Fixed by linking on one shared sky grid; leftover error fell from about 2.5 to 0.05-0.47 pixels.",
    "**Catalogue offset:** the CNN's most confident 'mistakes' were known asteroids; the SkyBoT positions were about 2.9 arcsec off. Switching to JPL Horizons cut the median error from 2.72 to 0.26 pixels.",
    "**Leakage in the first dataset:** 100% accuracy was a warning sign. A test showed 100% → 50% (a coin flip) on realistic data; after rebuilding, 99%. (These are accuracies on a 50/50 balanced set, the one case where accuracy is fair; on real data, 3% positive, we never use it.)",
    "**A bug in one of our own features** flagged 22% of real tracks; after the fix, 0.4%. A test was written for each fix.",
  ]));

  c.push(H1("10. The deep search"));
  c.push(P("A faint asteroid is hidden in the noise of any single photo. If we knew its motion, we could shift each photo so the asteroid stays on the same pixel and add them up: the signal grows N times but the noise only √N times. We do not know the motion, so we try **4,888** of them (every direction up to 120 arcsec/hour) on the laptop's GPU."));
  c.push(...bullets([
    "**The catch:** trying 4,888 motions at every pixel makes chance peaks common. On pure noise there were 6,179 false peaks per stack at 5σ but only 44 at 6σ, so we use **6σ**.",
    "**Real data are messier than noise:** the first version still gave about 47,550 peaks per stack; after clipping, background removal and masking it was 94.",
    "**Null test:** shuffling the photo times (so real movers no longer line up) gave 16 false tracks per night before a velocity-consistency cut and **0** after, while all 20 known asteroids were kept.",
    "**Result:** half the planted fakes found at **22.1** magnitude, vs 21.4 for the best single-photo system on the same night (+0.7).",
  ]));
  c.push(...figure("04_shift_and_stack.png", "Figure 9. Shift-and-stack (01_Advik/07_Diagrams/04_shift_and_stack.svg)."));

  c.push(H1("11. What did not work, limits, and what we do not claim"));
  c.push(...bullets([
    "**No discovery is claimed.** Nothing has been reported to the Minor Planet Center.",
    "**One night, one patch:** the best depth (22.1) is for one night and one CCD quadrant.",
    "**Labels are partly simulated:** the realism test says fakes look like real asteroids on the features we checked, which is a check, not a proof.",
    "**The test night was looked at** while fixing real bugs; that is why a separate holdout night exists.",
    "**Probabilities are inflated** because training was balanced (16.7% real vs about 3% in reality); we correct for the base rate when we need real probabilities.",
    "**A fancier model and more data did not help.** We report these negative results because they explain where the real limit is.",
    "**The deep search needs at least 20 photos in a night;** a night with 18 photos silently found nothing until we added a check.",
  ]));

  c.push(H1("12. Who did what"));
  c.push(table(["Member", "Part", "Main results", "Course marks it covers"], [
    ["Aaryan", "Problem, data, preprocessing, exploration, leakage audit", "1,552 exposures checked; split by night; leakage proved and fixed (100% → 50% → 99%)", "Problem understanding (3), preprocessing and exploration (3, shared)"],
    ["Paras", "Labels, dataset, the CNN", "AP 0.899 vs 0.780; 4.6x fewer false alarms; test AP 0.938; catalogue bug found", "Model development (4, shared), data preparation"],
    ["Satvik", "Linking, track classifier, metrics, cross-match", "pointing-shift fix; 17 features; 4 models tie; metric choices", "Model development (4, shared), evaluation basics"],
    ["Advik (lead)", "System, evaluation, deep search, live viewer, web app", "scoreboard; depth 20.5 → 21.4 → 22.1; 31 known asteroids", "Evaluation (3), documentation and presentation (2), integration"],
  ], [1300, 2600, 3446, 2400]));
  c.push(P("The hand-over chain: Aaryan → a clean table of blobs and a split by night → Paras → every blob with a CNN score → Satvik → scored tracks marked known or unknown → Advik → the scoreboard, the deep search and the demo. Each person's step-by-step process is in their own report and in 0X_Name/00_Process_Diagram_Name.svg."));
  for (const n of ["Aaryan", "Paras", "Satvik", "Advik"]) c.push(...figure(`process_${n}.png`, `Figure. ${n}'s process.`, 560));

  c.push(H1("13. Tools, code, and how to run it"));
  c.push(table(["Thing", "Where"], [
    ["Runnable code", "project/src/asthunt/<stage>/<verb_object>.py, one purpose per file"],
    ["Settings", "project/configs/*.yaml (night split, thresholds, sizes)"],
    ["Trained models", "project/models/v1 ... v5 (CNN + track model + thresholds + provenance)"],
    ["Tests", "project/tests: 166 offline tests (pytest -q, about 40 s)"],
    ["Lessons and notebooks", "project/docs/*.md and project/notebooks/*.ipynb, one per phase"],
    ["Live viewer", "python scripts/live_viewer.py (needs the raw images)"],
    ["Web app", "web/: cd web, npm install, npm run dev, open http://localhost:3000 (no Python or raw data needed)"],
    ["Software", "Python, NumPy, pandas, astropy, SEP, PyTorch, scikit-learn, XGBoost; Next.js for the web app"],
  ], [2600, 7146]));

  c.push(H1("14. References"));
  c.push(...bullets([
    "Duev, D. A. et al. (2019). Real-bogus classification for the Zwicky Transient Facility using deep learning. MNRAS 489, 3582. (closest published work to our CNN)",
    "Mahabal, A. et al. (2019). Machine Learning for the Zwicky Transient Facility. PASP 131, 038002.",
    "Masci, F. J. et al. (2019). The Zwicky Transient Facility: Data Processing, Products, and Archive. PASP 131, 018003.",
    "Whidden, P. J. et al. (2019). Fast algorithms for slow moving asteroids. AJ 157, 119. (the shift-and-stack idea)",
    "Shallue, C. J. & Vanderburg, A. (2018). Identifying exoplanets with deep learning. AJ 155, 94. (the inspiration)",
    "Chen, T. & Guestrin, C. (2016). XGBoost: a scalable tree boosting system. KDD.",
    "Kingma, D. & Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
    "Data: Zwicky Transient Facility, via NASA/IPAC Infrared Science Archive (IRSA). Use the acknowledgement text from the IRSA ZTF release notes.",
  ]));
  c.push(P("Only the abstracts and bibliographic details were checked online; read the papers yourselves before quoting details (see 00_Start_Here/04_Course_Mapping.md)."));

  c.push(H1("15. Use of AI (declaration)"));
  c.push(callout("Edit this so it is true for each of you before submitting.", "16202C", "EEF2F6"));
  c.push(P("This project was developed with the assistance of an AI coding assistant (Claude, by Anthropic), used under the direction of Advik Divekar. The assistant helped to plan the project, write and test the source code, write the lesson documents, diagrams, the web app and these reports, and prepare study material. Each team member then studied, re-ran and [describe what you personally did] the part they present: Advik (system, evaluation and deep search), Aaryan (data and preprocessing), Paras (labels and the neural network), Satvik (linking, tree models and metrics). We understand and can explain the methods and results, and all references used are cited."));

  c.push(H1("16. Glossary"));
  c.push(table(["Word", "Meaning in plain words"], GLOSSARY, [2800, 6946]));
  return makeDoc("Group report", c);
}

// ---------- the individual reports ----------
const PEOPLE = {
  Aaryan: {
    folder: "02_Aaryan", role: "Problem, data and preprocessing",
    marks: "Problem Understanding and Formulation (3); Data Preprocessing and Exploration (3, shared with Paras)",
    summary: "My part is the start of the pipeline. I defined the problem, found and downloaded the data, split it by night, cleaned every image, found the blobs that every later model works on, and proved that our first dataset was cheating (leakage). If the data is wrong or leaks, every model and number after it is wrong.",
    steps: [
      ["Define the problem", "I wrote down exactly what goes in (a night of photos of one patch of sky) and what comes out (a ranked list of tracks, each marked known or unknown), why it is hard, why machine learning helps, and how success is measured: depth, known asteroids found and false alarms, never accuracy.", "docs/project_setup.md, 00_Start_Here/01_Project_Overview.md"],
      ["Ask the archive before downloading", "One metadata query took seconds and told us there were 1,552 exposures of our patch (898 r, 478 g, 176 i band), that 16.4% had quality warnings (removed), and that only 40 nights have at least 4 good photos. Choosing from a table is cheap; downloading is not.", "acquire/query_ztf_metadata.py"],
      ["Split by night, before modelling", "8 nights for training, 2 for validation, 1 test night (2019-02-08) and later 1 holdout night, written in configs/acquire.yaml before any model was trained, so the split could not change after a bad result. Photos from one night are very alike, so a random split would leak.", "configs/acquire.yaml, evaluate/check_split_leakage.py"],
      ["Download safely", "4 files per exposure: 220 files, 3.55 GB, nothing missing. The downloader is idempotent (running it twice = once), atomic (writes a .part file and renames it only when complete), retries when the archive is busy, and records files that do not exist.", "acquire/download_ztf_images.py"],
      ["Clean every photo", "Read the files, build the bad-pixel mask (satellite trails, saturated, dead, noisy pixels, empty areas; but not 'ghosts', which cover 4% of the chip), compute reference = science − difference, and measure the background and the noise robustly (1.4826 x MAD).", "clean/load_exposure.py, clean/estimate_background.py"],
      ["Find the blobs", "Smooth with the star's own shape (a matched filter), keep groups of at least 5 pixels at least 2σ above the noise. About 7,800 blobs per photo, only 173 with SNR > 5. Detection is tuned to miss nothing; the CNN is tuned to remove junk.", "detect/extract_sources.py"],
      ["Explore the data (EDA)", "The threshold study showed a surprise: 1.5σ gives fewer blobs (6,141) than 2σ (7,764), because noise peaks merge into big blobs and can swallow a faint asteroid. Measuring beats rules of thumb.", "docs/difference_imaging.md"],
      ["Audit for leakage", "Our first dataset scored 100%: a warning sign. I found four kinds of leakage and proved it with an experiment: trained and tested on easy data 100%; tested on realistic near-misses 50% (a coin flip); trained on realistic data 99%. These are accuracies on a 50/50 balanced set, where accuracy is a fair measure; on the real, 3%-positive data we use precision and recall instead. We rebuilt the data.", "docs/leakage_audit.md"],
    ],
    results: [
      ["Exposures checked before downloading", "1,552 (16.4% removed for quality flags)"], ["Files downloaded", "220 files, 3.55 GB, nothing missing"],
      ["Split", "8 train / 2 validation / 1 test / 1 holdout nights"], ["Blobs per photo at 2σ", "about 7,800 (only 173 with SNR > 5)"],
      ["Sources: science vs difference image (5σ)", "1,950 vs 334: subtraction removes most stars"],
      ["Leakage experiment (accuracy, 50/50 balanced set)", "100% → 50% → 99%"], ["Hard negatives (track 3-10 / 1-3 / 0.5-1.5 px off the line; balanced accuracy)", "99% / 96% / 89%"],
    ],
    tables: [["Threshold study (one training photo)", ["Threshold", "Blobs", "Median size"], [["1.5σ", "6,141", "11 px"], ["**2.0σ (chosen)**", "**7,764**", "6 px"], ["3.0σ", "731", "8 px"], ["5.0σ", "334", "9 px"]], [3000, 3000, 3746]],
      ["The four kinds of leakage we found", ["Kind", "What it is", "In our first dataset"], [
        ["Target leakage", "a feature contains the answer", "magnitude and true speed were filled in only for planted fakes"],
        ["Group leakage", "train and test share a source", "all tracks came from the same 4 images"],
        ["Shortcut learning", "the classes differ in an easy, unreal way", "random junk tracks were 222.7+ px off a line; real ones at most 2.6 px"],
        ["Distribution shift", "real use differs from training", "the test night is sharper and deeper"]], [2200, 3200, 4346]]],
    problems: [
      ["The first dataset cheated", "A perfect score is a symptom, not a success. Asking 'what would have to be true for 100%?' found the leaks. Rules that came out: features must be measurable on new data, split by night, keep derived examples with their source, use realistic negatives."],
      ["Empty pixels made fake blobs", "Treating NaN as 0 creates edges that look like detections. Fix: mask them before detecting."],
      ["The archive was busy", "It answered '503 busy' twice during downloads; retries with growing waits recovered without losing anything."],
    ],
    handover: "Paras gets the output of load_exposure, the blob table from extract_sources (position, flux, SNR, shape), the split by night and the settings files.",
    figures: [["01_data_flow.png", "How the data flows (02_Aaryan/07_Diagrams/01_data_flow.svg)"], ["02_difference_imaging.png", "Difference imaging (02_difference_imaging.svg)"],
      ["03_preprocessing_funnel.png", "From pixels to blobs (03_preprocessing_funnel.svg)"], ["04_leakage_types.png", "Leakage types and the experiment (04_leakage_types.svg)"]],
    viva: [
      ["Why query metadata before downloading?", "It takes seconds and tells us which nights are worth downloading (enough good photos, no quality flags), so we never waste hours on unusable data."],
      ["Why split by night and not randomly?", "Photos from one night share sky, weather and blur. A random split puts near-copies in both training and test, so the score is too good. Splitting by night tests the model on genuinely new conditions."],
      ["Why does 1.5σ give fewer blobs than 2σ?", "At 1.5σ neighbouring noise peaks touch and merge into one big blob, so the count drops and a faint asteroid next to noise can be swallowed."],
      ["Why is the ghost bit not masked?", "Ghosts cover about 4% of the chip; masking them would hide real objects there. The CNN learns to reject ghosts instead."],
      ["What does 100% → 50% → 99% prove?", "That the first model learned a shortcut. On realistic hard examples it was a coin flip; trained on realistic data it is genuinely good."],
    ],
  },
  Paras: {
    folder: "03_Paras", role: "Labels, dataset and the neural network (CNN)",
    marks: "Model Development (4, shared with Satvik); the model-ready data part of Data Preprocessing (3, shared with Aaryan)",
    summary: "My part is the deep-learning heart of the project. The sky is not labelled, so I made labels (planted fakes and known asteroids), built a clean dataset of small three-channel pictures, and trained the CNN that decides for every blob whether it is real or junk. The model even found a mistake in our labels.",
    steps: [
      ["Plant fake asteroids", "For each night, 300 fakes with brightness 17-22 mag and speed 5-100 arcsec/hour in a random direction. Each fake is drawn with ZTF's own PSF for that photo, placed at a sub-pixel position, given realistic Poisson noise, and added to both science and difference images, exactly like a new object.", "simulate/plant_fakes_in_exposure.py"],
      ["Label every blob", "A blob within 2 pixels of a fake or of a known asteroid (positions from the catalogue, matched with a fast KD-tree look-up) is 'real'; the rest is 'bogus'. One training night: 1,059 real vs 36,161 bogus (1 : 34). We accept some label noise: a 'bogus' could be an uncatalogued asteroid.", "label/label_detections.py, label/match_detections_to_positions.py"],
      ["Cut stamps", "For each blob, a 31 x 31 pixel square from the science, reference and difference images: the CNN's input (3 x 31 x 31).", "detect/cut_triplet_stamps.py"],
      ["Prepare the data", "Normalise each stamp ((value − median) / (1.4826 x MAD), clipped to [−10, 50]); augment training stamps with rotations and mirroring; balance training to all 8,863 real + 5 bogus each = 53,178 stamps (16.7% real). Validation keeps the true mix (67,041 stamps, 3.2% real).", "dataset/assemble_stamp_dataset.py"],
      ["Learn the theory first", "In a from-scratch notebook I built a small network with NumPy: forward pass, loss, backpropagation, and a gradient check (both methods gave −0.02896520), then saw over-fitting and why early stopping helps.", "notebooks/neural_networks.ipynb"],
      ["Build and train the CNN", "5 convolution layers, 3 max-pools, dropout 0.3, two dense layers: 72,145 weights. Adam (learning rate 0.001), batch 256, up to 25 epochs, early stopping on validation AP (patience 4). Best validation AP 0.899 at epoch 21.", "models/define_realbogus_cnn.py, models/train_realbogus_cnn.py"],
      ["Measure it fairly", "Against the baseline (logistic regression on 6 numbers), at the same 90% recall: false alarms fall from 813 to 176 per photo (4.6x). On the untouched test night AP 0.938.", "docs/realbogus_cnn.md"],
      ["Learn from its mistakes", "The CNN's most confident 'bogus' blobs were clean point sources, and every one was a known asteroid. The catalogue (SkyBoT) was about 2.9 arcsec off in a constant direction. Switching to JPL Horizons cut the median error from 2.72 to 0.26 px.", "label/refine_known_positions.py, docs/label_quality.md"],
      ["Check the fakes are realistic", "A classifier asked to tell fakes from real known asteroids (same brightness, cross-validated by night) scored AUC 0.46-0.48: it cannot tell them apart.", "evaluate/measure_injection_realism.py"],
      ["Cross-fit the scores", "8 training nights in 4 folds of 2: each fold is scored by a CNN trained on the other 6 nights, so the next model never sees over-confident scores.", "models/cross_fit_realbogus_scores.py"],
    ],
    results: [
      ["Validation average precision", "0.899 (CNN) vs 0.780 (baseline)"], ["Precision at 90% recall", "0.522 vs 0.191"],
      ["False alarms per photo at 90% recall", "176 vs 813 (4.6x fewer)"], ["Test-night average precision", "0.938"],
      ["Test night, deployed cut (recall / precision)", `${pct(M.deployed.recall)} / ${pct(M.deployed.precision)}: tuned for recall on purpose`],
      ["Test night, cut for 90% recall (precision, false alarms per photo)", `${pct(M.at_recall["0.90"].precision)}, ${n0(M.at_recall["0.90"].false_alarms_per_photo)}`],
      ["Catalogue position error", "2.72 px (SkyBoT) → 0.26 px (Horizons)"], ["Fake vs real classifier (realism)", "AUC 0.46-0.48 (0.5 = cannot tell)"],
      ["Learnable weights", "72,145"],
    ],
    tables: [["Counting the 72,145 weights", ["Layer", "Formula", "Weights"], [
      ["conv1 3→16", "(3 x 9 + 1) x 16", "448"], ["conv2 16→16", "(16 x 9 + 1) x 16", "2,320"], ["conv3 16→32", "(16 x 9 + 1) x 32", "4,640"],
      ["conv4 32→32", "(32 x 9 + 1) x 32", "9,248"], ["conv5 32→64", "(32 x 9 + 1) x 64", "18,496"], ["dense 576→64", "576 x 64 + 64", "36,928"],
      ["output 64→1", "64 + 1", "65"], ["**Total**", "", "**72,145**"]], [3000, 3800, 2946]], CUT_TABLE],
    problems: [
      ["The labels were wrong in a hidden way", "A systematic offset hid behind chance coincidences. Lesson: check the direction of errors, not only their size, and look at the model's most confident 'mistakes'."],
      ["Fakes moved straight in pixels", "Real asteroids move straight on the sky, and the telescope shifts between photos. Fakes are now planned on the sky and converted to each photo's pixels."],
      ["Saturated stars broke the storage", "Star cores above 65,504 became 'infinity' in float16. Fix: clip before converting."],
      ["Probabilities are inflated", "Training is 16.7% real but the sky is about 3%, so a score of 0.6 was real only about 30% of the time. Satvik's base-rate adjustment corrects it."],
    ],
    handover: "Satvik gets the detections table with a CNN 'score' column (cross-fitted for training nights) and the CNN cut (0.0081 in model v4).",
    figures: [["01_label_sources.png", "Where labels come from (03_Paras/07_Diagrams/01_label_sources.svg)"], ["02_triplet_stamp.png", "The three channels of a stamp (02_triplet_stamp.svg)"],
      ["model.png", "The CNN architecture (00_Start_Here/00_How_Our_Model_Works.svg)"], ["04_cnn_results.png", "CNN results (04_cnn_results.svg)"], ["05_cross_fitting.png", "Cross-fitting (05_cross_fitting.svg)"]],
    viva: [
      ["Why three channels?", "A real new object appears in science and difference but not in reference; a bad subtraction appears in science and reference. A filter that sees all three learns that difference directly."],
      ["Why balance training but not validation?", "With 3% positives the easy answer 'always bogus' dominates training. Validation must keep the real mix so its precision and false-alarm numbers are honest."],
      ["Why not judge by training loss?", "Training data is balanced and augmented, so its loss is not comparable. We judge by validation average precision on realistic data."],
      ["Why add fakes to both science and difference?", "So that reference = science − difference stays unchanged: exactly what a new object looks like."],
      ["What is cross-fitting?", "Each training night is scored by a CNN that never saw it, so the next model learns from honest scores, like on new data."],
    ],
  },
  Satvik: {
    folder: "04_Satvik", role: "Linking, the tracklet classifier (trees) and metrics",
    marks: "Model Development (4, shared with Paras); the evaluation-metrics foundation of Model Evaluation (3, shared with Advik)",
    summary: "My part turns single blobs into moving objects and decides how we judge every model. I join the CNN's blobs into straight-line tracks, describe each track with 17 numbers, compare four classic ML models to score them, choose the operating point, check which tracks are known asteroids, and define the metrics the whole project uses.",
    steps: [
      ["Keep the CNN's blobs", "Only blobs above the CNN cut are linked. This removes most junk, so the search is about 15x cheaper (seed pairs fall from 52,000 to 3,500) and a night links in about 0.2 seconds.", "models/predict_realbogus.py"],
      ["Put every photo on one sky grid", "The telescope points slightly differently each time (up to 12 px on the test night). Every blob gets x_common, y_common: its sky position projected onto photo 1's pixel grid. Real asteroids' leftover error fell from about 2.5 px to 0.05-0.47 px.", "link/measure_sky_motion.py"],
      ["Link into tracks", "Never try all combinations (about 2 x 10^19). Seed: pairs of blobs in two photos at a plausible speed (KD-tree). Predict: where the object must be in the other photos; look it up (within 2 px). Keep: at least 3 of 5 photos, at most 1.5 px off a straight line, fitted speed at least 8 arcsec/hour. Remove duplicates.", "link/link_detections_into_tracklets.py, configs/link.yaml"],
      ["Describe each track with 17 numbers", "Motion, brightness, strength, the CNN's opinion, shape, plus (in v4) static neighbours and forced photometry. Each number has a physical reason; the last two groups came from looking at 15 false alarms by eye.", "features/compute_tracklet_features.py"],
      ["Compare four models", "Logistic regression, a decision tree, a random forest and XGBoost (300 trees, depth 3, learning rate 0.05, 80% rows and columns per tree), all cross-validated by night (GroupKFold).", "models/train_tracklet_classifier.py"],
      ["Choose the operating point", "On validation nights only: keep 95% of real tracks. The v4 cut is 0.601.", "evaluate/choose_threshold.py"],
      ["Define the metrics", "Confusion matrix, precision, recall, false-positive rate, average precision, PR vs ROC curves, calibration and base-rate adjustment. I showed why accuracy and ROC AUC mislead when real objects are rare.", "evaluate/compute_classification_metrics.py, docs/metrics_evaluation.md"],
      ["Known or unknown?", "Each blob is matched to the nearest catalogue position in its photo (4 px). A track is 'known' if at least half its blobs match the same asteroid. Every unknown goes through a written checklist, and we always look at the pictures.", "crossmatch/crossmatch_known_asteroids.py"],
    ],
    results: [
      ["Seed pairs before / after the CNN filter", "52,000 → 3,500 (about 15x cheaper)"], ["Linking time per night", "about 0.2 s"],
      ["Leftover error of real asteroids", "about 2.5 px → 0.05-0.47 px after the shared sky grid"],
      ["Track models (cross-validated AP)", "0.999 / 0.983 / 0.999 / 0.999 (LR / tree / forest / XGBoost): saturated, because 96% of tracks are easy planted fakes; not used to judge"],
      ["How versions are judged instead (test, v1 → v4)", "depth 20.53 → 21.34 mag; known found 39/44 → 40/44; unexplained 0.0 → 5.0 per night"],
      ["Test night 'false positives'", "9 of 9 were real known asteroids the labels missed"],
      ["Real search, validation night 2023-09-15", "31,953 blobs → 13 tracks → 10 known asteroids, 0 unknown"],
      ["Baseline at 90% recall", "cut 0.082, precision 19.3%, about 804 false alarms per photo"],
    ],
    tables: [["Confusion matrix of the baseline (validation, cut 0.5)", ["", "Predicted real", "Predicted junk"], [
      ["Really real (2,129)", "1,547 (true positives)", "582 (missed)"], ["Really junk (64,912)", "493 (false alarms)", "64,419 (true negatives)"]], [3200, 3273, 3273]],
      ["What the numbers say", ["Metric", "Value", "Meaning"], [
        ["Recall", "72.7%", "1,547 of 2,129 real blobs found"], ["Precision", "75.8%", "1,547 of 2,040 flagged were real"],
        ["Accuracy", "98.4%", "looks great, but 'always junk' gets 96.8%"], ["ROC AUC vs AP", "0.95 vs 0.78", "ROC hides false alarms when positives are rare"]], [2600, 2000, 5146]],
      CUT_TABLE,
      ["The 17 features", ["Group", "Numbers", "Physical idea"], [
        ["Motion", "n_detections, rate, max_residual", "steady straight motion"], ["Brightness", "mag_mean, mag_std", "brightness stays steady"],
        ["Strength", "snr_mean, snr_min", "one weak point = weak track"], ["CNN opinion", "score_mean, score_min", "what the CNN thought"],
        ["Shape", "npix_std, elongation_mean", "a point stays a point"], ["Static neighbours", "mean, max", "a mover leaves nothing behind"],
        ["Forced photometry", "min, median, std, stack_snr", "faint signal even where missed"]], [2200, 3600, 3946]]],
    problems: [
      ["The biggest bug of the project", "Bright known asteroids did not link while fakes linked perfectly. Every real asteroid had the same leftover pattern (0.6, 2.5, 2.4, 0.6, 1.7 px): identical errors across different objects point to a per-photo problem, the pointing shift. Fixed with the shared sky grid."],
      ["Stationary junk looked like slow movers", "78 'unknown' tracks moved at about 2.5 arcsec/hour: centroid wobble of stationary sources. The minimum speed must beat measurement noise, so it is 8 arcsec/hour."],
      ["My own feature had a bug", "It counted a slow asteroid's own blobs as 'static neighbours' and flagged 22% of real tracks. Found by checking real examples; fixed to 0.4%; a test was written."],
      ["Numbers can lie, pictures do not", "Three interesting unknowns on the test night looked like asteroids in the numbers but were a galaxy residual, satellite trails and a saturated-star bleed."],
    ],
    handover: "Advik gets tracks with a tracklet_score and a known_name, from the deployed function run_night_pipeline.",
    figures: [["01_linking.png", "Linking by seed and predict (04_Satvik/07_Diagrams/01_linking.svg)"], ["02_boosting.png", "Tree, forest, boosting (02_boosting.svg)"],
      ["03_operating_point.png", "Choosing the operating point (03_operating_point.svg)"], ["04_confusion_metrics.png", "Confusion matrix and metrics (04_confusion_metrics.svg)"]],
    viva: [
      ["Why is linking not machine learning?", "Physics gives the rule: over half an hour an asteroid moves in a straight line at a steady speed. It is a search, made fast with KD-trees."],
      ["Why cross-validate by night?", "Tracks from one night are alike; a random K-fold leaks them across folds and inflates scores."],
      ["Why is accuracy the wrong metric?", "With 3% positives, 'always junk' is 96.8% accurate and useless. Precision, recall and AP measure what we care about."],
      ["Why do four models tie, and why XGBoost?", "Most tracks are planted fakes, which are easy, so AP is saturated. XGBoost handles missing values and interactions without extra work; logistic regression would also be defensible."],
      ["Why does identical residual pattern mean a per-photo problem?", "Different asteroids have different motions; if their errors match photo by photo, the error belongs to the photos (pointing), not to the objects."],
    ],
  },
  Advik: {
    folder: "01_Advik", role: "System architecture, evaluation, deep search and live viewer (lead)",
    marks: "Model Evaluation and Performance (3); Documentation and Presentation (2); integration of all parts",
    summary: "My part holds everything together and decides whether the numbers can be believed. I designed the system and the hand-overs, set the testing rules, built the scoreboard every number comes from, found where asteroids were lost, tested bigger ideas fairly, built the deep search that goes 0.7 magnitude fainter, and made the live viewer and web app used in the demo.",
    steps: [
      ["Design the system", "Four stages with clear hand-overs (Aaryan → Paras → Satvik → me) and one deployed function, run_night_pipeline, used both for evaluation (with fakes) and for the real search (without), so what we test is what we deploy. Model versions are saved as bundles (CNN + track model + thresholds + provenance).", "pipeline/run_night_pipeline.py, models/save_model_bundle.py"],
      ["Set the testing rules", "Plant 300 fakes per night, run the normal pipeline, count what comes out by brightness (completeness). Only held-out nights count. Compare models at the same false-alarm level. A holdout night (2022-06-17) is opened only by the scoreboard.", "evaluate/measure_completeness.py"],
      ["Build the scoreboard", "One function computes blob AP and recall, junk kept per photo, track AP, depth, known asteroids recovered and unexplained alarms, for every version and split, and writes reports/scoreboard.csv. Before it, the same baseline showed 813 and 804 false alarms in two lessons.", "evaluate/build_scoreboard.py"],
      ["Find where asteroids are lost", "v1 looked perfect (track AP 0.999) but depth was only 20.5. Measuring each stage showed the loss was at linking: 4-of-5 linked 184/294 fakes, 3-of-5 linked 200/294 (v2). Looking at 15 false alarms by eye led to two new feature groups (v4).", "docs/scoreboard.md"],
      ["Test bigger ideas fairly", "A multi-frame CNN tied XGBoost (0.9933 vs 0.9924), so we kept the simple one. Three times more data (v5) looked 0.8 mag worse at its own threshold, but at matched false alarms v4 was equal or slightly better: more data did not help.", "docs/multiframe_model.md, docs/more_data.md"],
      ["Build the deep search", "Shift-and-stack over 4,888 trial motions on the GPU. Statistics: 6,179 false peaks per stack at 5σ but 44 at 6σ, so we use 6σ; a null test with shuffled times went from 16 false tracks to 0 after a velocity-consistency cut, keeping 20 of 20 known asteroids. Depth 22.1 vs 21.4.", "deep/stack_and_find_peaks.py, configs/deep.yaml"],
      ["Fast movers and the live monitor", "A trail-shaped filter finds fast near-Earth objects (97% of planted trails, 93% on a later night); the monitor watches the archive and runs every search on new nights.", "fast/, pipeline/run_live_monitor.py"],
      ["Live viewer and web app", "The live viewer runs the real functions stage by stage and draws each one. The web app (web/) shows the pipeline on a real window of the test night and runs the trained CNN in the browser.", "scripts/live_viewer.py, web/"],
      ["Make it reproducible", "Settings in config files, git commit in every model bundle, seeded random fakes, a pinned environment, and 166 offline tests (end-to-end, leakage, viewer equals pipeline).", "project/tests/"],
    ],
    results: [
      ["Depth on the test night (50% completeness)", "v1 20.5 → v4 21.4 → deep search 22.1"],
      ["Known asteroids, test split (v1 → v4)", "39/44 → 40/44"], ["Real search of the test night", "31 known asteroids recovered, 16.2 to 21.8 mag"],
      ["Deep search false peaks at 5σ / 6σ", "6,179 / 44 per stack"], ["Deep search peaks per stack, first → final", "47,550 → 94"],
      ["Null test false tracks per night", "16 → 0 (20 of 20 known asteroids kept)"], ["Multi-frame CNN vs XGBoost", "AP 0.9933 vs 0.9924: a tie"],
      ["Offline tests", "166"],
    ],
    tables: [["The scoreboard (project/reports/scoreboard.csv)", SB_HEAD, SCOREBOARD, SB_W]],
    problems: [
      ["A metric that cannot move", "Track AP was about 0.999 for every version because most tracks are easy planted fakes. Depth and known asteroids have room to show progress, so those became the headline metrics."],
      ["Comparing at different thresholds", "v5 looked 0.8 mag worse only because it sat at a different operating point. Always compare at the same false-alarm level."],
      ["Ghosts in the deep search", "151 of 242 unknown tracks were within 30 px of a known asteroid: echoes. Masking bright movers and the consistency cut removed them; the null test proves it."],
      ["A silent failure", "A night with 18 photos found nothing in the deep search, because it needs at least 20. Now it checks and says so."],
      ["The test night was looked at", "Real bugs were fixed using it, which is exactly why the separate holdout night exists."],
    ],
    handover: "To the team: one list of numbers with their source files, the demo (live viewer and web app), and the report chapters on evaluation, the deep search and limitations.",
    figures: [["01_architecture_full.png", "The full architecture (01_Advik/07_Diagrams/01_architecture_full.svg)"], ["02_evaluation_protocol.png", "The evaluation protocol (02_evaluation_protocol.svg)"],
      ["03_scoreboard_versions.png", "Versions on the scoreboard (03_scoreboard_versions.svg)"], ["04_shift_and_stack.png", "Shift-and-stack (04_shift_and_stack.svg)"], ["05_live_viewer_flow.png", "The live viewer (05_live_viewer_flow.svg)"]],
    viva: [
      ["Why does the scoreboard call the pipeline instead of reading saved files?", "So that it measures exactly what we deploy; saved files or notebook copies drift from the real code."],
      ["Why must thresholds live in the same bundle as the models?", "A threshold only makes sense for the model it was chosen for; mixing them silently changes the operating point."],
      ["Why 6σ and not 5σ in the deep search?", "With 4,888 motions per pixel, chance alone reaches 5σ thousands of times (6,179 false peaks); at 6σ only 44. The cut is set from pure-noise data."],
      ["What does the null test prove?", "With shuffled times no real object can line up, so anything found is false. 0 false tracks means the cuts remove chance alignments; it does not prove every candidate is real."],
      ["Why did more data not help?", "The limit is how much light one 30-second photo holds, not how much the model knows. Adding photos (the deep search) was the fix."],
    ],
  },
};

function personReport(name) {
  const p = PEOPLE[name];
  const c = [];
  c.push(...cover(`${name}: my part, start to end`, p.role, [`**Course:** ${COURSE}`, `**Team:** ${TEAM}`, `**Course marks this part covers:** ${p.marks}`, "**Date:** October 2026"]));
  c.push(...contents(["My part in one page", "Where my part sits in the project", "How I did it, step by step", "My results", "Problems I hit and how they were fixed",
    "What I handed over", "Diagrams for my slides", "Questions I should be ready for", "Use of AI"]));

  c.push(H1keep("1. My part in one page"));
  c.push(callout(AI_NOTE));
  c.push(P(p.summary));
  c.push(H2("The steps in one line each"));
  c.push(...numbered(p.steps.map(([t]) => `**${t}**`)));
  c.push(H2("My key results"));
  c.push(table(["What", "Result"], p.results, [4600, 5146]));

  c.push(H1("2. Where my part sits in the project"));
  c.push(P("The project finds moving asteroids in ZTF telescope images. Data flows through four parts, one per person: **Aaryan** (data and preprocessing) → **Paras** (labels and the CNN) → **Satvik** (linking, track model, metrics) → **Advik** (evaluation, deep search, demo). Each part hands a clear result to the next."));
  c.push(...figure("01_system_overview.png", "The whole system (00_Start_Here/11_Diagrams/01_system_overview.svg)."));
  c.push(P("The one ML architecture at the centre of the project is the real/bogus CNN (00_Start_Here/00_How_Our_Model_Works.svg): it looks at three 31 x 31 pictures of a blob and gives the probability that it is a real moving object (average precision 0.899 on validation nights, 0.938 on the test night, against 0.780 for a simple model)."));

  c.push(H1("3. How I did it, step by step"));
  c.push(...figure(`process_${name}.png`, `My process (${p.folder}/00_Process_Diagram_${name}.svg).`));
  p.steps.forEach(([title, text, code], i) => {
    c.push(H3(`Step ${i + 1}. ${title}`));
    c.push(P(text));
    c.push(new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text: "Code / file: ", font: FONT, bold: true, size: 18, color: MUTED }), new TextRun({ text: code, font: FONT, size: 18, color: MUTED, font: "Courier New" })] }));
  });

  c.push(H1("4. My results"));
  c.push(table(["What", "Result"], p.results, [4600, 5146]));
  for (const [title, head, rows, widths] of p.tables) {
    c.push(H2(title));
    c.push(table(head, rows, widths));
    if (title === CUT_TABLE[0]) c.push(P(CUT_NOTE));
    c.push(gap());
  }
  c.push(P("Every number above can be traced to a file in the project (the lesson documents in project/docs, the knowledge pack in this folder, or project/reports/scoreboard.csv). If a number changes, tell Advik so the slides and report stay in sync."));

  c.push(H1("5. Problems I hit and how they were fixed"));
  for (const [t, d] of p.problems) { c.push(H3(t)); c.push(P(d)); }

  c.push(H1("6. What I handed over"));
  c.push(P(p.handover));
  c.push(H2("Files in my folder"));
  c.push(table(["File", "What it is"], [
    [`README.md`, "my part step by step and a guide to the folder"], [`00_Process_Diagram_${name}.svg`, "how I did my part, as one picture"],
    ["01_Part_Brief.md", "what is mine, what I hand over, where the code is"], ["02_Knowledge_Pack.md", "everything I must understand, in detail"],
    ["03_Presentation_Guide.md", "slide by slide: what to say and for how long"], ["04_Viva_QA.md", "questions and model answers"],
    [`05_Handbook_${name}.html / .pdf`, "files 01-04 plus diagrams in one printable file"], [`06_Report_${name}.docx`, "this report"],
    ["07_Diagrams/", "pictures for my slides, numbered in step order"],
  ], [3600, 6146]));

  c.push(H1("7. Diagrams for my slides"));
  for (const [file, caption] of p.figures) c.push(...figure(file, caption));

  c.push(H1("8. Questions I should be ready for"));
  for (const [q, a] of p.viva) { c.push(H3(q)); c.push(P(a)); }
  c.push(P("More questions with model answers are in 04_Viva_QA.md and 00_Start_Here/06_Whole_Project_Viva_QA.md. If you do not know: say what you would check and how, never make up a number."));

  c.push(H1("9. Use of AI"));
  c.push(P("The project's code and documents were developed with the help of an AI coding assistant (Claude, by Anthropic) under Advik's direction, which the course allows for planning, learning and deployment when it is declared. Edit this paragraph so it says what **you** did yourself, for example: re-ran the notebooks of my part, changed one setting and recorded what happened, wrote my slides in my own words."));
  c.push(P("Before presenting: re-run your notebooks, change one setting, and write the explanation of each step above in your own words. That is what makes this part yours (00_Start_Here/07_AI_Use_and_Integrity.md)."));
  return makeDoc(`${name}'s report`, c);
}

(async () => {
  await save(groupReport(), path.join(ROOT, "00_Start_Here", "10_Group_Report_Asteroid_Hunter.docx"));
  for (const name of Object.keys(PEOPLE)) await save(personReport(name), path.join(ROOT, PEOPLE[name].folder, `06_Report_${name}.docx`));
})();
