// Checks both browser models give the same scores as Python:
//   the deployed classical model (lib/trees.ts) and the CNN kept for comparison (lib/cnn.ts).
// Needs Node 22.18+ (runs TypeScript directly).
import { readFileSync } from "node:fs";

const { fromBuffer, runCnn, stampAt } = await import("../lib/cnn.ts");
const { runTrees } = await import("../lib/trees.ts");
const floats = (path) => { const b = readFileSync(path); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

const trees = JSON.parse(readFileSync("public/data/trees.json"));
const raw = readFileSync("public/data/cnn_weights.bin");
const cnn = fromBuffer(JSON.parse(readFileSync("public/data/cnn.json")), raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
let worstTrees = 0, worstCnn = 0, n = 0, nearSplit = 0, decisionsDiffer = 0;
for (const [metaFile, stampFile] of [["public/data/stamps.json", "public/data/stamps.bin"], ["public/data/scene.json", "public/data/window_stamps.bin"]]) {
  const meta = JSON.parse(readFileSync(metaFile));
  const blobs = Array.isArray(meta) ? meta : meta.blobs;
  const stamps = floats(stampFile);
  blobs.forEach((blob, i) => {
    const stamp = stampAt(stamps, i);
    const js = runTrees(trees, stamp, blob).probability, py = blob.score_python ?? blob.score;
    worstTrees = Math.max(worstTrees, Math.abs(js - py));
    if (Math.abs(js - py) > 1e-4) nearSplit++;                                   // a feature on a split value: float32 vs float64 rounding
    if ((js >= trees.threshold) !== (py >= trees.threshold)) decisionsDiffer++;
    worstCnn = Math.max(worstCnn, Math.abs(runCnn(cnn, stamp).probability - blob.cnn_score));
    n++;
  });
}
console.log(`${n} stamps · classical model: ${decisionsDiffer} keep/drop decisions differ from Python, ${nearSplit} scores differ by more ` +
  `than 0.0001 (largest ${worstTrees.toFixed(4)}) · CNN: largest difference ${worstCnn.toExponential(2)}`);
// Python computes the features in float32, the browser in float64; a stamp whose feature sits exactly on a split can take the
// other branch of one tree. That moves its score slightly but must never change the decision at the cut.
process.exit(decisionsDiffer === 0 && worstTrees < 0.02 && worstCnn < 1e-3 ? 0 : 1);
