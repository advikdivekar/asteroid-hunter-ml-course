// Checks the browser CNN (lib/cnn.ts) gives the same scores as PyTorch did (stamps.json: score_python).
// Needs Node 22.18+ (runs TypeScript directly).
import { readFileSync } from "node:fs";

const { fromBuffer, runCnn, stampAt } = await import("../lib/cnn.ts");
const spec = JSON.parse(readFileSync("public/data/cnn.json"));
const raw = readFileSync("public/data/cnn_weights.bin");
const weights = fromBuffer(spec, raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const meta = JSON.parse(readFileSync("public/data/stamps.json"));
const buf = readFileSync("public/data/stamps.bin");
const stamps = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
let worst = 0;
for (const m of meta) worst = Math.max(worst, Math.abs(runCnn(weights, stampAt(stamps, m.i)).probability - m.score_python));
console.log(`${meta.length} stamps, largest difference from PyTorch: ${worst.toExponential(2)}`);
process.exit(worst < 1e-3 ? 0 : 1);
