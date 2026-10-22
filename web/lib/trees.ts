// The classical real/bogus model (bundle v6), run in the browser: the same 29 features as
// project/src/asthunt/features/compute_stamp_features.py, the same PCA, then the 600 XGBoost trees.

const STAMP = 31;                       // stamps are 3 x 31 x 31 (as in lib/cnn.ts; no import, so Node can run this file directly)
const STAMP_SIZE = 3 * STAMP * STAMP;

export const BLOB_COLUMNS = ["snr", "peak", "npix", "a", "b", "flux"] as const;
export type BlobNumbers = Record<(typeof BLOB_COLUMNS)[number], number>;
export type Tree = { f: number[]; t: number[]; y: number[]; n: number[]; m: number[]; v: number[] };
export type TreeModel = {
  version: string; threshold: number; base_margin: number; feature_names: string[];
  pca_mean: number[]; pca_components: number[][]; trees: Tree[];
};

export const STAMP_FEATURES = [
  "sci_centre", "ref_centre", "diff_centre", "sci_peak", "ref_peak", "diff_peak", "diff_annulus", "ref_annulus",
  "ref_to_sci", "diff_to_sci", "diff_concentration", "diff_sharpness", "diff_min", "diff_negative_fraction", "diff_dipole",
  "diff_psf_match", "ref_psf_match", "diff_size", "diff_elongation", "diff_offset", "diff_symmetry", "diff_bright_pixels", "diff_roughness",
] as const;

export async function loadTrees(base = ""): Promise<TreeModel> {
  return fetch(`${base}/data/trees.json`).then((r) => r.json() as Promise<TreeModel>);
}

const C = Math.floor(STAMP / 2);
type Px = { i: number; j: number; x: number; y: number; r: number };
const PIXELS: Px[] = [];
for (let i = 0; i < STAMP; i++) for (let j = 0; j < STAMP; j++) PIXELS.push({ i, j, x: j - C, y: i - C, r: Math.hypot(j - C, i - C) });
const CENTRE = PIXELS.filter((p) => p.r <= 1.5);
const CORE = PIXELS.filter((p) => p.r <= 3.5);
const ANNULUS = PIXELS.filter((p) => p.r >= 6 && p.r <= 10);
const WINDOW = PIXELS.filter((p) => Math.abs(p.x) <= 5 && Math.abs(p.y) <= 5);
const PSF = WINDOW.map((p) => Math.exp(-(p.r ** 2) / (2 * 1.2 ** 2)));

/** The 23 stamp features for one normalized stamp (3 x 31 x 31, channel 0 science, 1 reference, 2 difference). */
export function stampFeatures(stamp: Float32Array): Record<(typeof STAMP_FEATURES)[number], number> {
  const at = (channel: number, p: Px) => stamp[channel * STAMP * STAMP + p.i * STAMP + p.j];
  const values = (channel: number, set: Px[]) => set.map((p) => at(channel, p));
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const max = (a: number[]) => a.reduce((m, v) => Math.max(m, v), -Infinity);
  const sum = (a: number[]) => a.reduce((s, v) => s + v, 0);

  const sci_centre = mean(values(0, CENTRE)), ref_centre = mean(values(1, CENTRE)), diff_centre = mean(values(2, CENTRE));
  const sci_peak = max(values(0, CORE)), ref_peak = max(values(1, CORE)), diff_peak = max(values(2, CORE));
  const sciLevel = Math.max(sci_centre, 1);
  const patch = values(2, WINDOW);
  const diff_min = patch.reduce((m, v) => Math.min(m, v), Infinity);

  const weights = patch.map((v) => Math.max(v, 0));
  const total = Math.max(sum(weights), 1e-6);
  const mx = sum(weights.map((w, k) => w * WINDOW[k].x)) / total, my = sum(weights.map((w, k) => w * WINDOW[k].y)) / total;
  const sxx = sum(weights.map((w, k) => w * WINDOW[k].x ** 2)) / total - mx ** 2;
  const syy = sum(weights.map((w, k) => w * WINDOW[k].y ** 2)) / total - my ** 2;
  const sxy = sum(weights.map((w, k) => w * WINDOW[k].x * WINDOW[k].y)) / total - mx * my;
  const spread = Math.sqrt(Math.max((sxx - syy) ** 2 / 4 + sxy ** 2, 0));
  const major = (sxx + syy) / 2 + spread, minor = Math.max((sxx + syy) / 2 - spread, 1e-3);

  const rotated = WINDOW.map((p) => stamp[2 * STAMP * STAMP + (STAMP - 1 - p.i) * STAMP + (STAMP - 1 - p.j)]);
  let rough = 0;
  for (let i = C - 5; i <= C + 5; i++)
    for (let j = C - 5; j < C + 5; j++) rough += Math.abs(stamp[2 * STAMP * STAMP + i * STAMP + j + 1] - stamp[2 * STAMP * STAMP + i * STAMP + j]);

  return {
    sci_centre, ref_centre, diff_centre, sci_peak, ref_peak, diff_peak,
    diff_annulus: mean(values(2, ANNULUS)), ref_annulus: mean(values(1, ANNULUS)),
    ref_to_sci: ref_centre / sciLevel, diff_to_sci: diff_centre / sciLevel,
    diff_concentration: sum(values(2, CENTRE)) / Math.max(Math.abs(sum(values(2, CORE))), 1),
    diff_sharpness: diff_peak / Math.max(diff_centre, 1),
    diff_min, diff_negative_fraction: patch.filter((v) => v < -3).length / patch.length,
    diff_dipole: -diff_min / Math.max(diff_peak, 1),
    diff_psf_match: correlation(patch, PSF), ref_psf_match: correlation(values(1, WINDOW), PSF),
    diff_size: Math.sqrt(Math.max(major + minor, 0)), diff_elongation: Math.sqrt(Math.max(major, 0) / minor), diff_offset: Math.hypot(mx, my),
    diff_symmetry: correlation(patch, rotated), diff_bright_pixels: patch.filter((v) => v > 3).length,
    diff_roughness: rough / (11 * 10) / Math.max(Math.abs(diff_centre), 1),
  };
}

function correlation(a: number[], b: number[]) {
  const ma = a.reduce((s, v) => s + v, 0) / a.length, mb = b.reduce((s, v) => s + v, 0) / b.length;
  let ab = 0, aa = 0, bb = 0;
  for (let k = 0; k < a.length; k++) { ab += (a[k] - ma) * (b[k] - mb); aa += (a[k] - ma) ** 2; bb += (b[k] - mb) ** 2; }
  return ab / Math.max(Math.sqrt(aa) * Math.sqrt(bb), 1e-6);
}

/** The 20 PCA numbers: the stamp's central 15 x 15 pixels (all 3 channels) projected on the learned directions. */
export function pcaNumbers(model: TreeModel, stamp: Float32Array): number[] {
  const pixels: number[] = [];
  for (let c = 0; c < 3; c++) for (let i = 8; i < 23; i++) for (let j = 8; j < 23; j++) pixels.push(stamp[c * STAMP * STAMP + i * STAMP + j]);
  return model.pca_components.map((direction) => direction.reduce((s, w, k) => s + w * (pixels[k] - model.pca_mean[k]), 0));
}

export type TreeTrace = { inputs: number[]; names: string[]; margin: number; probability: number; firstTree: { feature: string; value: number; split: number; yes: boolean }[]; firstLeaf: number };

/** Score one blob: features, PCA, then sum the leaves of every tree. Also returns the questions tree 1 asked. */
export function runTrees(model: TreeModel, stamp: Float32Array, blob: BlobNumbers): TreeTrace {
  const features = stampFeatures(stamp);
  const inputs = [...BLOB_COLUMNS.map((c) => blob[c]), ...STAMP_FEATURES.map((n) => features[n]), ...pcaNumbers(model, stamp)].map(Math.fround);
  let margin = model.base_margin;
  let firstTree: TreeTrace["firstTree"] = [];
  let firstLeaf = 0;
  model.trees.forEach((tree, k) => {
    let node = 0;
    while (tree.f[node] >= 0) {
      const value = inputs[tree.f[node]];
      const yes = Number.isNaN(value) ? tree.m[node] === tree.y[node] : value < tree.t[node];
      if (k === 0) firstTree.push({ feature: model.feature_names[tree.f[node]], value, split: tree.t[node], yes });
      node = Number.isNaN(value) ? tree.m[node] : yes ? tree.y[node] : tree.n[node];
    }
    if (k === 0) firstLeaf = tree.v[node];
    margin += tree.v[node];
  });
  if (!firstTree.length) firstTree = [];
  return { inputs, names: model.feature_names, margin, probability: 1 / (1 + Math.exp(-margin)), firstTree, firstLeaf };
}

export const stampOf = (all: Float32Array, index: number) => all.subarray(index * STAMP_SIZE, (index + 1) * STAMP_SIZE);
