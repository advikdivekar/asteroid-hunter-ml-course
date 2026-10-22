// Loaders for the real data exported by tools/export_web_data.py (test night 2019-02-08, model v6 = classical ML;
// the v4 CNN's scores are included only for comparison).

export type Kind = "fake" | "known" | "bogus";
export type BlobNumbers = { snr: number; peak: number; npix: number; a: number; b: number; flux: number };
export type Blob = BlobNumbers & { frame: number; x: number; y: number; score: number; cnn_score: number; kind: Kind; name: string | null };
export type TrackPoint = { frame: number; x: number; y: number };
export type Track = {
  id: number; known_name: string; fake: boolean; mag: number; rate: number; residual: number; score_mean: number;
  track_score: number; accepted: boolean; points: TrackPoint[];
};
export type Scene = {
  unit: string;
  window: { x0: number; y0: number; size: number };
  frames: { frame: number; exposure: string; obsjd: number; noise: number }[];
  tracks: Track[];
  blobs: Blob[];
};
export type StampMeta = BlobNumbers & { i: number; kind: Kind; label: number; mag: number; frame: number; name: string | null; score_python: number; cnn_score: number };
export type ComparisonRow = Record<string, number> & { model: string };
export type ScoreRow = {
  version: string; split: string; nights: number; frames: number; detection_ap: number; detection_recall: number;
  bogus_kept_per_frame: number; tracklet_ap: number; mag_at_50_completeness: number; known_recovered: string; unexplained_per_night: number;
};
export type CutMetrics = {
  threshold: number; tp: number; fp: number; fn: number; tn: number;
  recall: number; precision: number; f1: number; accuracy: number; false_alarms_per_photo: number;
};
export type Metrics = { average_precision: number; roc_auc: number; always_bogus_accuracy: number; deployed: CutMetrics; at_recall: Record<"0.90" | "0.95", CutMetrics> };
export type Night = {
  unit: string; frames: number; blobs: number; blobs_real: number; kept_by_cnn: number; kept_real: number; real_total: number;
  tracklets: number; tracklets_known: number; tracklets_fake: number; fakes_planted: number;
  accepted: number; accepted_known: number; accepted_fake: number; accepted_unexplained: number;
  cnn_threshold: number; tracklet_threshold: number; scoreboard: ScoreRow[]; feature_importance: [string, number][];
  stage1_importance: [string, number][]; metrics: Metrics; cnn_metrics: Metrics; comparison: ComparisonRow[];
  junk_clusters: { cluster: number; blobs: number; share: number; snr: number; sci_centre: number; ref_centre: number; diff_centre: number; diff_min: number; diff_sharpness: number; diff_psf_match: number }[];
};

export const pct = (x: number, digits = 1) => `${(100 * x).toFixed(digits)}%`;
export const json = <T,>(path: string) => fetch(path).then((r) => r.json() as Promise<T>);
export const floats = (path: string) => fetch(path).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b));

// What each of the 29 measured numbers means, in plain words
export const FEATURE_MEANING: Record<string, string> = {
  snr: "how strong the blob is compared with the noise", peak: "brightest pixel of the blob", npix: "how many pixels the blob covers",
  a: "blob length", b: "blob width", flux: "total light in the blob",
  sci_centre: "brightness at the centre, tonight", ref_centre: "brightness at the centre, old picture", diff_centre: "brightness at the centre, difference",
  sci_peak: "brightest pixel near the centre, tonight", ref_peak: "brightest pixel near the centre, old picture", diff_peak: "brightest pixel near the centre, difference",
  diff_annulus: "difference brightness in a ring around it", ref_annulus: "old-picture brightness in a ring around it",
  ref_to_sci: "was it already in the old picture? (old ÷ tonight)", diff_to_sci: "how much of tonight's light is new (difference ÷ tonight)",
  diff_concentration: "how much of the light sits in the very centre", diff_sharpness: "one hot pixel or a smooth bump? (cosmic rays are sharp)",
  diff_min: "darkest pixel nearby (a dark hole means a bad subtraction)", diff_negative_fraction: "share of very dark pixels nearby",
  diff_dipole: "dark hole next to the bright spot (misaligned star)", diff_psf_match: "how much it looks like a star (match to the blur shape)",
  ref_psf_match: "how much the old picture looks like a star there", diff_size: "size of the bright spot", diff_elongation: "how stretched it is",
  diff_offset: "how far it is from the centre", diff_symmetry: "how round and symmetric it is", diff_bright_pixels: "pixels above 3x the noise",
  diff_roughness: "how jagged the pixels are",
};
export const featureLabel = (name: string) => FEATURE_MEANING[name] ?? (name.startsWith("pca_") ? `PCA number ${name.slice(4)}: a summary of the pixels` : name);

export const KIND_LABEL: Record<Kind, string> = { fake: "planted fake", known: "known asteroid", bogus: "junk" };
export const KIND_COLOR: Record<Kind, string> = { fake: "#ff5a1f", known: "#4f7fe0", bogus: "#8a8a86" };
export const FRAME_COLORS = ["#f4c20d", "#ff8a3d", "#ff5a1f", "#d63b8a", "#8a5cf6"];

/** Draw one 31x31 channel of a normalised stamp (values are in noise "sigmas"). */
export function drawChannel(canvas: HTMLCanvasElement, data: Float32Array, offset: number, size = 31, low = -3, high = 15) {
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(size, size);
  for (let p = 0; p < size * size; p++) {
    const v = Math.sqrt(Math.min(1, Math.max(0, (data[offset + p] - low) / (high - low))));
    const g = Math.round(255 * v);
    img.data.set([g, g, g, 255], p * 4);
  }
  ctx.putImageData(img, 0, 0);
}

/** Draw a feature map (non-negative after ReLU) with its own scale, warm colours. */
export function drawMap(canvas: HTMLCanvasElement, data: Float32Array, offset: number, h: number, w: number) {
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  let max = 1e-6;
  for (let p = 0; p < h * w; p++) max = Math.max(max, data[offset + p]);
  for (let p = 0; p < h * w; p++) {
    const v = data[offset + p] / max;
    img.data.set([Math.round(255 * Math.min(1, v * 1.4)), Math.round(255 * v * v * 0.75), Math.round(40 * v), 255], p * 4);
  }
  ctx.putImageData(img, 0, 0);
}
