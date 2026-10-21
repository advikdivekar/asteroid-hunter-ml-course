// The real/bogus CNN, run in the browser with the trained v4 weights.
// Same layers, same order as project/src/asthunt/models/define_realbogus_cnn.py:
//   conv 3->16, relu, conv 16->16, relu, maxpool      31x31 -> 15x15
//   conv 16->32, relu, conv 32->32, relu, maxpool     15x15 -> 7x7
//   conv 32->64, relu, maxpool                         7x7  -> 3x3
//   flatten 576, (dropout: off when scoring), linear 576->64, relu, linear 64->1, sigmoid

export type Layer = { name: string; shape: number[]; offset: number; length: number };
export type CnnSpec = { version: string; parameters: number; threshold: number; layers: Layer[] };
export type Tensor = { c: number; h: number; w: number; data: Float32Array };
export type Weights = { spec: CnnSpec; get: (name: string) => Float32Array };

export const STAMP = 31;
export const STAMP_SIZE = 3 * STAMP * STAMP;

export async function loadWeights(base = ""): Promise<Weights> {
  const [spec, buffer] = await Promise.all([
    fetch(`${base}/data/cnn.json`).then((r) => r.json() as Promise<CnnSpec>),
    fetch(`${base}/data/cnn_weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  return fromBuffer(spec, buffer);
}

export function fromBuffer(spec: CnnSpec, buffer: ArrayBuffer): Weights {
  const all = new Float32Array(buffer);
  const byName = new Map(spec.layers.map((l) => [l.name, all.subarray(l.offset, l.offset + l.length)]));
  return { spec, get: (name) => byName.get(name)! };
}

function conv3x3(x: Tensor, weight: Float32Array, bias: Float32Array, outC: number): Tensor {
  // 3x3 kernel, padding 1, stride 1, followed by ReLU
  const { c, h, w, data } = x;
  const out = new Float32Array(outC * h * w);
  for (let o = 0; o < outC; o++) {
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        let sum = bias[o];
        for (let k = 0; k < c; k++) {
          const wBase = ((o * c + k) * 3) * 3;
          const xBase = k * h * w;
          for (let di = -1; di <= 1; di++) {
            const ii = i + di;
            if (ii < 0 || ii >= h) continue;
            for (let dj = -1; dj <= 1; dj++) {
              const jj = j + dj;
              if (jj < 0 || jj >= w) continue;
              sum += weight[wBase + (di + 1) * 3 + (dj + 1)] * data[xBase + ii * w + jj];
            }
          }
        }
        out[(o * h + i) * w + j] = sum > 0 ? sum : 0;
      }
    }
  }
  return { c: outC, h, w, data: out };
}

function maxPool2(x: Tensor): Tensor {
  const h = Math.floor(x.h / 2), w = Math.floor(x.w / 2);
  const out = new Float32Array(x.c * h * w);
  for (let k = 0; k < x.c; k++)
    for (let i = 0; i < h; i++)
      for (let j = 0; j < w; j++) {
        const b = k * x.h * x.w;
        const a = x.data[b + 2 * i * x.w + 2 * j], bb = x.data[b + 2 * i * x.w + 2 * j + 1];
        const cc = x.data[b + (2 * i + 1) * x.w + 2 * j], d = x.data[b + (2 * i + 1) * x.w + 2 * j + 1];
        out[(k * h + i) * w + j] = Math.max(a, bb, cc, d);
      }
  return { c: x.c, h, w, data: out };
}

function linear(x: Float32Array, weight: Float32Array, bias: Float32Array, outN: number, relu: boolean): Float32Array {
  const inN = x.length;
  const out = new Float32Array(outN);
  for (let o = 0; o < outN; o++) {
    let sum = bias[o];
    for (let i = 0; i < inN; i++) sum += weight[o * inN + i] * x[i];
    out[o] = relu && sum < 0 ? 0 : sum;
  }
  return out;
}

export type Trace = {
  input: Tensor;
  block1: Tensor; // 16 x 31 x 31 (after both convs, before pooling)
  block2: Tensor; // 32 x 15 x 15
  block3: Tensor; // 64 x 7 x 7
  pooled: Tensor; // 64 x 3 x 3
  hidden: Float32Array; // 64
  logit: number;
  probability: number;
};

export function runCnn(weights: Weights, stamp: Float32Array): Trace {
  const g = weights.get;
  const input: Tensor = { c: 3, h: STAMP, w: STAMP, data: stamp };
  let x = conv3x3(input, g("0.weight"), g("0.bias"), 16);
  const block1 = conv3x3(x, g("2.weight"), g("2.bias"), 16);
  x = maxPool2(block1);
  x = conv3x3(x, g("5.weight"), g("5.bias"), 32);
  const block2 = conv3x3(x, g("7.weight"), g("7.bias"), 32);
  x = maxPool2(block2);
  const block3 = conv3x3(x, g("10.weight"), g("10.bias"), 64);
  const pooled = maxPool2(block3);
  const hidden = linear(pooled.data, g("15.weight"), g("15.bias"), 64, true);
  const logit = linear(hidden, g("17.weight"), g("17.bias"), 1, false)[0];
  return { input, block1, block2, block3, pooled, hidden, logit, probability: 1 / (1 + Math.exp(-logit)) };
}

export function stampAt(all: Float32Array, index: number): Float32Array {
  return all.subarray(index * STAMP_SIZE, (index + 1) * STAMP_SIZE);
}
