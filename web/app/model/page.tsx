"use client";

import { useEffect, useRef, useState } from "react";
import { loadWeights, runCnn, stampAt, type Trace, type Weights } from "@/lib/cnn";
import { KIND_LABEL, drawChannel, drawMap, floats, json, type StampMeta } from "@/lib/data";

const LAYERS = [
  ["Input", "3 x 31 x 31", "science, reference, difference", "0"],
  ["Conv 3x3 + ReLU, twice", "16 x 31 x 31", "finds small shapes: dots, edges, rings", "2,768"],
  ["Max-pool", "16 x 15 x 15", "keeps the strongest value in each 2x2 square", "0"],
  ["Conv 3x3 + ReLU, twice", "32 x 15 x 15", "combines shapes into bigger patterns", "13,888"],
  ["Max-pool", "32 x 7 x 7", "", "0"],
  ["Conv 3x3 + ReLU", "64 x 7 x 7", "patterns over the whole blob", "18,496"],
  ["Max-pool", "64 x 3 x 3", "", "0"],
  ["Flatten + Dropout 0.3", "576", "one long list; dropout switches off 30% of inputs while training", "0"],
  ["Fully connected + ReLU", "64", "weighs all the evidence", "36,928"],
  ["Fully connected + Sigmoid", "1", "one number: p(real) from 0 to 1", "65"],
];

export default function Model() {
  const [meta, setMeta] = useState<StampMeta[]>([]);
  const [stamps, setStamps] = useState<Float32Array | null>(null);
  const [weights, setWeights] = useState<Weights | null>(null);
  const [pick, setPick] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);

  useEffect(() => {
    json<StampMeta[]>("/data/stamps.json").then((m) => {
      setMeta(m);
      setPick(m.findIndex((s) => s.kind === "known" && s.score_python > 0.99));
    });
    floats("/data/stamps.bin").then(setStamps);
    loadWeights().then(setWeights);
  }, []);

  useEffect(() => {
    if (weights && stamps && meta.length) setTrace(runCnn(weights, stampAt(stamps, pick)));
  }, [weights, stamps, meta, pick]);

  const m = meta[pick];
  const threshold = weights?.spec.threshold ?? 0.0081;

  return (
    <>
      <section className="scene paper" style={{ paddingTop: 130 }}>
        <div className="wrap">
          <span className="kicker">The one ML architecture</span>
          <h1 className="h2" style={{ maxWidth: "24ch" }}>A small convolutional neural network decides: real or junk?</h1>
          <p className="lede">
            The project uses other tools too (straight-line linking, a tree model for tracks), but the heart of it is this one network.
            It looks at three 31 x 31 pictures of a blob and gives one number: how likely it is to be a real moving object.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="diagram" src="/diagrams/model.svg" alt="CNN architecture: three stamps go through three convolution and pooling blocks, are flattened, pass two fully connected layers and give p(real)" style={{ marginTop: 32 }} />
          <div className="grid-2" style={{ marginTop: 32 }}>
            <div className="t-scroll">
              <table className="t">
                <thead><tr><th>layer</th><th>output size</th><th>what it does</th><th>weights</th></tr></thead>
                <tbody>
                  {LAYERS.map(([a, b, c, d]) => (
                    <tr key={a + b}><td>{a}</td><td className="num">{b}</td><td>{c}</td><td className="num">{d}</td></tr>
                  ))}
                  <tr><td><b>Total</b></td><td /><td /><td className="num"><b>72,145</b></td></tr>
                </tbody>
              </table>
            </div>
            <div className="t-scroll">
              <table className="t">
                <thead><tr><th>how it was trained</th><th>value</th></tr></thead>
                <tbody>
                  {[
                    ["Training stamps", "53,178 (16.7% real), from the training nights only"],
                    ["Validation stamps", "67,041 (3.2% real, the true mix)"],
                    ["Optimizer", "Adam, learning rate 0.001"],
                    ["Batch size / max epochs", "256 / 25, early stopping after 4 bad epochs"],
                    ["Loss", "binary cross-entropy"],
                    ["Augmentation", "flips and 90° rotations (a sky has no up)"],
                    ["Against overfitting", "dropout 0.3, small model, early stopping"],
                    ["Judged by", "average precision on validation nights, never training loss"],
                    ["Framework", "PyTorch; here the same weights run in JavaScript"],
                  ].map(([a, b]) => (
                    <tr key={a}><td>{a}</td><td>{b}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section className="scene dark">
        <div className="wrap">
          <span className="kicker">Try it · the trained model runs in your browser</span>
          <h2 className="h2">Pick a blob and look inside the network</h2>
          <p className="lede">
            48 real blobs from the test night: planted fakes, known asteroids, and junk. Click one. Every number below is computed live
            from the trained weights.
          </p>
          <div className="gallery" style={{ marginTop: 26 }}>
            {stamps && meta.map((s) => <Thumb key={s.i} stamps={stamps} meta={s} on={s.i === pick} onPick={() => setPick(s.i)} />)}
          </div>

          {trace && m && (
            <div className="grid-2" style={{ marginTop: 30 }}>
              <div className="panel">
                <div className="panel-head">
                  <span>Input · {KIND_LABEL[m.kind]}{m.name ? ` ${m.name}` : ""}</span>
                  <span>SNR {m.snr} · mag {m.mag}</span>
                </div>
                <Channels trace={trace} />
                <p className="kicker" style={{ marginTop: 22 }}>Output</p>
                <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                  <span style={{ fontFamily: "var(--font-display)", fontSize: 64, lineHeight: 1, color: trace.probability >= threshold ? "var(--ember)" : "var(--text)" }}>
                    {trace.probability.toFixed(3)}
                  </span>
                  <span className="mono" style={{ fontSize: 12, color: "var(--text-2)" }}>p(real)</span>
                </div>
                <div className="meter" style={{ marginTop: 14 }} aria-label={`score ${trace.probability.toFixed(3)}`}>
                  <i style={{ width: `${100 * trace.probability}%` }} />
                  <em style={{ left: `${100 * threshold}%` }} title="the cut we use" />
                </div>
                <p className="caption">
                  {trace.probability >= threshold ? "Above the cut: this blob goes on to linking." : "Below the cut: dropped as junk."}{" "}
                  Our label says <b>{KIND_LABEL[m.kind]}</b>
                  {m.kind === "bogus" && trace.probability > 0.5
                    ? ". A confident 'real' on a junk label is worth a second look: labels can be wrong too. On the test night all 9 track-level 'false positives' turned out to be real known asteroids."
                    : m.kind !== "bogus" && trace.probability < 0.5
                      ? ". The model misses this one: often a very faint fake or a blob on a messy star residual."
                      : "."}{" "}
                  Python gave {m.score_python.toFixed(4)}; the browser gives {trace.probability.toFixed(4)}.
                </p>
              </div>
              <div className="panel">
                <div className="panel-head"><span>Inside: feature maps</span><span>brighter = stronger response</span></div>
                <Maps title="Block 1 · 16 maps of 31 x 31 · small shapes" t={trace.block1} cols={8} />
                <Maps title="Block 2 · 32 maps of 15 x 15 · bigger patterns" t={trace.block2} cols={8} />
                <Maps title="Block 3 · 64 maps of 7 x 7 · the whole blob" t={trace.block3} cols={16} />
                <p className="caption">
                  Each little square is what one learned filter &quot;sees&quot;. Early filters react to the dot and to edges; later ones respond to
                  whole patterns, such as &quot;bright in science and difference but empty in the reference&quot;, which is what a real new object looks like.
                </p>
              </div>
            </div>
          )}
        </div>
      </section>
    </>
  );
}

function Thumb({ stamps, meta, on, onPick }: { stamps: Float32Array; meta: StampMeta; on: boolean; onPick: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (ref.current) drawChannel(ref.current, stamps, meta.i * 2883 + 2 * 961); }, [stamps, meta]);
  return (
    <button className={on ? "on" : ""} onClick={onPick} aria-label={`${KIND_LABEL[meta.kind]} blob, SNR ${meta.snr}`} aria-pressed={on}>
      <canvas ref={ref} />
      <span className={`k-${meta.kind}`}>{meta.kind}</span>
    </button>
  );
}

function Channels({ trace }: { trace: Trace }) {
  const refs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  useEffect(() => {
    refs.forEach((r, c) => r.current && drawChannel(r.current, trace.input.data, c * 961));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace]);
  return (
    <div className="triplet">
      {["science", "reference", "difference"].map((n, c) => (
        <figure key={n}><canvas ref={refs[c]} /><figcaption>{n}</figcaption></figure>
      ))}
    </div>
  );
}

function Maps({ title, t, cols }: { title: string; t: Trace["block1"]; cols: number }) {
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    wrap.current?.querySelectorAll("canvas").forEach((c, k) => drawMap(c, t.data, k * t.h * t.w, t.h, t.w));
  }, [t]);
  return (
    <div style={{ marginBottom: 16 }}>
      <p className="mono" style={{ fontSize: 10.5, color: "var(--text-3)", textTransform: "uppercase", margin: "0 0 8px" }}>{title}</p>
      <div ref={wrap} className="maps" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {Array.from({ length: t.c }, (_, k) => <canvas key={k} />)}
      </div>
    </div>
  );
}
