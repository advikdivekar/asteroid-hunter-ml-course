"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { loadTrees, runTrees, stampOf, type TreeModel, type TreeTrace } from "@/lib/trees";
import { KIND_LABEL, drawChannel, featureLabel, floats, json, pct, type Night, type StampMeta } from "@/lib/data";

const INPUTS = [
  ["From the blob finder", "6", "strength (SNR), brightest pixel, size in pixels, length, width, total light"],
  ["Brightness in each picture", "8", "centre and peak of tonight, old and difference pictures; a ring around the blob"],
  ["Old vs new", "2", "was it already in the old picture? how much of tonight's light is new?"],
  ["Shape", "8", "star-shaped? one hot pixel? stretched? round? centred? jagged?"],
  ["Bad subtraction", "3", "a dark hole next to the bright spot, very dark pixels nearby"],
  ["Star shape in the old picture", "1", "is there a star at this spot in the old picture?"],
  ["PCA of the pixels", "20", "the central 15 x 15 pixels of all 3 pictures (675 numbers) squeezed into 20 by PCA, unsupervised"],
];

const RECIPE = [
  ["Model", "XGBoost: 600 decision trees, each at most 4 questions deep, learning rate 0.05"],
  ["Each tree sees", "80% of the training rows and 80% of the 49 inputs (random), to avoid over-fitting"],
  ["Training blobs", "53,178 (16.7% real) from the 8 training nights: every real blob plus 5 junk per real"],
  ["Validation blobs", "67,041 (3.2% real, the true mix), from 2 other nights"],
  ["How settings were chosen", "on the validation nights only: depth 4 beat 5 and 6; adding PCA raised AP by about 0.005"],
  ["Judged by", "average precision and false alarms per photo at a fixed recall, never accuracy"],
  ["Training time", "about 2 seconds on a laptop CPU (the CNN needed minutes on a GPU)"],
  ["Library", "scikit-learn (PCA) + XGBoost; here the same trees run in JavaScript"],
];

export default function Model() {
  const [meta, setMeta] = useState<StampMeta[]>([]);
  const [stamps, setStamps] = useState<Float32Array | null>(null);
  const [model, setModel] = useState<TreeModel | null>(null);
  const [night, setNight] = useState<Night | null>(null);
  const [pick, setPick] = useState(0);
  const [trace, setTrace] = useState<TreeTrace | null>(null);

  useEffect(() => {
    json<StampMeta[]>("/data/stamps.json").then((m) => {
      setMeta(m);
      setPick(m.findIndex((s) => s.kind === "known" && s.score_python > 0.99));
    });
    json<Night>("/data/night.json").then(setNight);
    floats("/data/stamps.bin").then(setStamps);
    loadTrees().then(setModel);
  }, []);

  useEffect(() => {
    if (model && stamps && meta.length) setTrace(runTrees(model, stampOf(stamps, pick), meta[pick]));
  }, [model, stamps, meta, pick]);

  const m = meta[pick];
  const threshold = model?.threshold ?? 0.005;
  const topInputs = useMemo(() => (night ? night.stage1_importance.slice(0, 10) : []), [night]);

  return (
    <>
      <section className="scene paper" style={{ paddingTop: 130 }}>
        <div className="wrap">
          <span className="kicker">The ML model · classical machine learning</span>
          <h1 className="h2" style={{ maxWidth: "26ch" }}>Measure 49 numbers per blob, then let 600 decision trees vote</h1>
          <p className="lede">
            For every blob the computer cuts three small pictures (tonight, the old sky, and the difference) and measures numbers a person
            would look at: is it star-shaped, was it already there, is it a single hot pixel? PCA adds 20 numbers that summarise the pixels.
            XGBoost, a set of gradient-boosted decision trees, turns the 49 numbers into one score: the chance that the blob is real.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="diagram" src="/diagrams/model.svg" alt="Classical ML model: three stamps, 29 measured features and 20 PCA numbers, 600 decision trees, a score from 0 to 1" style={{ marginTop: 32 }} />
          <div className="grid-2" style={{ marginTop: 32 }}>
            <div className="t-scroll">
              <table className="t">
                <thead><tr><th>the 49 inputs</th><th>how many</th><th>what they measure</th></tr></thead>
                <tbody>
                  {INPUTS.map(([a, b, c]) => <tr key={a}><td>{a}</td><td className="num">{b}</td><td>{c}</td></tr>)}
                  <tr><td><b>Total</b></td><td className="num"><b>49</b></td><td>29 hand-made features + 20 PCA numbers</td></tr>
                </tbody>
              </table>
            </div>
            <div className="t-scroll">
              <table className="t">
                <thead><tr><th>how it was trained</th><th>value</th></tr></thead>
                <tbody>{RECIPE.map(([a, b]) => <tr key={a}><td>{a}</td><td>{b}</td></tr>)}</tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section className="scene dark">
        <div className="wrap">
          <span className="kicker">Try it · the trained model runs in your browser</span>
          <h2 className="h2">Pick a blob and see what the model measured</h2>
          <p className="lede">48 real blobs from the test night: planted fakes, known asteroids and junk. Click one; every number below is computed live from the trained trees.</p>
          <div className="gallery" style={{ marginTop: 26 }}>
            {stamps && meta.map((s) => <Thumb key={s.i} stamps={stamps} meta={s} on={s.i === pick} onPick={() => setPick(s.i)} />)}
          </div>

          {trace && m && stamps && (
            <div className="grid-2" style={{ marginTop: 30 }}>
              <div className="panel">
                <div className="panel-head">
                  <span>Input · {KIND_LABEL[m.kind]}{m.name ? ` ${m.name}` : ""}</span>
                  <span>SNR {m.snr.toFixed(1)} · mag {m.mag}</span>
                </div>
                <Channels stamp={stampOf(stamps, pick)} />
                <p className="kicker" style={{ marginTop: 22 }}>Output</p>
                <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                  <span style={{ fontFamily: "var(--font-display)", fontSize: 64, lineHeight: 1, color: trace.probability >= threshold ? "var(--ember)" : "var(--text)" }}>
                    {trace.probability.toFixed(3)}
                  </span>
                  <span className="mono" style={{ fontSize: 12, color: "var(--text-2)" }}>p(real)</span>
                </div>
                <div className="meter" style={{ marginTop: 14 }} aria-label={`score ${trace.probability.toFixed(3)}`}>
                  <i style={{ width: `${100 * trace.probability}%` }} />
                  <em style={{ left: `${Math.max(0.5, 100 * threshold)}%` }} title="the cut we use" />
                </div>
                <p className="caption">
                  {trace.probability >= threshold ? "Above the cut: this blob goes on to linking." : "Below the cut: dropped as junk."} Our label says <b>{KIND_LABEL[m.kind]}</b>
                  {m.kind === "bogus" && trace.probability > 0.5
                    ? ". A confident 'real' on a junk label is worth a second look: labels can be wrong too (on the test night all 9 track-level 'false positives' were real known asteroids)."
                    : m.kind !== "bogus" && trace.probability < 0.5 ? ". The model misses this one: often a very faint fake or a blob on a messy star residual." : "."}{" "}
                  Python gave {m.score_python.toFixed(4)}. For comparison, the deep-learning CNN gave {m.cnn_score.toFixed(4)}.
                </p>

                <p className="kicker" style={{ marginTop: 22 }}>Tree 1 of 600: the questions it asked this blob</p>
                <ol style={{ margin: "8px 0 0", paddingLeft: 20, fontSize: 13.5, color: "var(--text-2)", lineHeight: 1.7 }}>
                  {trace.firstTree.map((q, k) => (
                    <li key={k}>
                      Is <b style={{ color: "var(--text)" }}>{featureLabel(q.feature)}</b> below {fmtNum(q.split)}? It is {fmtNum(q.value)}: <b style={{ color: "var(--text)" }}>{q.yes ? "yes" : "no"}</b>
                    </li>
                  ))}
                </ol>
                <p className="caption">This tree adds {trace.firstLeaf >= 0 ? "+" : ""}{trace.firstLeaf.toFixed(3)} to the vote. Each of the 600 trees asks up to 4 questions and adds a small amount; their sum becomes the score.</p>
              </div>

              <div className="panel">
                <div className="panel-head"><span>What the model measured</span><span>the 10 inputs it relies on most</span></div>
                <table className="t">
                  <thead><tr><th>input</th><th>this blob</th><th>importance</th></tr></thead>
                  <tbody>
                    {topInputs.map(([name, importance]) => {
                      const k = trace.names.indexOf(name);
                      return (
                        <tr key={name}>
                          <td>{featureLabel(name)}</td>
                          <td className="num">{fmtNum(trace.inputs[k])}</td>
                          <td style={{ width: 120 }}>
                            <div style={{ height: 8, background: "var(--raised)" }}><div style={{ height: "100%", width: `${(100 * importance) / topInputs[0][1]}%`, background: "#ff5a1f" }} /></div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p className="caption">
                  Importance = how much each input helped the trees&apos; decisions, from the saved model. The single most useful number is how much the
                  difference picture looks like a star; the next ones ask whether the spot was already in the old picture.
                </p>
              </div>
            </div>
          )}
        </div>
      </section>

      {night && <Comparison night={night} />}
    </>
  );
}

function fmtNum(v: number) {
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
}

function Comparison({ night }: { night: Night }) {
  const rows = night.comparison;
  const best = Math.max(...rows.map((r) => r["AP validation"]));
  return (
    <section className="scene paper">
      <div className="wrap">
        <span className="kicker">Why this model · classical ML compared, deep learning only as a reference</span>
        <h2 className="h2">Six models, the same nights, the same rules</h2>
        <p className="lede">
          Every model was trained on the same 8 nights and judged on the 2 validation nights; the cut for 90% recall was chosen there and the
          test night was scored once, at the end. The CNN is the deep-learning model we built first; it is shown only for comparison.
        </p>
        <div className="t-scroll" style={{ marginTop: 26 }}>
          <table className="t">
            <thead>
              <tr><th>model</th><th>AP validation</th><th>AP test</th><th>precision @90% recall</th><th>false alarms / photo @90% recall</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const chosen = r.model.includes("chosen");
                const reference = r.model.includes("CNN");
                return (
                  <tr key={r.model} style={reference ? { color: "var(--ink-3)" } : undefined}>
                    <td>{chosen ? <b>{r.model}</b> : r.model}</td>
                    <td className="num">{r["AP validation"] === best ? <b>{r["AP validation"].toFixed(3)}</b> : r["AP validation"].toFixed(3)}</td>
                    <td className="num">{r["AP test"].toFixed(3)}</td>
                    <td className="num">{pct(r["precision @90% validation"])}</td>
                    <td className="num">{Math.round(r["false alarms per photo @90% validation"])}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="caption">
          Precision and false alarms are on the validation nights at the cut that keeps 90% of real blobs. Logistic regression and the SVM do well
          on validation but fall on the test night, which is sharper and deeper than the training nights; the tree models hold up. The chosen
          XGBoost model edges the CNN on validation (AP {rows.find((r) => r.model.includes("chosen"))?.["AP validation"].toFixed(3)} vs{" "}
          {rows.find((r) => r.model.includes("CNN"))?.["AP validation"].toFixed(3)}) and is close on the test night, trains in seconds, and every one of its inputs
          has a meaning you can explain.
        </p>

        <div className="grid-2" style={{ marginTop: 40, alignItems: "start" }}>
          <div>
            <span className="kicker">Unsupervised learning · k-means on PCA</span>
            <h3 style={{ fontSize: 26, margin: "10px 0 12px" }}>What kinds of junk are there?</h3>
            <p style={{ color: "var(--ink-2)", fontSize: 15 }}>
              We took the {night.junk_clusters.reduce((s, c) => s + c.blobs, 0).toLocaleString("en-US")} junk blobs of the training set, reduced their
              pixels with PCA and grouped them with k-means into 6 clusters, without any labels. Each picture is the average stamp of one cluster.
            </p>
            <table className="t" style={{ marginTop: 14 }}>
              <thead><tr><th>cluster</th><th>share</th><th>what it looks like</th></tr></thead>
              <tbody>
                {night.junk_clusters.map((c) => (
                  <tr key={c.cluster}>
                    <td className="num">{c.cluster}</td><td className="num">{pct(c.share)}</td>
                    <td>{c.sci_centre < 5 && c.ref_centre < 5 ? "faint noise: a speck barely above the noise" : c.sci_centre < 5 ? "something in the old picture only (a star that was there before, or a bad reference)" : c.snr > 5 ? "bright star that did not subtract cleanly" : "star leftover: present in both pictures, dark hole in the difference"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="diagram" src="/data/junk_clusters.png" alt="Average stamp of each of the 6 junk clusters: science, reference and difference" />
        </div>
      </div>
    </section>
  );
}

function Thumb({ stamps, meta, on, onPick }: { stamps: Float32Array; meta: StampMeta; on: boolean; onPick: () => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (ref.current) drawChannel(ref.current, stamps, meta.i * 2883 + 2 * 961); }, [stamps, meta]);
  return (
    <button className={on ? "on" : ""} onClick={onPick} aria-label={`${KIND_LABEL[meta.kind]} blob, SNR ${meta.snr.toFixed(1)}`} aria-pressed={on}>
      <canvas ref={ref} />
      <span className={`k-${meta.kind}`}>{meta.kind}</span>
    </button>
  );
}

function Channels({ stamp }: { stamp: Float32Array }) {
  const refs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  useEffect(() => {
    refs.forEach((r, c) => r.current && drawChannel(r.current, stamp, c * 961));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stamp]);
  return (
    <div className="triplet">
      {["science", "reference", "difference"].map((n, c) => (
        <figure key={n}><canvas ref={refs[c]} /><figcaption>{n}</figcaption></figure>
      ))}
    </div>
  );
}
