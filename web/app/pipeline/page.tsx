"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { loadTrees, runTrees, stampOf, type TreeModel } from "@/lib/trees";
import { FRAME_COLORS, KIND_COLOR, KIND_LABEL, drawChannel, floats, json, pct, type Night, type Scene } from "@/lib/data";

const STEPS = [
  ["Five photos of one patch of sky", "The telescope (ZTF, at Palomar) photographs the same patch of sky every few minutes, 30 seconds each. We take 5 photos of one night. Press play: one dot creeps across. That is a real asteroid."],
  ["Subtract the old sky", "For every photo ZTF also gives a REFERENCE: a very deep old picture of the same sky. DIFFERENCE = tonight minus reference. Stars are in both, so they cancel. What is left is what changed: the asteroid, plus noise and junk."],
  ["Circle every blob", "A program circles every spot brighter than 2x the noise in the difference photo. The cut is low on purpose so nothing real is missed. Most circles are junk; that is fine, the next step sorts them."],
  ["Cut a small picture of each blob", "Around each blob we cut a 31x31 pixel square from all three photos: science, reference, difference. A real moving object shows in science and difference but NOT in the reference. Click any circle to see its three pictures."],
  ["A machine-learning model decides: real or junk?", "For every blob the computer measures 29 numbers from its three pictures (is it star-shaped? was it already in the old picture? is it one hot pixel?) plus 20 PCA numbers that summarise the pixels. Then XGBoost, 600 small decision trees, gives a score from 0 (junk) to 1 (real). Press the button: the real trained model runs right here in your browser. Blobs above the cut go on. The cut is set low on purpose, so almost no real object is lost."],
  ["Join blobs into straight-line tracks", "An asteroid moves in a straight line at a steady speed. We look for kept blobs in different photos that line up like that and join them into a track. Each colour is one photo."],
  ["Score each track and check the catalogue", "Each track gets 17 numbers (speed, straightness, steady brightness, the first model's opinion...). A second model, XGBoost (many small decision trees), turns them into one track score. Then we compare with the list of known asteroids."],
  ["The result for the whole night", "The same steps run on the full photo (3072 x 3080 pixels), not just this small window. Here is what happened on the whole test night, with 300 fake asteroids planted so we know the right answer."],
] as const;

const fmt = (n: number) => n.toLocaleString("en-US");

export default function Pipeline() {
  const [step, setStep] = useState(0);
  const [scene, setScene] = useState<Scene | null>(null);
  const [night, setNight] = useState<Night | null>(null);
  const [stamps, setStamps] = useState<Float32Array | null>(null);
  const [model, setModel] = useState<TreeModel | null>(null);
  const [frame, setFrame] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<"science" | "reference" | "difference">("science");
  const [picked, setPicked] = useState<number | null>(null);
  const [scores, setScores] = useState<Float32Array | null>(null);
  const [runMs, setRunMs] = useState(0);
  const [running, setRunning] = useState(false);
  const [cut, setCut] = useState(0.0081);

  useEffect(() => {
    json<Scene>("/data/scene.json").then(setScene);
    json<Night>("/data/night.json").then((n) => { setNight(n); setCut(n.cnn_threshold); });
    floats("/data/window_stamps.bin").then(setStamps);
    loadTrees().then(setModel);
    const asked = Number(new URLSearchParams(window.location.search).get("step"));   // deep link, e.g. /pipeline/?step=5
    if (asked >= 1 && asked <= STEPS.length) setStep(asked - 1);
  }, []);

  useEffect(() => {
    if (step === 4 && !scores && model && stamps && scene && new URLSearchParams(window.location.search).has("run")) runModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, model, stamps, scene]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setFrame((f) => (f % 5) + 1), 650);
    return () => clearInterval(id);
  }, [playing]);

  // what each step shows by default
  useEffect(() => {
    setView(step === 0 ? "science" : step === 1 ? view : "difference");
    if (step >= 2) setPlaying(false);
    if (step === 5 || step === 6) setFrame(5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step]);

  useEffect(() => {
    if (step === 3 && picked === null && scene) {
      const known = scene.blobs.findIndex((b) => b.kind === "known" && b.frame === frame);
      setPicked(known >= 0 ? known : 0);
    }
  }, [step, picked, scene, frame]);

  const runModel = () => {
    if (!model || !stamps || !scene) return;
    setRunning(true);
    setTimeout(() => {
      const t0 = performance.now();
      const out = new Float32Array(scene.blobs.length);
      scene.blobs.forEach((b, i) => (out[i] = runTrees(model, stampOf(stamps, i), b).probability));
      setRunMs(performance.now() - t0);
      setScores(out);
      setRunning(false);
    }, 30);
  };

  const inFrame = useMemo(() => (scene ? scene.blobs.map((b, i) => ({ ...b, i })).filter((b) => b.frame === frame) : []), [scene, frame]);
  const kept = (i: number) => (scores ? scores[i] >= cut : true);
  const stats = useMemo(() => {
    if (!scene || !scores) return null;
    const keep = scene.blobs.map((_, i) => scores[i] >= cut);
    const real = scene.blobs.map((b) => b.kind !== "bogus");
    return {
      kept: keep.filter(Boolean).length,
      dropped: keep.filter((k) => !k).length,
      realKept: keep.filter((k, i) => k && real[i]).length,
      real: real.filter(Boolean).length,
      maxDiff: Math.max(...scene.blobs.map((b, i) => Math.abs(b.score - scores[i]))),
    };
  }, [scene, scores, cut]);

  const imgSrc = `/data/scene/f${frame}_${view}.png`;
  const size = scene?.window.size ?? 160;

  return (
    <>
      <section className="scene" style={{ paddingBottom: 40, paddingTop: 130 }}>
        <div className="wrap">
          <span className="kicker">How it works · real data · test night 2019-02-08</span>
          <h1 className="h2" style={{ maxWidth: "26ch" }}>
            Follow one real asteroid through the whole pipeline
          </h1>
          <p className="lede">
            Every picture here is a real 160 x 160 pixel window of a ZTF photo from our untouched test night. Every score comes from our
            trained model. Click the steps on the left in order.
          </p>
        </div>
      </section>

      <section className="scene dark" style={{ paddingTop: 40 }}>
        <div className="wrap stepper">
          <ol className="steps">
            {STEPS.map(([t, d], i) => (
              <li key={t}>
                <button className={i === step ? "on" : ""} onClick={() => setStep(i)} aria-current={i === step}>
                  <span className="num">{String(i + 1).padStart(2, "0")}</span>
                  <span>
                    <span className="t">{t}</span>
                    <span className="d">{d}</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>

          <div>
            {step < 7 && (
              <div className="panel">
                <div className="panel-head">
                  <span>
                    Step {step + 1} · photo {frame} of 5 · {view}
                  </span>
                  <span>
                    window x {scene?.window.x0}, y {scene?.window.y0} · 1 px = 1 arcsec
                  </span>
                </div>

                <div className="row" style={{ marginBottom: 14, justifyContent: "space-between" }}>
                  <div className="frames" role="group" aria-label="Photo">
                    {[1, 2, 3, 4, 5].map((k) => (
                      <button key={k} className={k === frame ? "on" : ""} onClick={() => { setFrame(k); setPlaying(false); setPicked(null); }}>
                        {k}
                      </button>
                    ))}
                    <button onClick={() => setPlaying((p) => !p)} style={{ width: 64 }} aria-label={playing ? "Pause" : "Play"}>
                      {playing ? "pause" : "play"}
                    </button>
                  </div>
                  {step === 1 && (
                    <div className="seg" role="group" aria-label="Which picture">
                      {(["science", "reference", "difference"] as const).map((v) => (
                        <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>
                          {v}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="sky">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imgSrc} alt={`${view} image, photo ${frame}`} />
                  {scene && (
                    <svg viewBox={`0 0 ${size} ${size}`} aria-hidden={step < 2}>
                      {step === 0 && scene.tracks.filter((t) => t.known_name).map((t) => {
                        const p = t.points.find((q) => q.frame === frame);
                        return p ? <circle key={t.id} cx={p.x + 0.5} cy={p.y + 0.5} r={7} fill="none" stroke="#4f7fe0" strokeWidth={0.8} strokeDasharray="2 1.5" /> : null;
                      })}
                      {(step === 2 || step === 3 || step === 4) &&
                        inFrame.map((b) => {
                          const score = scores ? scores[b.i] : null;
                          const on = step === 4 && scores ? kept(b.i) : true;
                          const color = step === 4 && scores ? (on ? "#ff5a1f" : "#b8b8bc") : "#d8d8d4";
                          return (
                            <circle
                              key={b.i}
                              className="ring"
                              cx={b.x + 0.5}
                              cy={b.y + 0.5}
                              r={picked === b.i && step === 3 ? 6 : 4}
                              fill="transparent"
                              stroke={picked === b.i && step === 3 ? "#f4c20d" : color}
                              strokeWidth={picked === b.i && step === 3 ? 1.2 : on ? 0.8 : 0.6}
                              strokeDasharray={on ? undefined : "1.5 1.5"}
                              onClick={() => { setPicked(b.i); if (step !== 4) setStep(3); }}
                            >
                              <title>{`blob in photo ${b.frame} · SNR ${b.snr}${score !== null ? ` · model ${score.toFixed(3)}` : ""} · label: ${KIND_LABEL[b.kind]}${b.name ? ` ${b.name}` : ""}`}</title>
                            </circle>
                          );
                        })}
                      {(step === 5 || step === 6) && (
                        <>
                          {scene.blobs.map((b, i) =>
                            kept(i) ? <circle key={i} cx={b.x + 0.5} cy={b.y + 0.5} r={0.9} fill={FRAME_COLORS[b.frame - 1]} opacity={0.55} /> : null,
                          )}
                          {scene.tracks.map((t) => {
                            const dim = step === 6 && !t.accepted;               // after scoring: dropped chance tracks fade out
                            const start = t.points[0];
                            const labelY = start.y + 9 > size - 3 ? start.y - 5 : start.y + 9;
                            const labelX = Math.min(Math.max(start.x - 6, 2), size - 42);
                            return (
                              <g key={t.id} opacity={dim ? 0.3 : 1}>
                                <polyline points={t.points.map((p) => `${p.x + 0.5},${p.y + 0.5}`).join(" ")} fill="none" stroke={t.known_name ? "#4f7fe0" : "#ffffff"} strokeWidth={0.6} strokeDasharray={dim ? "1.5 1.5" : undefined} />
                                {t.points.map((p) => (
                                  <circle key={p.frame} cx={p.x + 0.5} cy={p.y + 0.5} r={2} fill={FRAME_COLORS[p.frame - 1]} stroke="#000" strokeWidth={0.4} />
                                ))}
                                {(step === 5 || t.accepted) && (
                                  <text x={labelX} y={labelY} fill="#fff" fontSize={4.5} fontFamily="JetBrains Mono, monospace" stroke="#000" strokeWidth={1.2} paintOrder="stroke">
                                    {t.known_name || `track ${t.id}`}
                                  </text>
                                )}
                              </g>
                            );
                          })}
                        </>
                      )}
                    </svg>
                  )}
                </div>

                {step === 0 && (
                  <p className="caption">
                    The dashed blue circle follows the known asteroid <b>1996 XG5</b> (brightness 17.5). Photo 1 was taken about two hours
                    before photos 2 to 5, so the jump from 1 to 2 is bigger. All pictures are lined up on the same patch of sky: the telescope points
                    slightly differently each time, and correcting for that was one of the biggest fixes of the project.
                  </p>
                )}
                {step === 1 && (
                  <p className="caption">
                    Switch between the three pictures. In <b>difference</b>, the stars are gone; the bright dot left near the middle is the asteroid.
                    Small rings and dark holes are leftovers of bright stars that did not subtract perfectly: that is the junk the model must learn to ignore.
                  </p>
                )}
                {step === 2 && scene && (
                  <div className="readouts">
                    <div><span className="mono">blobs in this window, photo {frame}</span><b>{inFrame.length}</b></div>
                    <div><span className="mono">all 5 photos, this window</span><b>{scene.blobs.length}</b></div>
                    <div className="hot"><span className="mono">whole night, full photos</span><b>{night ? fmt(night.blobs) : "…"}</b></div>
                  </div>
                )}
                {step === 3 && <Triplet stamps={stamps} scene={scene} picked={picked} score={scores && picked !== null ? scores[picked] : null} />}
                {step === 4 && (
                  <div style={{ marginTop: 16 }}>
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <button className="cta fire" onClick={runModel} disabled={!model || !stamps || running}>
                        <span>{running ? "Running…" : scores ? "Run again" : `Run the model on all ${scene?.blobs.length ?? ""} blobs`}</span>
                        <b>▶</b>
                      </button>
                      {scores && <span className="mono" style={{ fontSize: 12, color: "var(--text-2)" }}>done in {runMs.toFixed(0)} ms, in your browser</span>}
                    </div>
                    {scores && stats && (
                      <>
                        <div style={{ marginTop: 18 }}>
                          <label className="mono" style={{ fontSize: 11, color: "var(--text-3)" }} htmlFor="cut">
                            CUT: KEEP BLOBS WITH SCORE ≥ {cut < 0.01 ? cut.toFixed(4) : cut.toFixed(2)} {Math.abs(cut - (night?.cnn_threshold ?? 0)) < 1e-6 ? "(the value we use)" : ""}
                          </label>
                          <input id="cut" className="range" type="range" min={-4} max={-0.01} step={0.01} value={Math.log10(cut)} onChange={(e) => setCut(10 ** Number(e.target.value))} />
                        </div>
                        <div className="readouts">
                          <div className="hot"><span className="mono">kept (go on)</span><b>{stats.kept} of {scene?.blobs.length}</b></div>
                          <div><span className="mono">recall: real ones kept</span><b>{stats.realKept} of {stats.real}</b></div>
                          <div><span className="mono">precision: kept that are real</span><b>{stats.kept ? pct(stats.realKept / stats.kept, 0) : "–"}</b></div>
                        </div>
                        <p className="caption">
                          Solid orange = kept, dashed grey = dropped. Hover a circle for its score. Always read recall and precision together:
                          at the cut we use, almost every real blob survives (high recall) but most kept blobs are still junk (low precision), because a
                          missed asteroid can never be recovered later while junk is removed by the next steps. Move the cut up and precision rises
                          while recall falls. Every keep/drop decision matches the Python model.
                        </p>
                      </>
                    )}
                    <div className="legend">
                      <span><i style={{ borderColor: "#ff5a1f" }} />kept</span>
                      <span><i style={{ borderColor: "#b8b8bc", borderStyle: "dashed" }} />dropped</span>
                    </div>
                  </div>
                )}
                {step === 5 && scene && <TrackTable scene={scene} />}
                {step === 6 && scene && night && <Verdicts scene={scene} night={night} />}
              </div>
            )}
            {step === 7 && night && <Funnel night={night} />}
            <div className="row" style={{ marginTop: 18, justifyContent: "space-between" }}>
              <button className="cta" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>
                <span>Back</span>
                <b>←</b>
              </button>
              {step < 7 ? (
                <button className="cta fire" onClick={() => setStep((s) => s + 1)}>
                  <span>Next: {STEPS[step + 1][0]}</span>
                  <b>→</b>
                </button>
              ) : (
                <Link href="/model/" className="cta fire">
                  <span>Look inside the model</span>
                  <b>→</b>
                </Link>
              )}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}

function Triplet({ stamps, scene, picked, score }: { stamps: Float32Array | null; scene: Scene | null; picked: number | null; score: number | null }) {
  const refs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  useEffect(() => {
    if (!stamps || picked === null) return;
    refs.forEach((r, c) => r.current && drawChannel(r.current, stamps, picked * 2883 + c * 961));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stamps, picked]);
  if (!scene || picked === null) return null;
  const b = scene.blobs[picked];
  return (
    <div style={{ marginTop: 16 }}>
      <div className="triplet">
        {["science", "reference", "difference"].map((n, c) => (
          <figure key={n}>
            <canvas ref={refs[c]} />
            <figcaption>{n}</figcaption>
          </figure>
        ))}
      </div>
      <p className="caption">
        Blob in photo {b.frame} · strength (SNR) {b.snr} · our label: <b style={{ color: KIND_COLOR[b.kind] }}>{KIND_LABEL[b.kind]}{b.name ? ` ${b.name}` : ""}</b>
        {score !== null ? ` · model score ${score.toFixed(3)}` : " · go to step 5 to score it"}. From these three 31 x 31 pictures the model measures its 29 numbers and 20 PCA numbers.
      </p>
    </div>
  );
}

function TrackTable({ scene }: { scene: Scene }) {
  return (
    <div style={{ marginTop: 16 }}>
      <div className="legend">
        {FRAME_COLORS.map((c, i) => (
          <span key={c}><i style={{ borderColor: c, background: c }} />photo {i + 1}</span>
        ))}
      </div>
      <div className="t-scroll" style={{ marginTop: 12 }}>
        <table className="t">
          <thead>
            <tr><th>track</th><th>photos</th><th>speed</th><th>off the line</th><th>brightness</th><th>model score (mean)</th></tr>
          </thead>
          <tbody>
            {scene.tracks.map((t) => (
              <tr key={t.id}>
                <td>{t.known_name || `track ${t.id}`}</td>
                <td className="num">{t.points.length} of 5</td>
                <td className="num">{t.rate}″/h</td>
                <td className="num">{t.residual} px</td>
                <td className="num">mag {t.mag}</td>
                <td className="num">{t.score_mean.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Most of these are chance alignments of junk that happen to line up; the next step (the track model) sorts them out.
        &quot;Off the line&quot; is how far the worst point is from a perfect straight line: real asteroids are within a fraction of a pixel.
        We never try every combination (about 2 x 10^19 on a night); a pair of blobs predicts where the others should be, and a fast look-up (KD-tree) checks.
      </p>
    </div>
  );
}

function Verdicts({ scene, night }: { scene: Scene; night: Night }) {
  const top = night.feature_importance.slice(0, 7);
  const max = top[0][1];
  const names: Record<string, string> = {
    n_detections: "number of photos it appears in", score_mean: "first model score (average)", stack_snr: "strength when photos are added up",
    forced_snr_min: "faintest measured point on the path", static_neighbours_mean: "stars sitting nearby", score_min: "first model score (lowest)",
    elongation_mean: "how stretched the blobs are", mag_std: "how steady the brightness is", max_residual: "distance off a straight line",
  };
  return (
    <div style={{ marginTop: 16 }}>
      <div className="t-scroll">
        <table className="t">
          <thead><tr><th>track</th><th>track score</th><th>decision (cut {night.tracklet_threshold.toFixed(2)})</th><th>catalogue check</th></tr></thead>
          <tbody>
            {[...scene.tracks].sort((a, b) => b.track_score - a.track_score).map((t) => (
              <tr key={t.id}>
                <td>{t.known_name || `track ${t.id}`}</td>
                <td className="num">{t.track_score.toFixed(3)}</td>
                <td>{t.accepted ? <b style={{ color: "var(--ember)" }}>kept</b> : <span style={{ color: "var(--text-3)" }}>dropped as a chance alignment</span>}</td>
                <td>{t.known_name ? <span style={{ color: "#8fb0f0" }}>known asteroid</span> : t.fake ? "our planted fake" : t.accepted ? "unknown → a human checks it" : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">Scores from the deployed track model (XGBoost, bundle v6). 1996 XG5 is bright and clean; 2006 DU133 is a fainter known asteroid, just above the cut.</p>
      <p className="kicker" style={{ marginTop: 22 }}>What the track model relies on most</p>
      <div className="bars" style={{ marginTop: 10 }}>
        {top.map(([f, v]) => (
          <div className="bar-row" key={f} data-tip={`${f}: importance ${v.toFixed(3)}`}>
            <span style={{ fontSize: 12.5, color: "var(--text-2)" }}>{names[f] ?? f}</span>
            <span className="track" style={{ borderColor: "var(--rule)" }}>
              <span className="fillbar" style={{ width: `${(100 * v) / max}%`, background: "#ff5a1f" }} />
            </span>
            <span className="v" style={{ fontSize: 12 }}>{v.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <p className="caption">
        Importance = how much each number helped the trees&apos; decisions (from the saved model). A track is kept if its score is at least {night.tracklet_threshold.toFixed(2)}.
        Tracks that match nothing known and are not our fakes go to a person with a written checklist. We never claim a discovery from a score alone.
      </p>
    </div>
  );
}

function Funnel({ night }: { night: Night }) {
  const m = night.metrics;
  const d = m.deployed;
  const rows: [string, number, string][] = [
    ["blobs found in 5 photos", night.blobs, `${pct(night.blobs_real / night.blobs)} are real (fakes + known asteroids): the base rate`],
    ["kept by the first model", night.kept_by_cnn, `recall ${pct(d.recall)} · precision ${pct(d.precision)} · ${fmt(Math.round(d.false_alarms_per_photo))} junk blobs per photo still kept`],
    ["joined into tracks", night.tracklets, "most are chance alignments of junk: 3 of 5 photos is a loose rule on purpose"],
    ["kept by the track model", night.accepted, `${night.accepted_fake} planted fakes · ${night.accepted_known} known asteroids · ${night.accepted_unexplained} unexplained, for a human to check`],
  ];
  const ops: [string, typeof d][] = [["the cut we use (stage 1)", d], ["cut for 95% recall", m.at_recall["0.95"]], ["cut for 90% recall", m.at_recall["0.90"]]];
  return (
    <div className="panel">
      <div className="panel-head"><span>Whole test night · {night.unit}</span><span>{night.fakes_planted} fakes planted</span></div>
      <div className="bars">
        {rows.map(([label, n, note]) => (
          <div key={label}>
            <div className="bar-row" style={{ gridTemplateColumns: "1fr 90px" }}>
              <span style={{ fontSize: 14 }}>{label}</span>
              <span className="v" style={{ fontSize: 15 }}>{fmt(n)}</span>
            </div>
            <div style={{ height: 14, background: "var(--raised)", marginTop: 6 }}>
              <div style={{ height: "100%", width: `${Math.max(0.6, (100 * n) / night.blobs)}%`, background: "#ff5a1f" }} />
            </div>
            <p className="caption" style={{ marginTop: 6 }}>{note}</p>
          </div>
        ))}
      </div>

      <p className="kicker" style={{ marginTop: 24 }}>The first model on this whole night: recall and precision at three cuts</p>
      <div className="t-scroll">
        <table className="t">
          <thead><tr><th>cut</th><th>recall</th><th>precision</th><th>F1</th><th>junk kept / photo</th></tr></thead>
          <tbody>
            {ops.map(([label, o]) => (
              <tr key={label}>
                <td>{label} <span className="mono" style={{ color: "var(--text-3)", fontSize: 11 }}>≥ {o.threshold < 0.01 ? o.threshold.toFixed(4) : o.threshold.toFixed(2)}</span></td>
                <td className="num">{pct(o.recall)}</td><td className="num">{pct(o.precision)}</td><td className="num">{o.f1.toFixed(2)}</td>
                <td className="num">{fmt(Math.round(o.false_alarms_per_photo))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Average precision over all cuts: <b>{m.average_precision.toFixed(3)}</b> (a model that guesses scores {pct(night.blobs_real / night.blobs)}, the base rate).
        Accuracy is not shown on purpose: a model that always says &quot;junk&quot; would be {pct(m.always_bogus_accuracy)} accurate and find nothing.
      </p>
      <div className="note" style={{ marginTop: 10 }}>
        <b>Why keep 99% of real blobs at only {pct(d.precision, 0)} precision?</b> A real asteroid dropped here is lost for good, but junk is cheap
        to remove later: it does not line up across photos, and the track model filters what does. So stage 1 is tuned for recall, and the whole
        pipeline is judged at the end by metrics that cannot be gamed by one cut: <b>depth</b> (half of the fakes found down to magnitude 21.3 on the test
        night, 22.1 with the deep search), <b>known asteroids recovered</b> (40 of 44) and <b>unexplained alarms per night</b>; see the Results page.
      </div>
    </div>
  );
}
