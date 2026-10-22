"use client";

import { useEffect, useState } from "react";
import { json, pct, type Night, type ScoreRow } from "@/lib/data";

const OURS = "#ff5a1f";
const OTHER = "#3b6fd0";

function Bars({ rows, unit = "", max }: { rows: [string, number, string, string][]; unit?: string; max: number }) {
  return (
    <div className="bars">
      {rows.map(([label, v, color, tip]) => (
        <div className="bar-row" key={label} data-tip={tip}>
          <span>{label}</span>
          <span className="track">
            <span className="fillbar" style={{ width: `${(100 * v) / max}%`, background: color }} />
          </span>
          <span className="v">{v < 1 ? v.toFixed(3) : v}{unit}</span>
        </div>
      ))}
    </div>
  );
}

export default function Results() {
  const [night, setNight] = useState<Night | null>(null);
  const [split, setSplit] = useState<"test" | "validation" | "holdout">("test");
  useEffect(() => { json<Night>("/data/night.json").then(setNight); }, []);
  const board: ScoreRow[] = night ? night.scoreboard.filter((r) => r.split === split && r.version !== "v3") : [];

  return (
    <>
      <section className="scene paper" style={{ paddingTop: 130 }}>
        <div className="wrap">
          <span className="kicker">Results · evaluation metrics</span>
          <h1 className="h2" style={{ maxWidth: "24ch" }}>How well does it work? Measured on nights nobody tuned on.</h1>
          <p className="lede">
            Rule of the project: a number only counts if it comes from the deployed pipeline on held-out nights. Nights were split before
            any modelling: 8 for training, 2 for validation, 1 test night, 1 holdout night.
          </p>
          <div className="facts">
            {[
              ["0.902", "Average precision of our XGBoost model on validation nights (old 6-number baseline: 0.783).", "blob model · validation"],
              ["0.938", "Average precision on the untouched test night.", "blob model · test"],
              ["158", "False alarms per photo at 90% recall, vs 806 for the old baseline.", "5.1x fewer"],
              ["22.1", "Magnitude where half the planted fakes are found, with the deep search (single photos: 21.3).", "depth · test night"],
            ].map(([n, p, s], i) => (
              <div className="fact" key={n}>
                {i === 0 && <span className="cross tl" />}
                {i === 3 && <span className="cross br" />}
                <span className="n">{n}</span>
                <p>{p}</p>
                <small>{s}</small>
              </div>
            ))}
          </div>

          <div className="grid-2" style={{ marginTop: 48 }}>
            <div className="panel">
              <div className="panel-head"><span>Our model vs the old baseline · validation nights</span><span>67,041 blobs, 3.2% real</span></div>
              <p className="kicker">Average precision (higher is better)</p>
              <Bars max={1} rows={[["XGBoost + PCA (ours)", 0.902, OURS, "average precision 0.902"], ["Logistic regression, 6 numbers", 0.783, OTHER, "baseline average precision 0.783"]]} />
              <p className="kicker" style={{ marginTop: 22 }}>False alarms per photo at 90% recall (lower is better)</p>
              <Bars max={806} rows={[["XGBoost + PCA (ours)", 158, OURS, "158 junk blobs kept per photo"], ["Logistic regression, 6 numbers", 806, OTHER, "806 junk blobs kept per photo"]]} />
              <p className="caption">Both models must find 90% of the real blobs; then we count how much junk each lets through. Comparing at the same recall is the fair way.</p>
            </div>
            <div className="panel">
              <div className="panel-head"><span>Depth: how faint can we go?</span><span>test night 2019-02-08</span></div>
              <p className="kicker">Magnitude at 50% completeness (bigger = fainter = better)</p>
              <div className="bars">
                {[["v1: first pipeline", 20.5, OTHER], ["v6: classical ML (deployed)", 21.3, OTHER], ["Deep search (adds photos)", 22.1, OURS]].map(([l, v, c]) => (
                  <div className="bar-row" key={l as string} data-tip={`${l}: magnitude ${v}`}>
                    <span>{l}</span>
                    <span className="track"><span className="fillbar" style={{ width: `${((Number(v) - 19.5) / 3) * 100}%`, background: c as string }} /></span>
                    <span className="v">{v}</span>
                  </div>
                ))}
              </div>
              <p className="caption">
                Axis starts at magnitude 19.5. Each step of 1 magnitude is 2.5x fainter, so +0.7 from the deep search is about 1.9x fainter objects.
                Half of planted fakes at that brightness are found.
              </p>
            </div>
          </div>

          {night && <TestNight night={night} />}

          <div className="grid-2" style={{ marginTop: 24 }}>
            <div className="panel">
              <div className="panel-head"><span>Confusion matrix · simple model at cut 0.5</span><span>validation</span></div>
              <div className="cm">
                <div className="h" />
                <div className="h">predicted real</div>
                <div className="h">predicted junk</div>
                <div className="h">really real</div>
                <div className="ok"><b>1,547</b>true positives</div>
                <div className="bad"><b>582</b>missed (false negatives)</div>
                <div className="h">really junk</div>
                <div className="bad"><b>493</b>false alarms</div>
                <div className="ok"><b>64,419</b>true negatives</div>
              </div>
              <p className="caption">
                Accuracy here is 98.4%, but a model that always says &quot;junk&quot; scores 96.8% while finding nothing. That is why we
                use precision, recall and average precision instead of accuracy.
              </p>
            </div>
            <div className="panel">
              <div className="panel-head"><span>Track classifier · 4 models compared</span><span>saturated · not used to judge</span></div>
              <Bars max={1} rows={[
                ["Logistic regression", 0.999, OTHER, "average precision 0.999"],
                ["Decision tree", 0.983, OTHER, "average precision 0.983"],
                ["Random forest", 0.999, OTHER, "average precision 0.999"],
                ["XGBoost (used)", 0.999, OURS, "average precision 0.999"],
              ]} />
              <p className="caption">
                These 0.999 values are <b>not</b> a sign of a near-perfect system. 96% of tracks are our planted fakes, which are easy, so every model
                scores the same and the metric cannot show progress (it is &quot;saturated&quot;). Only 38 junk tracks existed in training, too few for a
                reliable precision. We therefore judge versions by depth, known asteroids recovered and unexplained alarms per night (table below).
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="scene dark">
        <div className="wrap">
          <span className="kicker">The scoreboard · project/reports/scoreboard.csv</span>
          <h2 className="h2">Every version, measured the same way</h2>
          <div className="seg" role="group" aria-label="Which nights" style={{ marginTop: 10 }}>
            {(["test", "validation", "holdout"] as const).map((s) => (
              <button key={s} className={split === s ? "on" : ""} onClick={() => setSplit(s)}>{s} nights</button>
            ))}
          </div>
          <div className="t-scroll" style={{ marginTop: 18 }}>
            <table className="t">
              <thead>
                <tr><th>version</th><th>depth (mag 50%)</th><th>known found</th><th>unexplained / night</th><th>blob AP</th><th>stage-1 recall</th><th>junk kept / photo</th><th>track AP</th></tr>
              </thead>
              <tbody>
                {board.map((r) => (
                  <tr key={r.version} style={r.version === "v6" ? { background: "rgba(255,90,31,.10)" } : undefined}>
                    <td>{r.version === "v6" ? <b>v6 · classical ML (deployed)</b> : r.version === "v4" ? "v4 · with CNN (comparison)" : r.version}</td>
                    <td className="num"><b>{r.mag_at_50_completeness.toFixed(2)}</b></td>
                    <td className="num"><b>{r.known_recovered}</b></td>
                    <td className="num"><b>{r.unexplained_per_night.toFixed(1)}</b></td>
                    <td className="num">{r.detection_ap.toFixed(3)}</td>
                    <td className="num">{(100 * r.detection_recall).toFixed(1)}%</td>
                    <td className="num">{Math.round(r.bogus_kept_per_frame).toLocaleString("en-US")}</td>
                    <td className="num" style={{ color: "var(--text-3)" }}>{r.tracklet_ap.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">
            The first three columns are how we judge a version: they reflect the whole pipeline and have room to improve. Stage-1 recall is about 99% by
            design and must always be read with the junk it lets through (next column); track AP is saturated (grey).
            Planted-fake runs over {board[0]?.nights ?? "…"} quadrant-nights ({board[0]?.frames ?? "…"} photos). v1: first system · v2: relaxed linking (finds more,
            but more chance tracks) · v4: v2 + 2 extra feature groups (17 features), with the CNN · v5: v4 trained on 3x more data, which did not help at
            equal false alarms · <b>v6: the v4 design with classical ML only</b> (XGBoost on 29 features + PCA instead of the CNN). v6 ties v4: the same known
            asteroids on every split, depth within 0.07 mag, and fewer junk blobs kept per photo. (v3, run on one quadrant only, is left out.)
          </p>

          <div className="grid-3" style={{ marginTop: 40 }}>
            {[
              ["What worked", "Finding where asteroids were lost (at linking, not at the classifier) and fixing it; correcting for the telescope's pointing shifts; a better catalogue (JPL Horizons, 2.72 px → 0.26 px error); adding photos together (deep search, +0.7 mag)."],
              ["What did not", "Deep learning did not beat classical ML: our CNN and a bigger multi-frame CNN both tied XGBoost, so the deployed system (v6) uses classical ML only. Three times more training data (v5) did not help, because one 30-second photo only holds so much light."],
              ["Statistics, not just code", "The deep search tries 4,888 motions, so pure noise reaches 5 sigma by chance. We require 6 sigma, and a test with shuffled times gave 0 false tracks (down from 16), while all 20 known asteroids were kept."],
            ].map(([t, d]) => (
              <div className="panel" key={t}>
                <h3 style={{ fontSize: 20, marginBottom: 10 }}>{t}</h3>
                <p style={{ color: "var(--text-2)", fontSize: 14.5, margin: 0 }}>{d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

function TestNight({ night }: { night: Night }) {
  const m = night.metrics;
  const d = m.deployed;
  const rows: [string, typeof d][] = [["Stage-1 cut (deployed)", d], ["Cut for 95% recall", m.at_recall["0.95"]], ["Cut for 90% recall", m.at_recall["0.90"]]];
  return (
    <div className="grid-2" style={{ marginTop: 24 }}>
      <div className="panel">
        <div className="panel-head"><span>Our model (XGBoost + PCA) · whole test night, one quadrant</span><span>{night.blobs.toLocaleString("en-US")} blobs · {pct(night.blobs_real / night.blobs)} real</span></div>
        <div className="t-scroll">
          <table className="t">
            <thead><tr><th>cut</th><th>recall</th><th>precision</th><th>F1</th><th>false alarms / photo</th></tr></thead>
            <tbody>
              {rows.map(([label, o]) => (
                <tr key={label}>
                  <td>{label}</td><td className="num">{pct(o.recall)}</td><td className="num">{pct(o.precision)}</td>
                  <td className="num">{o.f1.toFixed(2)}</td><td className="num">{Math.round(o.false_alarms_per_photo).toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          Average precision {m.average_precision.toFixed(3)} (random guessing: {(night.blobs_real / night.blobs).toFixed(3)}). Recall and precision trade
          against each other: the deployed cut keeps {pct(d.recall)} of real blobs but only {pct(d.precision)} of what it keeps is real; demanding
          90% recall instead gives {pct(m.at_recall["0.90"].precision)} precision. We deploy the high-recall cut on purpose because linking and the track
          model remove the junk later, while a dropped asteroid is lost for good. For comparison, the deep-learning CNN scores AP{" "}
          {night.cnn_metrics.average_precision.toFixed(3)} on the same night, and {pct(night.cnn_metrics.at_recall["0.90"].precision)} precision at 90% recall.
        </p>
      </div>
      <div className="panel">
        <div className="panel-head"><span>Confusion matrix · our model at the deployed cut</span><span>test night</span></div>
        <div className="cm">
          <div className="h" />
          <div className="h">kept (predicted real)</div>
          <div className="h">dropped (predicted junk)</div>
          <div className="h">really real</div>
          <div className="ok"><b>{d.tp.toLocaleString("en-US")}</b>true positives</div>
          <div className="bad"><b>{d.fn.toLocaleString("en-US")}</b>missed</div>
          <div className="h">really junk</div>
          <div className="bad"><b>{d.fp.toLocaleString("en-US")}</b>false alarms</div>
          <div className="ok"><b>{d.tn.toLocaleString("en-US")}</b>true negatives</div>
        </div>
        <p className="caption">
          Accuracy here would be {pct(d.accuracy)}, and a model that always says &quot;junk&quot; would get {pct(m.always_bogus_accuracy)}: accuracy says
          nothing useful when only {pct(night.blobs_real / night.blobs)} of blobs are real. ROC AUC ({m.roc_auc.toFixed(3)}) also flatters rare-class
          problems, so we report precision, recall, F1 and average precision.
        </p>
      </div>
    </div>
  );
}
