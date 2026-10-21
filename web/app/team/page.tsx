"use client";

import { useState } from "react";

const TEAM = [
  {
    name: "Aaryan", role: "Problem, data and preprocessing",
    steps: ["Define the problem and how success is measured", "Ask the archive first: 1,552 exposures, 16.4% flagged", "Split by night before modelling: 8 / 2 / 1 / 1", "Download 4 files per exposure: 220 files, 3.55 GB", "Clean: bad-pixel mask, reference, noise level", "Find blobs at 2 sigma: about 7,800 per photo", "Leakage audit: 100% → 50% → 99%, data rebuilt"],
    out: "A clean table of blobs and a leak-free split by night",
  },
  {
    name: "Paras", role: "Labels, dataset and the CNN",
    steps: ["Plant 300 fake asteroids per night", "Label every blob: fake / known asteroid = real", "Cut 3 x 31 x 31 stamps", "Normalise, augment, balance: 53,178 stamps", "Build and train the CNN: 72,145 weights", "Measure: AP 0.899 vs 0.780; 176 vs 813 false alarms", "Learn from mistakes: catalogue was 2.9″ off", "Cross-fit the scores for the next model"],
    out: "Every blob with a CNN score, and the cut",
  },
  {
    name: "Satvik", role: "Linking, tracklet classifier and metrics",
    steps: ["Keep the blobs the CNN likes", "Put all photos on one shared sky grid", "Link blobs into straight-line tracks (KD-tree)", "17 numbers per track", "Compare 4 models, cross-validated by night", "Pick the cut on validation nights only", "Use the right metrics: precision, recall, AP", "Check the catalogue; checklist for a human"],
    out: "Scored tracks, each marked known or unknown",
  },
  {
    name: "Advik", role: "System, evaluation, deep search, live viewer (lead)",
    steps: ["Design the system and the hand-overs", "Set the test rules: held-out nights, planted fakes", "Build the scoreboard: v1 → v5", "Find where asteroids are lost; fix linking", "Test bigger ideas fairly (they tied or did not help)", "Deep search: 4,888 motions, 6 sigma → 22.1 mag", "Live viewer and demo", "Reproducible: model bundles, configs, 166 tests"],
    out: "Depth 20.5 → 21.4 → 22.1; 31 known asteroids recovered",
  },
];

export default function Team() {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <section className="scene paper" style={{ paddingTop: 130 }}>
      <div className="wrap">
        <span className="kicker">Team · who did what, step by step</span>
        <h1 className="h2">Four people, one pipeline</h1>
        <p className="lede">
          Each person owns one stage and hands a clear result to the next. Data flows left to right: Aaryan → Paras → Satvik → Advik.
          Open a process diagram to see each part as a picture.
        </p>
        <div className="grid-2" style={{ marginTop: 36 }}>
          {TEAM.map((p, i) => (
            <article className="team-card" key={p.name}>
              <span className="kicker">Stage {i + 1} of 4</span>
              <h2 style={{ fontSize: 30 }}>{p.name}</h2>
              <p style={{ margin: 0, color: "var(--ink-2)" }}>{p.role}</p>
              <ol>
                {p.steps.map((s) => <li key={s}><b>{s}</b></li>)}
              </ol>
              <div className="out"><span>OUTPUT</span>{p.out}</div>
              <button className="cta" onClick={() => setOpen(open === p.name ? null : p.name)} aria-expanded={open === p.name}>
                <span>{open === p.name ? "Hide" : "Show"} process diagram</span><b>{open === p.name ? "↑" : "↓"}</b>
              </button>
              {open === p.name && (
                // eslint-disable-next-line @next/next/no-img-element
                <img className="diagram" src={`/diagrams/process_${p.name}.svg`} alt={`${p.name}'s process diagram`} />
              )}
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
