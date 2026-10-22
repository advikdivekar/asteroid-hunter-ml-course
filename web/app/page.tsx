import Link from "next/link";
import SkyLoop from "@/components/SkyLoop";

const TICKER = [
  "41,522 blobs on one night",
  "about 3% are real",
  "classical ML: XGBoost on 49 measured numbers",
  "average precision 0.938 on the untouched test night",
  "5.1x fewer false alarms than the baseline",
  "depth 20.5 -> 21.4 -> 22.1 magnitude",
  "31 known asteroids recovered",
  "600 decision trees, trained in seconds",
  "166 offline tests",
  "no discovery claimed",
];

const STAGES = [
  ["01", "Photos", "ZTF photographs the same patch of sky every few minutes. We take 5 photos of one night."],
  ["02", "Difference", "Tonight minus an old deep photo: stars cancel, only what changed is left."],
  ["03", "Blobs", "Every bright spot is circled. Thousands per photo, almost all junk."],
  ["04", "ML model", "The computer measures 49 numbers per blob and 600 decision trees (XGBoost) say how likely it is to be real."],
  ["05", "Tracks", "Real asteroids move in straight lines: we join blobs that line up across the photos."],
  ["06", "Known?", "A second model scores each track, then we check the asteroid catalogue."],
];

export default function Home() {
  return (
    <>
      <section className="scene" style={{ paddingTop: 0, paddingBottom: 0 }}>
        <div className="wrap hero">
          <div className="hero-copy">
            <span className="kicker">Machine Learning course project · MPSTME NMIMS · 2026-27</span>
            <h1 className="h1">
              Asteroid
              <br />
              <span className="ember">Hunter</span>
            </h1>
            <p className="lede">
              We built a machine-learning pipeline that looks through public telescope images from the Zwicky Transient Facility for
              tiny moving dots, which are asteroids, and we measured honestly how well it works.
            </p>
            <div className="actions">
              <Link href="/pipeline/" className="cta fire">
                <span>Watch it work, step by step</span>
                <b>→</b>
              </Link>
              <Link href="/model/" className="cta">
                <span>Run the model in your browser</span>
                <b>→</b>
              </Link>
            </div>
          </div>
          <SkyLoop />
        </div>
      </section>

      <div className="ticker" aria-hidden="true">
        <div className="ticker-track">
          {[0, 1].map((k) => (
            <div className="ticker-row" key={k}>
              {TICKER.map((t) => (
                <span key={t}>{t}</span>
              ))}
            </div>
          ))}
        </div>
      </div>

      <section className="scene paper">
        <div className="wrap">
          <span className="kicker">The problem in one paragraph</span>
          <h2 className="h2">Stars stay put. Asteroids move. The hard part is everything else.</h2>
          <p className="lede">
            When you subtract an old photo of the sky from tonight&apos;s, the stars disappear and only things that changed remain. But
            most of what remains is junk: noise, leftovers of bright stars, cosmic-ray hits. On our test night one patch of sky gave
            41,522 blobs and only about 3% were real. Writing one rule by hand for every kind of junk does not work, so we measure 49 numbers
            per blob and a machine-learning model (XGBoost decision trees) learns from labelled examples which combinations mean real.
          </p>
          <div className="facts">
            {[
              ["0.938", "Average precision of our XGBoost model on a night it never saw (old baseline: 0.508 there).", "ML model · test night"],
              ["5.1x", "Fewer false alarms per photo than the old baseline, at the same 90% of real objects found (158 vs 806).", "ML model vs baseline"],
              ["22.1", "Faintest magnitude where half the planted fakes are found, using our deep search. Bigger = fainter.", "Depth · deep search"],
              ["31", "Known asteroids found by the pipeline on its own on the test night, brightness 16.2 to 21.8.", "Real search"],
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
        </div>
      </section>

      <section className="scene dark">
        <div className="wrap">
          <span className="kicker">The pipeline</span>
          <h2 className="h2">Six steps from a photo to a list of asteroids</h2>
          <div className="grid-3" style={{ marginTop: 36 }}>
            {STAGES.map(([n, t, d]) => (
              <div className="panel" key={n}>
                <div className="panel-head">
                  <span>Step {n}</span>
                  {t === "ML model" && <span className="ember">classical ML</span>}
                </div>
                <h3 style={{ fontSize: 24 }}>{t}</h3>
                <p style={{ color: "var(--text-2)", fontSize: 14.5, marginBottom: 0 }}>{d}</p>
              </div>
            ))}
          </div>
          <div className="actions" style={{ marginTop: 36 }}>
            <Link href="/pipeline/" className="cta fire">
              <span>See every step on real images</span>
              <b>→</b>
            </Link>
            <Link href="/results/" className="cta">
              <span>See the results</span>
              <b>→</b>
            </Link>
          </div>
        </div>
      </section>

      <section className="scene paper">
        <div className="wrap">
          <span className="kicker">Honesty first</span>
          <h2 className="h2">What we do not claim</h2>
          <div className="grid-3" style={{ marginTop: 24 }}>
            <p className="note">
              <b>No discovery.</b> One faint object in a related study passes every check we have, but it still needs a human and the
              Minor Planet Center. Nothing has been reported.
            </p>
            <p className="note">
              <b>One night, one patch.</b> The best depth number (22.1) is for one night and one CCD quadrant.
            </p>
            <p className="note">
              <b>Labels are partly simulated.</b> Our &quot;real&quot; examples are planted fakes and known asteroids. A test showed
              fakes look like real asteroids (AUC 0.47), but that is a check, not a proof.
            </p>
          </div>
        </div>
      </section>
    </>
  );
}
