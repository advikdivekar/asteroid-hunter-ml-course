# Asteroid Hunter: web app

Shows the project working on real data, step by step, with the deployed **classical ML model** (bundle v6: XGBoost on 29 measured
features + 20 PCA numbers) running in the browser. The deep-learning CNN is included only as a comparison.

```bash
npm install
npm run dev            # http://localhost:3000
npm run build          # static site in out/ (open with any web server, e.g. npx serve out)
npm run check-models   # checks the browser models against Python (Node 22.18+)
```

| Page | What it shows |
|---|---|
| `/` Overview | the problem, headline numbers, the pipeline in six steps, what we do not claim |
| `/pipeline/` How it works | 8 steps on a real 160 x 160 window of the test night (2019-02-08): photos, difference, blobs, stamps, the model scoring all 118 blobs live (with a threshold slider), linking, the track model's verdicts, whole-night funnel and metrics. Deep links: `?step=5&run=1` |
| `/model/` The model | the 49 inputs, the training recipe; pick any of 48 real blobs and see what the model measured, the questions tree 1 asked and p(real), computed live; the five-model comparison (CNN as a reference); k-means clusters of the junk |
| `/results/` Results | evaluation metrics: our model vs the baseline, depth, recall/precision/F1 at three cuts, confusion matrices, the scoreboard (v6 highlighted, v4 = CNN) |
| `/team/` Team | who did what, step by step, with each person's process diagram |

Data in `public/data/` is exported from the full project by `../tools/export_web_data.py` (needs the raw data and the `asthunt`
environment): the v6 trees and PCA (`trees.json`), the v4 CNN weights (comparison), real stamps, the sky window as PNGs, the
whole-night metrics and the scoreboard. `lib/trees.ts` re-implements `project/src/asthunt/features/compute_stamp_features.py` and walks
the exported trees; `lib/cnn.ts` is a port of the CNN. Python computes the features in float32 and the browser in float64, so a blob
sitting exactly on a split can move by up to 0.01, but no keep/drop decision changes (`scripts/check_models.mjs`).
Styling follows the certifyai/temper design language (steel and paper scenes, one ember accent).
