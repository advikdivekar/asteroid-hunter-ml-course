# Asteroid Hunter: web app

Shows the project working on real data, step by step, with the trained CNN running in the browser.

```bash
npm install
npm run dev          # http://localhost:3000
npm run build        # static site in out/ (open with any web server, e.g. npx serve out)
npm run check-cnn    # checks the browser CNN gives the same scores as PyTorch (Node 22.18+)
```

| Page | What it shows |
|---|---|
| `/` Overview | the problem, headline numbers, the pipeline in six steps, what we do not claim |
| `/pipeline/` How it works | 8 steps on a real 160 x 160 window of the test night (2019-02-08): photos, difference, blobs, stamps, the CNN scoring all 118 blobs live (with a threshold slider), linking, track scoring, whole-night funnel. Deep links: `?step=5&run=1` |
| `/model/` The model | the one ML architecture (CNN), its layers and training recipe; pick any of 48 real blobs and see its feature maps and p(real), computed live |
| `/results/` Results | evaluation metrics: CNN vs baseline, depth, confusion matrix, model comparison, the scoreboard by split |
| `/team/` Team | who did what, step by step, with each person's process diagram |

Data in `public/data/` is exported from the full project by `../tools/export_web_data.py` (needs the raw data and the `asthunt` environment):
the v4 CNN weights, real stamps, the sky window as PNGs, and `scoreboard.csv`. The browser CNN is `lib/cnn.ts`, a line-by-line port of
`project/src/asthunt/models/define_realbogus_cnn.py`. Styling follows the certifyai/temper design language (steel and paper scenes, one ember accent).
