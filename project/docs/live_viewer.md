# Tool: the live viewer

## Goal
Watch the real pipeline work, one stage at a time, in a browser: the four files of an exposure, the noise, the blobs found, the CNN's verdicts, the tracks, the scores and the cross-match. It is for understanding, for debugging and for demonstrations. It adds no new method.

**Files built:**
- `viewer/live_state.py`: the shared record of a run (stages, numbers, pictures, a gate that waits for "Next" in step mode, cancel)
- `viewer/run_traced_night.py`: runs the point-source search on one night, calling the same functions in the same order as `build_night_detections` and `run_night_pipeline`, and reports each stage
- `viewer/draw_stage_figures.py`: one picture function per stage (matplotlib, PNG bytes)
- `viewer/list_viewer_units.py`: which nights on disk can be shown (all four products present, cached archive table, enough frames)
- `viewer/serve_live_viewer.py` and `viewer/live_viewer.html`: a small web server (Python standard library) and the page (no external libraries)
- `scripts/live_viewer.py`: starts it; `tests/viewer/`: 14 tests

```bash
python scripts/live_viewer.py        # opens http://127.0.0.1:8765
```

## How I thought about it
1. **Show the real thing.** A viewer that re-implements the pipeline would drift from it. The traced run calls the real functions; a test and a run on a real night show identical detections, scores, tracks and candidates (maximum difference 0.0).
2. **Stages, not logs.** Twelve fixed stages (pick the night, open files, noise, plant fakes, blobs, stamps, CNN, link, features, classifier, known?, result) are listed from the start, so you see the route and where you are.
3. **Step mode.** After each stage the run waits for "Next", so a presenter can explain each picture; "automatic" runs the whole night in about 20 seconds.
4. **Fakes make it vivid.** A real night usually has nothing unexplained. "fakes" plants fake asteroids first, so you can watch bright ones found and faint ones missed, which is the depth limit made visible.
5. **No network needed.** Only nights whose files and cached archive table are on disk are offered; known-asteroid positions come from the cache (if a lookup fails the run says so and carries on).

## Concepts
- **Polling:** the page asks `/api/state` every 0.8 s and redraws only when something changed; images are fetched by URL (`/img/<n>.png`) and kept in memory (the last 400).
- **A gate (threading.Event):** the worker thread blocks after a stage until the page posts `/api/next`; switching to automatic or pressing Stop releases it.
- **Same results as the pipeline:** `tests/viewer/test_run_traced_night.py` compares the traced run with `run_night_pipeline` on tiny synthetic frames (positions, CNN scores, tracklet scores), and a night without tracks must skip the later stages cleanly.

## Pitfalls
- The viewer runs the point search only (no deep or fast search). Those searches are shown by their own lessons.
- Held-out nights are listed with a label; watching them changes nothing, but do not use what you see to tune anything.
- Do not zip a folder that links to the data directory (it would try to include about 50 GB).

## Quiz
1. Why does the traced run call the real functions instead of copies?
2. What does step mode add for a presenter?
3. Why do the faint planted fakes fail to appear in the result table, and what is that called?
4. How does the page know when a stage has finished?
