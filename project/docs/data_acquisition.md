# Phase: Data acquisition

## Goal
Go from "what does the archive have?" to a reproducible set of files on disk, **already split by night**, without ever editing the raw data.

**Files built:**
- `src/asthunt/acquire/query_ztf_metadata.py`: one web query → a table of every exposure of our field/CCD/quadrant, cached to CSV
- `src/asthunt/acquire/select_night_exposures.py`: the first N good exposures of one night, in one filter
- `src/asthunt/acquire/download_ztf_images.py`: downloads image products; skips existing files; writes `*.part` then renames; retries when the server is busy
- `src/asthunt/acquire/download_ztf_reference_image.py`: the deep reference image
- `configs/acquire.yaml`: the field, filter, products, **the night split**, and the mask bits counted as bad
- tests (all offline), `notebooks/data_acquisition.ipynb`

---

## How I thought about it
1. **Ask before downloading.** The archive answers a metadata query in seconds: 1,552 exposures of our quadrant. Choosing nights from that table costs nothing; downloading the wrong 3.5 GB costs an hour.
2. **What does linking need?** Several exposures of the same field on the **same night**. Only 40 nights have ≥ 4 good r-band frames.
3. **Pick the test night first and lock it away.** **2019-02-08** was the best night: 55 exposures, 1.6 px seeing, depth 21.1 mag, and the field near **opposition** (opposite the Sun, where asteroids are brightest and most numerous). That makes it the most promising night for a discovery, so it's the one we must not tune on.
4. **Write the split into a config file before any modelling.** Then it can't quietly change when results disappoint.
5. **Expect the network to fail.** Big public archives are busy. The downloader is safe to re-run, writes complete files only, and retries. The archive did return `503 Service Unavailable` twice during our download, and the retries handled both.
6. **Read the mask's own header.** Its `BIT00…BIT15` keywords define each flag, which settled which bits count as bad.

---

## Concepts

### Metadata first
Metadata = data about the data: time, filter, seeing, depth, quality flags. Exploring it lets you choose data on purpose. Our table showed:
- 898 r-band, 478 g-band, 176 i-band exposures,
- 16.4% of exposures carry quality flags (`infobits ≠ 0`). The difference-image header lists what each flag means (e.g. "SEEING > 5\"", "PHOTOMETRIC CALIBRATION NOT POSSIBLE"). We keep only `infobits == 0`.

### Splitting by night, decided up front
| split | nights | median seeing | median depth |
|---|---|---|---|
| train | 8 | 1.86 px | 20.32 |
| validation | 2 | 2.08 px | 19.85 |
| test | 1 | 1.60 px | 21.08 |

The test night is *better* than the rest: sharper and deeper. That's a deliberate **distribution shift**, because we want to search the best data. Expect the test night to contain fainter sources than the model has seen much of, and read its scores with that in mind.

### Caching and provenance
`metadata__000618.csv` is the exact archive answer we used. Re-runs read the file, not the internet, so they're fast, work offline, and are repeatable even if the archive changes. Knowing exactly which data produced a result is called **provenance**.

### Idempotent, atomic downloads
- **Idempotent:** running it twice gives the same result as running it once. Existing files are skipped.
- **Atomic:** a file appears under its real name only once it's complete (`.part` → rename). A crash can't leave a half file that looks finished.
- **Retry with growing waits:** 20 s, 40 s, 60 s… Busy servers recover; hammering them doesn't help.

### What we downloaded (per exposure)
| product | size | purpose |
|---|---|---|
| `sciimg.fits` | 38 MB | the science image |
| `mskimg.fits` | 19 MB | 16 yes/no flags per pixel |
| `scimrefdiffimg.fits.fz` | 7 MB | science − reference, compressed |
| `diffimgpsf.fits` | 6 KB | PSF model of the difference image |

Total: 11 nights × 5 frames + 1 reference = **220 files, 3.55 GB**, 0 missing.

### Bad mask bits (from the mask header)
Masked: 0 satellite track, 2–3 bad responsivity, 4 noisy, 7 pixel spike, 8 saturated, 9 dead, 10 NaN, 12 halo.
Not masked: 1 and 11 ("a source is here", which is not a defect), and 5 "ghost". Ghosts cover ~4% of the chip, so masking them would hide real asteroids that cross them. The model will learn what ghosts look like instead.

---

## Steps, in order
1. Queried the IRSA metadata service for field 618, CCD 3, quadrant 4 → 1,552 rows.
2. Counted good exposures per night and filter, and listed linkable nights.
3. Chose the test night (2019-02-08), then 2 validation and 8 training nights from the 2020–2023 September nights; wrote them into `configs/acquire.yaml`.
4. Checked product URLs and sizes with HEAD requests.
5. Downloaded one mask and read its `BITnn` header → chose `bad_mask_bits`.
6. Wrote the four acquire files + offline tests.
7. Ran the download in the background; it hit `503` twice and recovered by retrying (I added the retry after the first failure).
8. Notebook: explored metadata, showed the split, verified 220 files / 3.55 GB / no `.part` leftovers, and looked at a science / difference / mask cut-out.

## Commands
```bash
pytest tests/acquire
jupyter lab notebooks/data_acquisition.ipynb   # downloads anything missing, then verifies
```

## Pitfalls
- **Choosing nights after seeing results** turns the test set into a training set.
- **Downloading without a manifest:** later you can't say which files a model saw.
- **Assuming the network works.** Plan for retries and partial files.
- **Mixing filters.** Each filter has its own reference image and difference images.

## Quiz
1. Why do we query metadata before downloading anything?
2. Why is it important that the split lives in a config file written before training?
3. What two properties make our downloader safe to re-run after a crash?
4. Why wasn't the ghost bit (5) treated as bad?
5. The test night is sharper and deeper than training. Is that a problem? What should we expect?
