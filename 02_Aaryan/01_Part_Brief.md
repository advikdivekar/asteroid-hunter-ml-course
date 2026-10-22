# Aaryan: problem, domain, data and preprocessing

> **Update: classical ML only (bundle v6).** Paras's stage-1 model is now XGBoost on 29 measured features + PCA (no neural network).
> Your part is unchanged; the blob table you hand over now also feeds those features (`snr`, `peak`, `npix`, `a`, `b`, `flux` are 6 of the 29).

**Marks this part carries:** Problem Understanding and Formulation (3), Data Preprocessing and Exploration (3, shared with Paras for the stamp-level preparation).
**Why it matters:** if the data is wrong or leaks, every model and metric downstream is wrong. This is where the project learned its most expensive lessons (leakage, label and coordinate errors started here).

## Your four jobs
1. **Problem formulation:** what we are solving, why it is hard, why difference images, why ML, how success is measured.
2. **The data:** where it comes from (ZTF via IRSA), what the four files are, how it was chosen and downloaded (metadata first, resumable, atomic), how nights were split, and why.
3. **Preprocessing and detection:** loading an exposure, bad-pixel masks, reference = science minus difference, background and noise, sky coordinates, and finding blobs in difference images (SEP, 2-sigma, minimum 5 pixels).
4. **Exploration and the leakage audit:** the threshold study, "most detections are noise", the stamp zoo, and the four kinds of leakage, with the experiment (100% -> 50% -> 99%) that proved it.

## What is NOT yours
Labels, stamps and the CNN: Paras. Linking, trees, metrics: Satvik. Scoreboard, deep search, viewer: Advik. You hand Paras a clean table of blobs and a clean split.

## Where your code is (in `project/`)
| Area | Files | What they do |
|---|---|---|
| Acquire | `src/asthunt/acquire/query_ztf_metadata.py`, `query_ztf_field_night.py`, `select_night_exposures.py`, `load_unit_metadata.py`, `list_units.py`, `download_ztf_images.py`, `download_ztf_reference_image.py`, `find_quadrants_covering_position.py` | ask the archive; pick exposures; list units; download safely |
| Clean | `src/asthunt/clean/load_exposure.py`, `read_fits_image.py`, `read_exposure_header.py`, `build_bad_pixel_mask.py`, `estimate_background.py`, `load_night_images.py` | read the four files, build the mask, estimate background and noise |
| Detect | `src/asthunt/detect/extract_sources.py` | SEP matched-filter detection of positive blobs |
| Config | `src/asthunt/config/load_config.py`, `configs/acquire.yaml`, `detect.yaml`, `project.yaml` | all settings, including the **night split** |
| Leakage | `src/asthunt/evaluate/check_split_leakage.py` | lists groups that appear in two splits (must be empty) |
| Tests | `tests/acquire`, `tests/clean`, `tests/detect`, `tests/config` | all offline |
| Lessons and notebooks | `docs/project_setup.md`, `astronomy_data_primer.md`, `data_acquisition.md`, `difference_imaging.md`, `leakage_audit.md`, `kepler_post_check.md`; notebooks with the same names | |

## What you hand over
- The problem statement and related-work paragraph for the report (with the papers in `00_Start_Here/04_Course_Mapping.md`).
- Figures: the four-file picture, the noise histogram, the threshold table, the split diagram, the leakage experiment.
- A clean description of the data for the report (sources, size, split, preprocessing steps, what was masked and why).

## Your seven key messages
1. **Difference images remove everything that did not change**, so only new or moving things remain.
2. **Ask before you download:** a metadata query (seconds) told us 1,552 exposures exist, 16.4% carry quality flags, and only 40 nights have at least 4 good frames.
3. **The split is by night, decided in a config file before modelling**, so it cannot change after disappointing results.
4. **Detection is tuned for completeness; classification for purity.** About 7,800 blobs per frame at 2 sigma, only 173 with SNR > 5, most are noise, and that is fine.
5. **The first dataset cheated** (four kinds of leakage); we proved it with an experiment and rebuilt the data.
6. **The test night is deliberately different** (sharper, deeper, near opposition): a distribution shift we want, and must read with care.
7. **Provenance:** you can say exactly which files and settings produced any result.

## Numbers you must know by heart
| Number | Meaning | Source |
|---|---|---|
| 1,552 | exposures of our quadrant in the archive table | `docs/data_acquisition.md` |
| 898 / 478 / 176 | r / g / i band exposures | same |
| 16.4% | exposures with quality flags (we keep `infobits == 0`) | same |
| 8 / 2 / 1 / 1 | train / validation / test / holdout nights | `configs/acquire.yaml` |
| 220 files, 3.55 GB | first 11 nights x 5 frames + 1 reference | `docs/data_acquisition.md` |
| 38 / 19 / 7 / 0.006 MB | science / mask / difference / PSF per exposure | same |
| 3072 x 3080 px, 1.01 arcsec/px, 30 s | one quadrant, pixel scale, exposure | domain primer |
| 1,950 -> 334 | sources in science vs difference image at 5 sigma | `docs/difference_imaging.md` |
| 7,764 blobs at 2 sigma; 173 with SNR > 5 | detection counts on one frame | same |
| 6.1k / 7.8k / 731 / 334 | detections at 1.5 / 2 / 3 / 5 sigma | same |
| 100% -> 50% -> 99% | leakage experiment | `docs/leakage_audit.md` |
| 222.7 px vs 2.6 px | smallest random-bogus residual vs largest real residual | `docs/ml_basics.md` |

## Hands-on tasks (do these yourself)
1. `cd project && pytest tests/acquire tests/clean tests/detect tests/config -q`.
2. Re-run `notebooks/difference_imaging.ipynb`; produce the threshold table yourself; explain in two sentences why 1.5 sigma gives *fewer* detections than 2 sigma.
3. Re-run `notebooks/leakage_audit.ipynb`; reproduce 100% -> 50% -> 99%.
4. Open one mask file's header (`BIT00...BIT15`) and list which bits we treat as bad and why the ghost bit (5) is not.
5. Change `threshold_sigma` in a copy of `configs/detect.yaml` to 3.0; predict then check how the number of blobs changes.

## Pitfalls
- Saying "we downloaded the data" without saying *how it was chosen* and *how it was split*.
- Confusing the **reference image** (old, deep) with the **science image** (tonight).
- Forgetting that `reference = science - difference` is computed (ZTF already aligned and blurred it).
- Mixing 0-based (numpy/SEP) and 1-based (FITS) pixel coordinates.
- Treating NaN as 0 without masking: its edge becomes a fake "detection".
