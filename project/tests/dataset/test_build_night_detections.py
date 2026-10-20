"""End-to-end test of one night on tiny fake images (no network needed)."""

import numpy as np
import pandas as pd
from astropy.io import fits

from asthunt.dataset.assemble_stamp_dataset import assemble_stamp_dataset
from asthunt.dataset.build_night_detections import build_night_detections

NIGHT = "2020-01-01"
UNIT = {"field": 618, "ccd": 3, "quadrant": 4, "night": NIGHT}
ACQUIRE = {"field": 618, "ccd": 3, "quadrant": 4, "filter": "zr", "frames_per_night": 2, "bad_mask_bits": [0]}
INJECT = {"asteroids_per_night": 3, "magnitude_range": [16.0, 16.5], "rate_range_arcsec_per_hour": [30.0, 30.0],
          "edge_margin_pixels": 30, "match_radius_pixels": 2.0, "known_match_radius_pixels": 4.0,
          "known_asteroid_max_v_mag": 20.5}
DETECT = {"threshold_sigma": 3.0, "min_area_pixels": 5, "stamp_size": 11}


def write_frame(folder, fracday, obsjd, rng, pointing_shift=0.0):
    prefix = folder / f"ztf_{fracday}_000618_zr_c03_o_q4_"
    header = fits.Header({"SATURATE": 60000.0, "OBSJD": obsjd, "MAGZP": 26.0, "GAIN": 6.2, "SEEING": 2.0,
                          "MAGLIM": 20.5, "PIXSCALE": 1.0, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                          "CRVAL1": 135.6, "CRVAL2": 22.8, "CRPIX1": 100.0 + pointing_shift, "CRPIX2": 100.0 + pointing_shift,
                          "CD1_1": -1 / 3600, "CD2_2": 1 / 3600})
    science = 500 + rng.normal(0, 5, (200, 200))
    difference = rng.normal(0, 5, (200, 200))
    yy, xx = np.indices((25, 25)) - 12
    psf = np.exp(-(xx**2 + yy**2) / 2.0)
    fits.PrimaryHDU(science, header).writeto(f"{prefix}sciimg.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(difference.astype(np.float32), header)]).writeto(f"{prefix}scimrefdiffimg.fits.fz")
    fits.PrimaryHDU(np.zeros((200, 200), dtype=np.int16)).writeto(f"{prefix}mskimg.fits")
    fits.PrimaryHDU(psf / psf.sum()).writeto(f"{prefix}diffimgpsf.fits")
    return f"ztf_{fracday}_000618_zr_c03_o_q4"


def test_one_fake_night_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    paths = {"raw": tmp_path / "raw", "interim": tmp_path / "interim", "processed": tmp_path / "processed"}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)

    # The telescope pointing moves 8 px between the frames, as it does on real nights
    names = [write_frame(night_dir, 20200101400000 + i, 2458849.9 + i / 96, rng, pointing_shift=8.0 * i) for i in range(2)]
    metadata = pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0,
                             "obsjd": [2458849.9, 2458849.9 + 1 / 96], "filefracday": [20200101400000, 20200101400001]})

    # Pretend SkyBoT already answered "no known asteroids" for both frames
    (paths["interim"] / "known_asteroids").mkdir(parents=True)
    for name in names:
        pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"]).to_csv(
            paths["interim"] / "known_asteroids" / f"known_asteroids__{name}.csv", index=False)

    detections, stamps, injections = build_night_detections(UNIT, metadata, ACQUIRE, INJECT, DETECT, paths, rng)

    assert len(injections) == 3 * 2                               # 3 fakes x 2 frames
    assert (detections["label"] == 1).sum() >= 5                   # bright fakes are found
    assert stamps.shape == (len(detections), 3, 11, 11)
    assert {"ra", "dec", "obsjd", "frame_index", "mag", "x_common", "y_common"} <= set(detections.columns)

    # Fakes move steadily ON THE SKY: in frame 0's common grid each one moves by its planned
    # amount, even though frame 1's own pixels are shifted by 8 px.
    found = detections[detections["injection_id"] >= 0]
    for _, pair in found.groupby("injection_id"):
        if len(pair) == 2:
            step = np.hypot(*np.diff(pair.sort_values("frame_index")[["x_common", "y_common"]].to_numpy(), axis=0)[0])
            assert abs(step - 30.0 * 0.25) < 1.0      # 30 "/hr x 15 minutes at 1 "/px = 7.5 px

    stamps_out, labels, rows = assemble_stamp_dataset([UNIT], paths["processed"], rng)
    assert len(stamps_out) == len(labels) == len(rows) == len(detections)


def test_planted_images_contain_the_fakes(tmp_path):
    from asthunt.clean.load_night_images import load_night_images

    rng = np.random.default_rng(0)
    paths = {"raw": tmp_path / "raw", "interim": tmp_path / "interim", "processed": tmp_path / "processed"}
    night_dir = paths["raw"] / "ztf" / NIGHT
    night_dir.mkdir(parents=True)
    names = [write_frame(night_dir, 20200101400000 + i, 2458849.9 + i / 96, rng) for i in range(2)]
    metadata = pd.DataFrame({"night": NIGHT, "filtercode": "zr", "infobits": 0,
                             "obsjd": [2458849.9, 2458849.9 + 1 / 96], "filefracday": [20200101400000, 20200101400001]})
    (paths["interim"] / "known_asteroids").mkdir(parents=True)
    for name in names:
        pd.DataFrame(columns=["name", "ra", "dec", "v_mag", "x", "y"]).to_csv(
            paths["interim"] / "known_asteroids" / f"known_asteroids__{name}.csv", index=False)
    _, _, injections = build_night_detections(UNIT, metadata, ACQUIRE, INJECT, DETECT, paths, rng)

    planted = load_night_images(UNIT, names, paths, ACQUIRE["bad_mask_bits"], with_injections=True)
    original = load_night_images(UNIT, names, paths, ACQUIRE["bad_mask_bits"], with_injections=False)

    fake = injections[injections["frame_index"] == 0].iloc[0]
    row, column = int(round(fake["y"])), int(round(fake["x"]))
    added = planted[names[0]]["difference"][row, column] - original[names[0]]["difference"][row, column]
    assert added > 100                                                   # a mag-16 fake is bright
    assert np.allclose(planted[names[0]]["reference"], original[names[0]]["reference"])   # never in the reference
