"""Describe each stamp triplet with a few dozen numbers, so a classical model (trees, SVM) can judge it."""

import numpy as np
import pandas as pd

# Numbers measured by the blob finder (detect/extract_sources.py), already in the detections table.
DETECTION_COLUMNS = ["snr", "peak", "npix", "a", "b", "flux"]

STAMP_COLUMNS = [
    "sci_centre", "ref_centre", "diff_centre",
    "sci_peak", "ref_peak", "diff_peak",
    "diff_annulus", "ref_annulus",
    "ref_to_sci", "diff_to_sci",
    "diff_concentration", "diff_sharpness",
    "diff_min", "diff_negative_fraction", "diff_dipole",
    "diff_psf_match", "ref_psf_match",
    "diff_size", "diff_elongation", "diff_offset",
    "diff_symmetry", "diff_bright_pixels", "diff_roughness",
]

FEATURE_COLUMNS = DETECTION_COLUMNS + STAMP_COLUMNS


def compute_stamp_features(stamps, detections):
    """Return a table with FEATURE_COLUMNS, one row per stamp.

    stamps     : (n, 3, size, size), NORMALIZED (dataset/normalize_stamps.py), so a value
                 of 5 means 5 times the noise on any night. Channel 0 science, 1 reference,
                 2 difference.
    detections : the matching rows of the detections table (for the blob finder's numbers).

    Each number answers one question a person would ask looking at the three pictures:
      - is there something in the middle of tonight's picture and of the difference? (centre, peak)
      - was it already there in the old picture? (ref_centre, ref_to_sci): a star that did not subtract
      - is it shaped like a star, not a sharp spike or a smear? (psf_match, sharpness, size, elongation)
      - is the difference a clean bump or a plus/minus pair (a dipole from misalignment)? (min, dipole)
      - is it centred and round? (offset, symmetry)
    """
    stamps = np.asarray(stamps, dtype=np.float32)
    n, _, size, _ = stamps.shape
    c = size // 2
    sci, ref, diff = stamps[:, 0], stamps[:, 1], stamps[:, 2]

    yy, xx = np.mgrid[:size, :size] - c
    radius = np.hypot(xx, yy)
    centre = radius <= 1.5                 # the 3x3-ish core
    core = radius <= 3.5                   # where a star's light is
    annulus = (radius >= 6) & (radius <= 10)
    window = (np.abs(xx) <= 5) & (np.abs(yy) <= 5)

    def mean_in(images, mask):
        return images[:, mask].mean(axis=1)

    def max_in(images, mask):
        return images[:, mask].max(axis=1)

    f = {}
    f["sci_centre"], f["ref_centre"], f["diff_centre"] = mean_in(sci, centre), mean_in(ref, centre), mean_in(diff, centre)
    f["sci_peak"], f["ref_peak"], f["diff_peak"] = max_in(sci, core), max_in(ref, core), max_in(diff, core)
    f["diff_annulus"], f["ref_annulus"] = mean_in(diff, annulus), mean_in(ref, annulus)

    # A new object: bright in science, nothing in the reference, so all of its light is in the difference.
    sci_level = np.maximum(f["sci_centre"], 1.0)
    f["ref_to_sci"] = f["ref_centre"] / sci_level
    f["diff_to_sci"] = f["diff_centre"] / sci_level

    core_sum = diff[:, core].sum(axis=1)
    f["diff_concentration"] = diff[:, centre].sum(axis=1) / np.maximum(np.abs(core_sum), 1.0)
    f["diff_sharpness"] = f["diff_peak"] / np.maximum(f["diff_centre"], 1.0)      # cosmic rays: one hot pixel

    f["diff_min"] = diff[:, window].min(axis=1)
    f["diff_negative_fraction"] = (diff[:, window] < -3).mean(axis=1)
    f["diff_dipole"] = -f["diff_min"] / np.maximum(f["diff_peak"], 1.0)           # strong negative next to positive

    psf = np.exp(-(radius[window] ** 2) / (2 * 1.2 ** 2))                        # a round star, about ZTF's blur
    f["diff_psf_match"] = _correlation(diff[:, window], psf)
    f["ref_psf_match"] = _correlation(ref[:, window], psf)

    weights = np.clip(diff[:, window], 0, None)                                   # second moments of the positive light
    total = np.maximum(weights.sum(axis=1), 1e-6)
    x, y = xx[window], yy[window]
    mx, my = (weights * x).sum(axis=1) / total, (weights * y).sum(axis=1) / total
    sxx = (weights * x ** 2).sum(axis=1) / total - mx ** 2
    syy = (weights * y ** 2).sum(axis=1) / total - my ** 2
    sxy = (weights * x * y).sum(axis=1) / total - mx * my
    spread = np.sqrt(np.maximum((sxx - syy) ** 2 / 4 + sxy ** 2, 0))
    major, minor = (sxx + syy) / 2 + spread, np.maximum((sxx + syy) / 2 - spread, 1e-3)
    f["diff_size"] = np.sqrt(np.maximum(major + minor, 0))
    f["diff_elongation"] = np.sqrt(np.maximum(major, 0) / minor)
    f["diff_offset"] = np.hypot(mx, my)

    patch = diff[:, window]
    rotated = diff[:, ::-1, ::-1][:, window]                                      # the same patch turned 180 degrees
    f["diff_symmetry"] = _correlation_pairs(patch, rotated)
    f["diff_bright_pixels"] = (patch > 3).sum(axis=1)
    f["diff_roughness"] = (np.abs(np.diff(diff[:, window.any(axis=1)][:, :, window.any(axis=0)], axis=2)).mean(axis=(1, 2))
                           / np.maximum(np.abs(f["diff_centre"]), 1.0))

    table = pd.DataFrame({name: f[name] for name in STAMP_COLUMNS}, index=detections.index)
    for name in DETECTION_COLUMNS:
        table.insert(DETECTION_COLUMNS.index(name), name, detections[name].to_numpy())
    return table[FEATURE_COLUMNS].astype(np.float32)


def _correlation(patches, template):
    """Pearson correlation of every row of `patches` with one template (1 = same shape)."""
    a = patches - patches.mean(axis=1, keepdims=True)
    b = template - template.mean()
    return (a @ b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b), 1e-6)


def _correlation_pairs(a, b):
    """Pearson correlation of each row of a with the same row of b."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6)
