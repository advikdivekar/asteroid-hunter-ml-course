"""The pictures the live viewer shows, one function per stage. Each returns PNG bytes."""

import io
import warnings

import matplotlib

matplotlib.use("Agg")                                           # no window: the figures only go to the web page
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

BACKGROUND, FOREGROUND, GRID = "#10151f", "#d7deea", "#2a3446"
TEAL, AMBER, PINK, BLUE, GREY = "#5eead4", "#fbbf24", "#f472b6", "#60a5fa", "#7b869a"
FRAME_COLOURS = ["#5eead4", "#60a5fa", "#a78bfa", "#f472b6", "#fbbf24", "#fb923c", "#a3e635"]
DISPLAY_PIXELS = 760                                            # a full chip (about 3,000 px) is shown at about this width


def png_from_figure(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _style(axis, title=None):
    axis.set_facecolor(BACKGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.tick_params(colors=FOREGROUND, labelsize=8)
    if title:
        axis.set_title(title, color=FOREGROUND, fontsize=10)


def block_mean(image, factor):
    """Shrink an image by averaging factor x factor blocks (NaN-safe), so thin features are not skipped."""
    height, width = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    blocks = image[:height, :width].reshape(height // factor, factor, width // factor, factor)
    with warnings.catch_warnings():                              # blocks that are entirely NaN give NaN, which is what we want
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def stretch(image, low=1.0, high=99.5):
    """Scale to 0..1 between two percentiles of the finite values (the usual way to look at astronomical images)."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=float)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((np.nan_to_num(image, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)


def shrink_factor(shape):
    return max(1, int(round(max(shape) / DISPLAY_PIXELS)))


def draw_four_panels(science, reference, difference, bad, headline):
    """Science, reference, difference and the bad-pixel mask of one exposure, side by side."""
    factor = shrink_factor(difference.shape)
    panels = [("science (what the telescope saw)", science, "gray", (1, 99.5)),
              ("reference (the same sky, long ago)", reference, "gray", (1, 99.5)),
              ("difference = science - reference", difference, "gray", (5, 99.5)),
              ("bad pixels (masked)", bad.astype(float), "magma", (0, 100))]
    figure, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for axis, (title, image, cmap, (lo, hi)) in zip(axes, panels):
        if image is None:
            axis.text(0.5, 0.5, "not loaded\n(light mode)", color=FOREGROUND, ha="center", va="center", transform=axis.transAxes)
        else:
            axis.imshow(stretch(block_mean(image.astype(float), factor), lo, hi), cmap=cmap, origin="lower")
        _style(axis, title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(headline, color=FOREGROUND, fontsize=11)
    return png_from_figure(figure)


def draw_noise_histogram(values, sigma_robust, sigma_plain):
    """Histogram of the difference image's pixel values with the Gaussian the noise should follow."""
    values = values[np.isfinite(values)]
    clipped = values[np.abs(values) < 8 * sigma_robust]
    figure, axis = plt.subplots(figsize=(7.5, 3.8))
    counts, edges, _ = axis.hist(clipped, bins=160, color=TEAL, alpha=0.75, label="pixels of this frame")
    centres = (edges[:-1] + edges[1:]) / 2
    gaussian = counts.max() * np.exp(-0.5 * (centres / sigma_robust) ** 2)
    axis.plot(centres, gaussian, color=AMBER, lw=1.6, label=f"pure noise, sigma = {sigma_robust:.1f}")
    axis.set_yscale("log")
    axis.set_ylim(bottom=1)
    for k in (3, 5):
        axis.axvline(k * sigma_robust, color=PINK, lw=1, ls="--")
        axis.text(k * sigma_robust, axis.get_ylim()[1], f" {k} sigma", color=PINK, fontsize=8, va="top")
    _style(axis, f"Pixel values of the difference image (robust sigma {sigma_robust:.1f}, plain std {sigma_plain:.1f})")
    axis.set_xlabel("pixel value (counts)", color=FOREGROUND, fontsize=9)
    axis.legend(facecolor=BACKGROUND, edgecolor=GRID, labelcolor=FOREGROUND, fontsize=8)
    return png_from_figure(figure)


CROP_PIXELS = 360                                               # the zoomed view is this many pixels wide, at full resolution


def draw_detections(difference, detections, headline):
    """Left: the whole difference image, the zoom box, and a numbered ring on EVERY planted fake and known asteroid.
    Right: the zoom box at full resolution with a ring on every blob.

    Ring colours: teal = planted fake (numbered), blue = known asteroid, grey = unlabelled. The zoom is centred on a fake or known
    object if there is one (so you see what a real object looks like), else on the brightest blob.
    """
    factor = shrink_factor(difference.shape)
    height, width = difference.shape
    fake = detections["injection_id"].to_numpy() >= 0 if len(detections) else np.array([], dtype=bool)
    known = (detections["known_asteroid"].to_numpy() != "") & ~fake if len(detections) else fake
    if len(detections):
        labelled = np.where(fake | known)[0]
        centre_row = labelled[0] if len(labelled) else int(np.argmax(detections["snr"].to_numpy())) if "snr" in detections else 0
        cx, cy = detections["x"].iloc[centre_row], detections["y"].iloc[centre_row]
    else:
        cx, cy = width / 2, height / 2
    x0 = int(np.clip(cx - CROP_PIXELS / 2, 0, max(width - CROP_PIXELS, 0)))
    y0 = int(np.clip(cy - CROP_PIXELS / 2, 0, max(height - CROP_PIXELS, 0)))

    figure, (whole, zoom) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1, 1]})
    whole.imshow(stretch(block_mean(difference.astype(float), factor), 5, 99.7), cmap="gray", origin="lower")
    whole.add_patch(plt.Rectangle((x0 / factor, y0 / factor), CROP_PIXELS / factor, CROP_PIXELS / factor, fill=False, edgecolor=AMBER, lw=1.6))
    if len(detections):                                         # big rings, so a single fake is easy to spot on a 3,000-pixel chip
        for mask, colour in [(known, BLUE), (fake, TEAL)]:
            whole.scatter(detections["x"].to_numpy()[mask] / factor, detections["y"].to_numpy()[mask] / factor, s=420, facecolors="none", edgecolors=colour, linewidths=2)
        for _, row in detections[fake].iterrows():
            whole.annotate(f"fake #{int(row['injection_id'])}", (row["x"] / factor, row["y"] / factor), color=TEAL, fontsize=9, fontweight="bold",
                           xytext=(14, 14), textcoords="offset points")
    _style(whole, f"{headline}  |  yellow square = the zoom on the right")
    whole.set_xticks([])
    whole.set_yticks([])

    crop = difference[y0:y0 + CROP_PIXELS, x0:x0 + CROP_PIXELS].astype(float)
    zoom.imshow(stretch(crop, 5, 99.7), cmap="gray", origin="lower", extent=(x0, x0 + crop.shape[1], y0, y0 + crop.shape[0]))
    if len(detections):
        inside = ((detections["x"] >= x0) & (detections["x"] < x0 + CROP_PIXELS) & (detections["y"] >= y0) & (detections["y"] < y0 + CROP_PIXELS)).to_numpy()
        for mask, colour, radius in [(~(fake | known), GREY, 6), (known, BLUE, 9), (fake, TEAL, 9)]:
            pick = inside & mask
            zoom.scatter(detections["x"].to_numpy()[pick], detections["y"].to_numpy()[pick], s=radius ** 2 * 2.2, facecolors="none", edgecolors=colour, linewidths=1.1)
        _style(zoom, f"zoom: {int(inside.sum())} blobs inside this {CROP_PIXELS} x {CROP_PIXELS} pixel square")
    else:
        _style(zoom, "zoom")
    zoom.set_xticks([])
    zoom.set_yticks([])
    return png_from_figure(figure)


def draw_stamp_mosaic(stamps, captions, headline, columns=4):
    """Triplet stamps (n, 3, size, size) as rows of science | reference | difference, each with a caption."""
    count = len(stamps)
    rows = int(np.ceil(count / columns))
    figure = plt.figure(figsize=(columns * 3.3, rows * 1.2 + 0.7))
    grid = figure.add_gridspec(rows, columns * 4, wspace=0.05, hspace=0.4, width_ratios=[1, 1, 1, 0.45] * columns)
    for k in range(count):
        row, column = divmod(k, columns)
        for channel in range(3):
            axis = figure.add_subplot(grid[row, column * 4 + channel])
            axis.imshow(stretch(stamps[k, channel].astype(float), 1, 99.5), cmap="gray", origin="lower")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(GRID)
            if channel == 1:
                axis.set_title(captions[k], color=FOREGROUND, fontsize=7)
    figure.suptitle(headline + "   (each group: science | reference | difference)", color=FOREGROUND, fontsize=10)
    return png_from_figure(figure)


def draw_score_histogram(scores, threshold, groups=None):
    """How the real/bogus model scored every detection, with the cut that decides who goes on to linking."""
    figure, axis = plt.subplots(figsize=(7.5, 3.8))
    bins = np.linspace(0, 1, 51)
    axis.hist(scores, bins=bins, color=GREY, alpha=0.85, label=f"all detections ({len(scores):,})")
    for label, colour, mask in (groups or []):
        if mask.any():
            axis.hist(scores[mask], bins=bins, color=colour, alpha=0.9, label=f"{label} ({int(mask.sum())})")
    axis.axvline(threshold, color=AMBER, lw=1.6)
    axis.text(threshold, axis.get_ylim()[1], f" cut = {threshold:.3f}", color=AMBER, fontsize=8, va="top")
    axis.set_yscale("log")
    _style(axis, "real/bogus score: probability that a detection is a real moving object")
    axis.set_xlabel("p(real)", color=FOREGROUND, fontsize=9)
    axis.legend(facecolor=BACKGROUND, edgecolor=GRID, labelcolor=FOREGROUND, fontsize=8)
    return png_from_figure(figure)


def draw_tracklets(kept, members, tracklets, frame_count, headline):
    """Left: every detection that passed the real/bogus cut on the common grid, with the tracklets over them. Right: close-ups of the best tracklets."""
    figure = plt.figure(figsize=(14, 5.4))
    grid = figure.add_gridspec(2, 5, width_ratios=[2.2, 1, 1, 1, 1], wspace=0.35, hspace=0.45)
    chip = figure.add_subplot(grid[:, 0])
    chip.scatter(kept["x_common"], kept["y_common"], s=1, color=GREY, alpha=0.5)
    for _, tracklet in tracklets.iterrows():
        rows = members[members["tracklet_id"] == tracklet["tracklet_id"]]
        chip.scatter(kept.loc[rows["detection_id"], "x_common"], kept.loc[rows["detection_id"], "y_common"], s=4, color=TEAL)
    chip.set_aspect("equal")
    _style(chip, f"{len(kept):,} detections kept by the real/bogus model; {len(tracklets)} tracklets (teal)")
    best = tracklets.sort_values(["n_detections", "max_residual"], ascending=[False, True]).head(8)
    for k, (_, tracklet) in enumerate(best.iterrows()):
        axis = figure.add_subplot(grid[k // 4, 1 + k % 4])
        rows = members[members["tracklet_id"] == tracklet["tracklet_id"]]
        points = kept.loc[rows["detection_id"]]
        axis.plot(points["x_common"], points["y_common"], color=GRID, lw=1)
        axis.scatter(points["x_common"], points["y_common"], s=30, c=[FRAME_COLOURS[int(f) % len(FRAME_COLOURS)] for f in points["frame_index"]])
        axis.set_aspect("equal")
        _style(axis, f"#{int(tracklet['tracklet_id'])}: {int(tracklet['n_detections'])}/{frame_count} frames, {tracklet['rate_arcsec_per_hour']:.0f}\"/h")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(headline, color=FOREGROUND, fontsize=10)
    return png_from_figure(figure)


def draw_tracklet_scores(tracklets, threshold, title):
    """Every tracklet's classifier score, sorted, coloured by what it turned out to be."""
    ordered = tracklets.sort_values("tracklet_score", ascending=False).head(40)
    colours = [TEAL if fake else BLUE if name else PINK for fake, name in zip(ordered["contains_fake"], ordered["known_name"])]
    figure, axis = plt.subplots(figsize=(7.8, max(2.6, 0.2 * len(ordered) + 1.2)))
    axis.barh(range(len(ordered)), ordered["tracklet_score"], color=colours)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([f"#{int(t)}" for t in ordered["tracklet_id"]], fontsize=6)
    axis.invert_yaxis()
    axis.axvline(threshold, color=AMBER, lw=1.5)
    axis.set_xlim(0, 1)
    _style(axis, title)
    return png_from_figure(figure)
