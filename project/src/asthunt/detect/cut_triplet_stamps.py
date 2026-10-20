"""Cut small square images (stamps) around detections."""

import numpy as np


def cut_triplet_stamps(science_image, reference_image, difference_image, x_positions, y_positions, size=31):
    """Return an array of shape (n_detections, 3, size, size).

    For every detection we cut the same square from three images:
        channel 0: science     (the new exposure)
        channel 1: reference   (the deep template, on the same pixel grid)
        channel 2: difference  (science - reference)

    Seeing all three lets a classifier tell, for example, a real new point
    source (present in science and difference, absent in reference) from a
    bad subtraction (a bright star in all three with a messy residual).

    Pixels outside the image or NaN are set to 0.
    """
    half = size // 2
    images = [science_image, reference_image, difference_image]
    stamps = np.zeros((len(x_positions), 3, size, size), dtype=np.float32)

    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        column, row = int(round(x)), int(round(y))
        for channel, image in enumerate(images):
            stamps[i, channel] = _cut_one(image, row, column, half)

    return np.nan_to_num(stamps, nan=0.0)


def _cut_one(image, row, column, half):
    """Cut a (2*half+1) square centred on (row, column), padding with 0 outside the image."""
    size = 2 * half + 1
    stamp = np.zeros((size, size), dtype=np.float32)

    # Overlap between the wanted square and the image
    row_start, row_end = max(row - half, 0), min(row + half + 1, image.shape[0])
    col_start, col_end = max(column - half, 0), min(column + half + 1, image.shape[1])
    if row_start >= row_end or col_start >= col_end:
        return stamp

    stamp[row_start - (row - half): row_end - (row - half),
          col_start - (column - half): col_end - (column - half)] = image[row_start:row_end, col_start:col_end]
    return stamp
