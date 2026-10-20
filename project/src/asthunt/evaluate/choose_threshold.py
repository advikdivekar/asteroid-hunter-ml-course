"""Pick the score threshold that keeps a chosen fraction of real objects."""

import numpy as np


def choose_threshold(labels, scores, target_recall):
    """Return the highest threshold whose recall on these data is >= target_recall.

    Use VALIDATION data: choosing the threshold on test data would leak
    the test set into a decision, making the final test score optimistic.

    Example: target_recall = 0.95 means "accept whatever false positives it
    takes to keep 95% of real objects".
    """
    real_scores = np.sort(np.asarray(scores)[np.asarray(labels) == 1])
    # Keep the top target_recall fraction of real scores: the cut sits at the
    # (1 - target_recall) quantile of the real objects' scores.
    index = int(np.floor((1 - target_recall) * len(real_scores)))
    return float(real_scores[index])
