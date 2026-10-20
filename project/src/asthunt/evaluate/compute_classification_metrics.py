"""Summarise how good a classifier's scores are."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_classification_metrics(labels, scores, threshold=0.5):
    """Return a dict of counts and scores for one decision threshold.

    labels    : 1 = real, 0 = bogus
    scores    : model output, higher = more likely real
    threshold : scores >= threshold are called "real"

    Counts (the confusion matrix):
      true_positives  real, called real       false_negatives real, called bogus
      false_positives bogus, called real      true_negatives  bogus, called bogus
    Rates:
      recall (completeness) = TP / (TP + FN)   "how many real ones did we catch?"
      precision (purity)    = TP / (TP + FP)   "how many alerts are real?"
      false_positive_rate   = FP / (FP + TN)
    Threshold-free summaries:
      average_precision : area under the precision-recall curve (1.0 = perfect)
      roc_auc           : area under the ROC curve (0.5 = coin flip)
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold

    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": (tp + tn) / len(labels),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "average_precision": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
    }
