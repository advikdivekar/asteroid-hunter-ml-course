import pytest

from asthunt.evaluate.compute_classification_metrics import compute_classification_metrics


def test_counts_and_rates():
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]

    m = compute_classification_metrics(labels, scores, threshold=0.5)

    assert (m["true_positives"], m["false_negatives"], m["false_positives"], m["true_negatives"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)
