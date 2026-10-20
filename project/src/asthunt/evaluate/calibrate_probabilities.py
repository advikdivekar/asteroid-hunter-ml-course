"""Learn a mapping from raw model scores to honest probabilities."""

from sklearn.isotonic import IsotonicRegression


def calibrate_probabilities(validation_scores, validation_labels):
    """Fit and return a calibrator; use it as calibrator.predict(new_scores).

    A model's score of 0.8 does not automatically mean "80% of such cases
    are real". Training on balanced classes, for example, inflates scores.
    Isotonic regression learns a monotonic (never decreasing) curve from
    score to observed fraction of real objects, using validation data that
    has the REAL class balance.
    """
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(validation_scores, validation_labels)
    return calibrator
