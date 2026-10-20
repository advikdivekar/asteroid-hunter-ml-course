"""Train the simplest useful classifier: logistic regression on a few numeric features."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_baseline_classifier(features, labels, seed=42):
    """Fit a logistic regression and return the trained model.

    features : 2-D table (rows = examples, columns = features)
    labels   : 1-D array of 0 (bogus) or 1 (real)

    The scaler first puts every feature on the same scale (mean 0, std 1).
    Without it, a feature measured in thousands (rate in "/hr) would
    dominate one measured in fractions, purely because of its units.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed),
    )
    model.fit(features, labels)
    return model
