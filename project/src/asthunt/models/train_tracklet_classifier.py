"""Train a gradient-boosted tree model that scores tracklets."""

from xgboost import XGBClassifier


def train_tracklet_classifier(features, labels, seed=42):
    """Fit and return an XGBoost classifier (use .predict_proba(features)[:, 1]).

    Gradient boosting builds many small decision trees one after another;
    each new tree focuses on the examples the previous trees got wrong.
    It works very well on tables of numbers like our tracklet features,
    needs no feature scaling, and copes with features on different scales.

    Settings are deliberately modest (shallow trees, small learning rate)
    because we have only hundreds to a few thousand tracklets.
    """
    model = XGBClassifier(
        n_estimators=300,       # number of trees
        max_depth=3,            # each tree asks at most 3 questions in a row
        learning_rate=0.05,     # how much each new tree is allowed to change the answer
        subsample=0.8,          # each tree sees a random 80% of rows (less overfitting)
        colsample_bytree=0.8,   # ...and a random 80% of features
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(features, labels)
    return model
