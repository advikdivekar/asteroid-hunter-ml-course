"""Correct probabilities from a model trained on a different class balance."""

import numpy as np


def adjust_for_base_rate(probabilities, training_real_fraction, true_real_fraction):
    """Return probabilities corrected for how common real objects really are.

    Bayes' rule in odds form: posterior odds = likelihood ratio x prior odds.
    A model trained with 1 real per 5 bogus (real fraction 1/6) has the
    training prior baked in. To use it where only 1 in 1000 is real, swap
    the prior odds:

        odds_new = odds_model x (true_prior_odds / training_prior_odds)

    Example: model says 0.9 (odds 9). Training odds 1/5, true odds 1/999:
    new odds = 9 x (1/999) / (1/5) = 0.045 -> probability 0.043.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    model_odds = p / (1 - p)
    training_odds = training_real_fraction / (1 - training_real_fraction)
    true_odds = true_real_fraction / (1 - true_real_fraction)
    new_odds = model_odds * true_odds / training_odds
    return new_odds / (1 + new_odds)
