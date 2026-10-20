"""Choose which rows to train on when one class vastly outnumbers the other."""

import numpy as np


def balance_classes(labels, bogus_per_real, rng):
    """Return sorted row indices: every real row plus a random sample of bogus rows.

    With 34 bogus per real detection, a model can score 97% by always
    answering "bogus". Keeping at most `bogus_per_real` bogus rows per real
    one makes the model pay attention to the real class.
    The price: predicted probabilities are no longer "true" probabilities
    on the real sky. The metrics phase shows how to account for that.
    """
    labels = np.asarray(labels)
    real = np.flatnonzero(labels == 1)
    bogus = np.flatnonzero(labels == 0)

    n_bogus = min(len(bogus), bogus_per_real * len(real))
    chosen_bogus = rng.choice(bogus, size=n_bogus, replace=False)
    return np.sort(np.concatenate([real, chosen_bogus]))
