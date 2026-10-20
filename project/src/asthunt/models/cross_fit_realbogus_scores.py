"""Score training units with CNNs that never saw those nights (out-of-fold scores)."""

import numpy as np

from asthunt.dataset.assemble_stamp_dataset import assemble_stamp_dataset
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.models.train_realbogus_cnn import train_realbogus_cnn


def cross_fit_realbogus_scores(train_units, n_folds, validation_stamps, validation_labels,
                               processed_dir, bogus_per_real, settings, device, seed):
    """Return {unit tag: scores for ALL detections of that unit}, for every training unit.

    Why: the tracklet classifier uses CNN scores as features. A CNN scoring
    the very detections it was trained on is over-confident, so the tracklet
    classifier would learn from scores that never happen on new nights.

    How (k-fold by NIGHT): split the training nights into n_folds groups; for
    each group, train a CNN on the units of the OTHER nights and score the
    held-out night's units. All quadrants of one night are always in the same
    fold, so a night's conditions can never leak into its own scores.
    """
    rng = np.random.default_rng(seed)
    nights = sorted({unit["night"] for unit in train_units})
    folds = np.array_split(np.array(nights), n_folds)
    scores = {}

    for fold_number, held_out in enumerate(folds):
        held_out = {str(night) for night in held_out}
        fit_units = [unit for unit in train_units if unit["night"] not in held_out]
        print(f"fold {fold_number + 1}/{n_folds}: train on {len(fit_units)} units, score nights {sorted(held_out)}", flush=True)
        stamps, labels, _ = assemble_stamp_dataset(fit_units, processed_dir, rng, bogus_per_real)
        model, _ = train_realbogus_cnn(stamps, labels, validation_stamps, validation_labels,
                                       settings, device, seed + fold_number)
        for unit in train_units:
            if unit["night"] in held_out:
                unit_stamps, _, _ = assemble_stamp_dataset([unit], processed_dir, rng)
                scores[make_unit_tag(unit)] = predict_realbogus(model, unit_stamps, device)

    return scores
