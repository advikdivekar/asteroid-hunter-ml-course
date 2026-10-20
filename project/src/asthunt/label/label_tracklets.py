"""Decide which linked tracklets are real objects."""

import pandas as pd


def label_tracklets(tracklets, members, detections):
    """Return a copy of `tracklets` with `label` (1 real, 0 bogus) and `injection_id` columns.

    A tracklet is real only if EVERY one of its detections belongs to the
    SAME object: the same injected fake, or the same known asteroid.
    A tracklet mixing a fake with noise, or two different fakes, is bogus.
    Those mixed ones are the most valuable "hard negatives": they look
    almost right.
    """
    joined = members.merge(detections[["injection_id", "known_asteroid"]],
                           left_on="detection_id", right_index=True)

    def is_single_object(group):
        one_fake = (group["injection_id"] >= 0).all() and group["injection_id"].nunique() == 1
        one_known = (group["known_asteroid"] != "").all() and group["known_asteroid"].nunique() == 1
        return one_fake or one_known

    real = joined.groupby("tracklet_id").apply(is_single_object, include_groups=False)
    first_injection = joined.groupby("tracklet_id")["injection_id"].max()

    labelled = tracklets.copy()
    labelled["label"] = labelled["tracklet_id"].map(real).fillna(False).astype(int)
    labelled["injection_id"] = labelled["tracklet_id"].map(first_injection).where(labelled["label"] == 1, -1).astype(int)
    return labelled
