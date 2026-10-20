"""Check that no group (night, field, image...) appears in both train and test."""


def check_split_leakage(train_groups, test_groups):
    """Return the sorted list of groups found in BOTH train and test.

    An empty list means the split is clean.

    Example: if detections from night 2018-04-28 are in train AND test, the
    model can memorise that night's seeing, artifacts and stars, and the
    test score will be too optimistic.
    """
    shared = set(train_groups) & set(test_groups)
    return sorted(shared)
