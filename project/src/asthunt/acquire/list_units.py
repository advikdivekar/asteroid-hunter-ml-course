"""List every unit (one CCD quadrant on one night) that belongs to a split."""


def list_units(acquire, split, include_extra=True):
    """Return a list of {field, ccd, quadrant, night} dicts for split = train / validation / test / holdout.

    Where units come from (configs/acquire.yaml):
      - the original quadrant (field, ccd, quadrant) on `nights[split]` (holdout: `holdout_nights`)
      - every `extra_quadrants` entry on the SAME nights, so a night stays in one split
      - every `extra_sources` entry whose `split` matches
    include_extra=False returns only the original quadrant (what round 1 used).

    Splitting is always by NIGHT: all quadrants of a night share its split, so
    nothing about a night's conditions (seeing, moon, artifacts) can leak.
    """
    nights = acquire["holdout_nights"] if split == "holdout" else acquire["nights"][split]
    quadrants = [(acquire["ccd"], acquire["quadrant"])]
    if include_extra:
        quadrants += [tuple(q) for q in acquire.get("extra_quadrants", [])]

    units = [{"field": acquire["field"], "ccd": ccd, "quadrant": quadrant, "night": night}
             for night in nights for ccd, quadrant in quadrants]

    if include_extra:
        for source in acquire.get("extra_sources", []):
            if source["split"] == split:
                units += [{"field": source["field"], "ccd": source["ccd"], "quadrant": source["quadrant"], "night": night}
                          for night in source["nights"]]
    return units
