"""The one place that decides how files of one 'unit' (one CCD quadrant on one night) are named."""


def make_unit_tag(unit):
    """Return e.g. "000618_c03_q4_2019-02-08" for unit = {field, ccd, quadrant, night}.

    Every per-night file uses this tag: detections__<tag>.parquet, stamps__<tag>.npy, ...
    Including CCD and quadrant means two quadrants observed on the same night
    can never overwrite each other's files.
    """
    return f"{int(unit['field']):06d}_c{int(unit['ccd']):02d}_q{int(unit['quadrant'])}_{unit['night']}"
