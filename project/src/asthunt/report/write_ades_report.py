"""Write candidate observations in the Minor Planet Center's ADES format (draft only)."""

from astropy.time import Time

EXPOSURE_SECONDS = 30.0


def write_ades_report(candidates, members, detections, path, observatory_code="I41"):
    """Write a pipe-separated ADES (PSV) file with one line per detection; return the path.

    This is a DRAFT for learning the format. Do not submit it: the MPC needs
    vetted objects, usually seen on two or more nights, and ZTF already
    reports its own observations.

    Times: ZTF's OBSJD is when the shutter opened; ADES wants the middle of
    the exposure, so we add half the exposure time (15 s).
    Each tracklet gets a temporary designation (trkSub) like "ah00001".
    """
    lines = [
        "# version=2017",
        "# observatory",
        f"! mpcCode {observatory_code}",
        "# comment",
        "! line DRAFT produced by the asthunt learning project - NOT FOR SUBMISSION",
        "trkSub|mode|stn|obsTime|ra|dec|astCat|mag|band",
    ]
    for number, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        ids = members.loc[members["tracklet_id"] == candidate["tracklet_id"], "detection_id"]
        for _, detection in detections.loc[ids].sort_values("obsjd").iterrows():
            middle = Time(detection["obsjd"] + EXPOSURE_SECONDS / 2 / 86400, format="jd")
            lines.append("|".join([
                f"ah{number:05d}", "CCD", observatory_code,
                middle.isot[:23] + "Z",
                f"{detection['ra']:.6f}", f"{detection['dec']:.6f}",
                "Gaia2", f"{detection['mag']:.2f}", "r",
            ]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
