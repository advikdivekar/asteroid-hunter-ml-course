"""Turn evaluation runs of the pipeline into one row of headline numbers."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from asthunt.evaluate.find_magnitude_at_completeness import find_magnitude_at_completeness
from asthunt.evaluate.measure_completeness import measure_completeness


def build_scoreboard(night_results, thresholds, min_detections, known_max_v_mag):
    """Return a dict of headline metrics over one or more nights.

    night_results : list of run_night_pipeline outputs made WITH injections
    thresholds    : the bundle's thresholds (realbogus, tracklet)

    Definitions (the same for every version, so comparisons are fair):
      detection_ap            average precision of the real/bogus scores, all detections
      detection_recall        fraction of real detections kept by the realbogus threshold
      bogus_kept_per_frame    bogus detections kept by that threshold, per frame
      tracklet_ap             average precision of tracklet scores; a tracklet counts as
                              real if it is a fake OR matches any known asteroid
      mag_at_50_completeness  magnitude where 50% of planted asteroids end up as accepted tracklets
      known_recovered         known asteroids (V <= known_max_v_mag, in >= min_detections frames)
                              found as accepted tracklets / how many there were
      unexplained_per_night   accepted tracklets that are neither a fake nor a known
                              asteroid, per night: what a human would have to vet
    """
    detections = pd.concat([r["detections"] for r in night_results], ignore_index=True)
    n_frames = detections.groupby(["night", "exposure"]).ngroups
    kept = detections["score"] >= thresholds["realbogus"]

    accepted_fakes, planted, tracklet_tables = [], [], []
    known_total, known_found, unexplained = 0, 0, 0
    for result in night_results:
        tracklets, night = result["tracklets"], result["detections"]["night"].iloc[0]
        injections = result["injections"]
        accepted = tracklets[tracklets["tracklet_score"] >= thresholds["tracklet"]] if len(tracklets) else tracklets

        # Planted asteroids: one row each, found if an accepted tracklet is exactly that fake
        per_fake = injections.groupby("injection_id").agg(mag=("mag", "first")).reset_index()
        found_ids = set(accepted.loc[accepted["label"] == 1, "injection_id"]) if len(accepted) else set()
        per_fake["found"] = per_fake["injection_id"].isin(found_ids)
        planted.append(per_fake)

        # Known asteroids we should be able to see tonight
        known = result["known"]
        bright = known[known["v_mag"] <= known_max_v_mag]
        frames_seen = bright.groupby("name")["exposure"].nunique()
        detectable = set(frames_seen[frames_seen >= min_detections].index)
        recovered = set(accepted["known_name"]) & detectable if len(accepted) else set()
        known_total += len(detectable)
        known_found += len(recovered)

        if len(accepted):
            unexplained += int(((accepted["known_name"] == "") & (~accepted["contains_fake"].astype(bool))).sum())
            tracklet_tables.append(tracklets)

    planted = pd.concat(planted, ignore_index=True)
    completeness = measure_completeness(planted, "found", "mag", np.arange(17, 22.25, 0.25))

    row = {
        "nights": len(night_results),
        "frames": n_frames,
        "detection_ap": average_precision_score(detections["label"], detections["score"]),
        "detection_recall": float(kept[detections["label"] == 1].mean()),
        "bogus_kept_per_frame": float((kept & (detections["label"] == 0)).sum() / n_frames),
        "tracklet_ap": float("nan"),
        "mag_at_50_completeness": find_magnitude_at_completeness(completeness, 0.5),
        "known_recovered": f"{known_found}/{known_total}",
        "unexplained_per_night": unexplained / len(night_results),
    }
    if tracklet_tables:
        all_tracklets = pd.concat(tracklet_tables, ignore_index=True)
        is_real = (all_tracklets["label"] == 1) | (all_tracklets["known_name"] != "")
        if is_real.nunique() == 2:
            row["tracklet_ap"] = average_precision_score(is_real, all_tracklets["tracklet_score"])
    return row
