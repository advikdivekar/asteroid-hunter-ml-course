"""Export real data from the test night for the web app (web/public/data).

What it writes:
  cnn_weights.bin + cnn.json   the v4 real/bogus CNN weights, so the browser runs the real model
  stamps.bin + stamps.json     48 real 3x31x31 stamps (planted fakes, known asteroids, junk), normalised
  scene/*.png + scene.json     a 160x160 window of the night: 5 frames x (science, reference, difference),
                               every blob in it with its CNN score, and the tracks that pass through it
  night.json                   the whole-night funnel (blobs -> kept -> tracks -> known) and the scoreboard

Run from anywhere with the asthunt environment:
  python tools/export_web_data.py --source ~/Desktop/asteroid
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

UNIT = "000618_c03_q4_2019-02-08"
WINDOW = 160


def stretch(image, low, high):
    """asinh stretch to 8 bit, the usual way to show faint and bright things together."""
    scaled = np.clip((image - low) / (high - low), 0, None)
    return (255 * np.clip(np.arcsinh(scaled * 10) / np.arcsinh(10), 0, 1)).astype(np.uint8)


def detection_metrics(labels, scores, cut, frames):
    """Blob-level metrics on the whole night: at the cut we deploy, and at fixed recall (how models are compared)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    def at(threshold):
        kept = scores >= threshold
        tp, fp = int((kept & (labels == 1)).sum()), int((kept & (labels == 0)).sum())
        fn, tn = int((~kept & (labels == 1)).sum()), int((~kept & (labels == 0)).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": recall, "precision": precision,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12), "accuracy": (tp + tn) / len(labels),
                "false_alarms_per_photo": fp / frames}

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    at_recall = {}
    for target in (0.90, 0.95):
        i = np.where(recall[:-1] >= target)[0][-1]                 # highest threshold that still reaches the target recall
        at_recall[f"{target:.2f}"] = at(thresholds[i])
    return {"average_precision": float(average_precision_score(labels, scores)), "roc_auc": float(roc_auc_score(labels, scores)),
            "always_bogus_accuracy": float((labels == 0).mean()), "deployed": at(cut), "at_recall": at_recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path.home() / "Desktop" / "asteroid")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "web" / "public" / "data")
    args = parser.parse_args()
    src = args.source
    sys.path.insert(0, str(src / "src"))
    from asthunt.clean.load_exposure import load_exposure
    from asthunt.dataset.normalize_stamps import normalize_stamps
    from asthunt.models.load_model_bundle import load_model_bundle
    from asthunt.models.predict_realbogus import predict_realbogus

    out = args.out
    (out / "scene").mkdir(parents=True, exist_ok=True)
    processed = src / "data" / "processed"
    bundle = load_model_bundle(src / "models", "v4")
    model, thresholds = bundle["realbogus_model"], bundle["thresholds"]

    # 1. CNN weights, in PyTorch order
    layers, blobs, offset = [], [], 0
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().numpy().astype(np.float32).ravel()
        layers.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": int(values.size)})
        blobs.append(values)
        offset += values.size
    np.concatenate(blobs).tofile(out / "cnn_weights.bin")
    (out / "cnn.json").write_text(json.dumps({"version": "v4", "parameters": offset, "threshold": thresholds["realbogus"], "layers": layers}, indent=1))

    # 2. Detections, stamps and the real CNN score for the whole night
    detections = pd.read_parquet(processed / f"detections__{UNIT}.parquet")
    stamps = np.load(processed / f"stamps__{UNIT}.npy", mmap_mode="r")
    normalized = normalize_stamps(np.asarray(stamps, dtype=np.float32))
    detections["score"] = predict_realbogus(model, normalized)
    detections["kind"] = np.where(detections["injection_id"] >= 0, "fake", np.where(detections["known_asteroid"] != "", "known", "bogus"))

    rng = np.random.default_rng(7)
    picks = []
    fakes = detections[detections["kind"] == "fake"].sort_values("mag")
    picks += list(fakes.iloc[np.linspace(0, len(fakes) - 1, 14).astype(int)].index)
    known = detections[detections["kind"] == "known"]
    picks += list(known.drop_duplicates("known_asteroid").sample(min(10, known["known_asteroid"].nunique()), random_state=1).index)
    bogus = detections[detections["kind"] == "bogus"]
    picks += list(bogus.nlargest(8, "snr").index)                    # bright junk: star residuals, spikes
    picks += list(bogus.nlargest(6, "score").index)                  # junk the CNN finds hardest
    picks += list(rng.choice(bogus.index, 10, replace=False))        # typical junk: noise
    picks = list(dict.fromkeys(picks))
    sample = detections.loc[picks]
    normalized[picks].astype(np.float32).tofile(out / "stamps.bin")
    (out / "stamps.json").write_text(json.dumps([
        {"i": n, "kind": row["kind"], "label": int(row["label"]), "snr": round(float(row["snr"]), 1), "mag": round(float(row["mag"]), 2),
         "frame": int(row["frame_index"]) + 1, "name": row["known_asteroid"] or None, "score_python": round(float(row["score"]), 5)}
        for n, (_, row) in enumerate(sample.iterrows())], indent=1))

    # 3. Tracks of the night, with names of known asteroids
    tracklets = pd.read_parquet(processed / f"tracklets__{UNIT}.parquet")
    members = pd.read_parquet(processed / f"tracklet_members__{UNIT}.parquet")
    members = members.join(detections[["x_common", "y_common", "frame_index", "known_asteroid", "score", "mag"]], on="detection_id")
    names = members[members["known_asteroid"] != ""].groupby("tracklet_id")["known_asteroid"].agg(lambda s: s.mode().iloc[0])
    tracklets["known_name"] = tracklets["tracklet_id"].map(names).fillna("")

    # 4. A window around one clear known asteroid track with a fake nearby
    exposures = detections.drop_duplicates("frame_index").sort_values("frame_index")[["frame_index", "exposure", "obsjd"]]
    first_reference = load_exposure(src / "data" / "raw" / "ztf" / "2019-02-08" / f"{exposures.iloc[0]['exposure']}_sciimg.fits",
                                    [0, 2, 3, 4, 7, 8, 9, 10, 12])["reference"]
    good = tracklets[(tracklets["known_name"] != "") & tracklets["mag_mean"].between(16.5, 19.0) & (tracklets["n_detections"] == 5)]
    best, best_count = None, -1
    for _, track in good.iterrows():
        path = members[members["tracklet_id"] == track["tracklet_id"]]
        x0 = int(np.clip(path["x_common"].mean() - WINDOW / 2, 0, 3072 - WINDOW))
        y0 = int(np.clip(path["y_common"].mean() - WINDOW / 2, 0, 3080 - WINDOW))
        ref = first_reference[y0:y0 + WINDOW, x0:x0 + WINDOW]
        if (np.isfinite(ref) & (ref != 0)).mean() < 0.99:          # skip windows at the edge of the old reference picture
            continue
        near = detections[(detections["kind"] == "fake") & detections["x_common"].between(x0 + 10, x0 + WINDOW - 10)
                          & detections["y_common"].between(y0 + 10, y0 + WINDOW - 10)]
        if near["injection_id"].nunique() > best_count:
            best, best_count = (x0, y0), near["injection_id"].nunique()
    x0, y0 = best

    planted = src / "data" / "interim" / "night_images" / UNIT
    frames = []
    for _, row in exposures.iterrows():
        k, name = int(row["frame_index"]), row["exposure"]
        exposure = load_exposure(src / "data" / "raw" / "ztf" / "2019-02-08" / f"{name}_sciimg.fits", [0, 2, 3, 4, 7, 8, 9, 10, 12])
        science = np.load(planted / f"{name}__science.npy").astype(np.float64)
        difference = np.load(planted / f"{name}__difference.npy").astype(np.float64)
        reference = exposure["reference"]
        # the telescope points slightly differently each time: cut the same patch of SKY, not the same pixels
        in_frame = detections[detections["frame_index"] == k]
        dx = int(round(float((in_frame["x"] - in_frame["x_common"]).median())))
        dy = int(round(float((in_frame["y"] - in_frame["y_common"]).median())))
        cut = (slice(y0 + dy, y0 + dy + WINDOW), slice(x0 + dx, x0 + dx + WINDOW))
        sky = np.nanmedian(science[cut])
        noise = 1.4826 * np.nanmedian(np.abs(difference[cut] - np.nanmedian(difference[cut])))
        for label, image, low, high in [("science", science, sky - 2 * noise, sky + 60 * noise),
                                        ("reference", reference, sky - 2 * noise, sky + 60 * noise),
                                        ("difference", difference, -3 * noise, 25 * noise)]:
            Image.fromarray(stretch(np.nan_to_num(image[cut], nan=low), low, high)).save(out / "scene" / f"f{k + 1}_{label}.png")
        frames.append({"frame": k + 1, "exposure": name, "obsjd": round(float(row["obsjd"]), 5), "noise": round(float(noise), 2)})

    inside = detections[detections["x_common"].between(x0, x0 + WINDOW - 1) & detections["y_common"].between(y0, y0 + WINDOW - 1)]
    normalized[inside.index.to_numpy()].astype(np.float32).tofile(out / "window_stamps.bin")    # same order as scene["blobs"]
    in_tracks = members[members["detection_id"].isin(inside.index)]["tracklet_id"].unique()
    scene_tracks = []
    for tid in in_tracks:
        track = tracklets[tracklets["tracklet_id"] == tid].iloc[0]
        pts = members[members["tracklet_id"] == tid].sort_values("frame_index")
        scene_tracks.append({"id": int(tid), "known_name": track["known_name"], "fake": int(track["injection_id"]) >= 0,
                             "mag": round(float(track["mag_mean"]), 2), "rate": round(float(track["rate_arcsec_per_hour"]), 1),
                             "residual": round(float(track["max_residual"]), 3), "score_mean": round(float(track["score_mean"]), 4),
                             "points": [{"frame": int(p["frame_index"]) + 1, "x": round(float(p["x_common"]) - x0, 2), "y": round(float(p["y_common"]) - y0, 2)} for _, p in pts.iterrows()]})
    scene = {"unit": UNIT, "window": {"x0": x0, "y0": y0, "size": WINDOW}, "frames": frames, "tracks": scene_tracks,
             "blobs": [{"frame": int(r["frame_index"]) + 1, "x": round(float(r["x_common"]) - x0, 2), "y": round(float(r["y_common"]) - y0, 2),
                        "snr": round(float(r["snr"]), 1), "score": round(float(r["score"]), 4), "kind": r["kind"], "name": r["known_asteroid"] or None}
                       for _, r in inside.iterrows()]}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))

    # 5. Whole-night funnel, and the tracklet scores
    X = tracklets.copy()
    kept = detections["score"] >= thresholds["realbogus"]
    night = {
        "unit": UNIT, "frames": len(exposures), "blobs": int(len(detections)),
        "blobs_real": int((detections["label"] == 1).sum()), "kept_by_cnn": int(kept.sum()),
        "kept_real": int((kept & (detections["label"] == 1)).sum()), "real_total": int((detections["label"] == 1).sum()),
        "tracklets": int(len(X)), "tracklets_known": int((X["known_name"] != "").sum()), "tracklets_fake": int((X["injection_id"] >= 0).sum()),
        "fakes_planted": int(pd.read_parquet(processed / f"injections__{UNIT}.parquet")["injection_id"].nunique()),
        "cnn_threshold": thresholds["realbogus"], "tracklet_threshold": thresholds["tracklet"],
        "scoreboard": pd.read_csv(src / "reports" / "scoreboard.csv").to_dict("records"),
        "metrics": detection_metrics(detections["label"].to_numpy(), detections["score"].to_numpy(), thresholds["realbogus"], len(exposures)),
        "feature_importance": sorted(([c, round(float(v), 4)] for c, v in zip(bundle["info"]["feature_columns"], bundle["tracklet_model"].feature_importances_)), key=lambda p: -p[1]),
    }
    (out / "night.json").write_text(json.dumps(night, indent=1, default=float))
    print(json.dumps({k: v for k, v in night.items() if k != "scoreboard"}, indent=1), len(sample), "stamps;", len(scene["blobs"]), "blobs in window;", len(scene_tracks), "tracks")


if __name__ == "__main__":
    main()
