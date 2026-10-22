"""Export real data from the test night for the web app (web/public/data).

The deployed model is v6 (classical ML: XGBoost on 29 stamp features + 20 PCA components). The v4 CNN is exported
too, only so the app can show the deep-learning comparison next to it.

What it writes:
  trees.json                   the v6 real/bogus model: 600 trees, the PCA mean and directions, the cut
  cnn_weights.bin + cnn.json   the v4 CNN (comparison only)
  stamps.bin + stamps.json     48 real 3x31x31 stamps (planted fakes, known asteroids, junk), normalised,
                               with the blob finder's numbers the trees need and both models' scores
  scene/*.png + scene.json     a 160x160 window of the night: 5 frames x (science, reference, difference),
                               every blob in it with its scores, and the v6 tracks that pass through it
  window_stamps.bin            the stamps of those blobs, in the same order
  night.json                   whole-night funnel and metrics (v6, and the CNN for comparison), the scoreboard,
                               the model comparison and the junk clusters
  junk_clusters.png            k-means clusters of junk blobs (from scripts/compare_realbogus_models.py)

Needs the raw data (default ~/Desktop/asteroid/data) and the asthunt environment:
  python tools/export_web_data.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
UNIT = "000618_c03_q4_2019-02-08"
WINDOW = 160
BLOB_COLUMNS = ["snr", "peak", "npix", "a", "b", "flux"]


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


def export_trees(model, threshold, feature_names):
    """The XGBoost trees as flat arrays: for each node the feature index (-1 = leaf), the split value, the children
    (yes = value < split, no, missing) and the leaf value. A tree's output is the leaf the stamp ends in; the score is
    sigmoid(base_margin + sum of all trees' leaves)."""
    booster = model["xgb"].get_booster()
    base_score = float(json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
    trees = []
    for dump in booster.get_dump(dump_format="json"):
        nodes = {}

        def walk(node):
            nodes[node["nodeid"]] = node
            for child in node.get("children", []):
                walk(child)
        walk(json.loads(dump))
        size = max(nodes) + 1
        f, t, yes, no, missing, leaf = ([-1] * size, [0.0] * size, [0] * size, [0] * size, [0] * size, [0.0] * size)
        for i, node in nodes.items():
            if "leaf" in node:
                leaf[i] = node["leaf"]
            else:
                f[i], t[i] = int(node["split"][1:]), node["split_condition"]
                yes[i], no[i], missing[i] = node["yes"], node["no"], node["missing"]
        trees.append({"f": f, "t": t, "y": yes, "n": no, "m": missing, "v": leaf})
    return {"version": "v6", "threshold": float(threshold), "base_margin": float(np.log(base_score / (1 - base_score))),
            "feature_names": feature_names, "pca_mean": model["pca_mean"].round(6).tolist(),
            "pca_components": model["pca_components"].round(6).tolist(), "trees": trees}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "Desktop" / "asteroid" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "public" / "data")
    args = parser.parse_args()
    sys.path.insert(0, str(PROJECT / "src"))
    from asthunt.clean.load_exposure import load_exposure
    from asthunt.clean.load_night_images import load_night_images
    from asthunt.detect.cut_tracklet_stamps import cut_tracklet_stamps
    from asthunt.features.add_path_features import add_path_features
    from asthunt.crossmatch.crossmatch_known_asteroids import crossmatch_known_asteroids
    from asthunt.features.compute_tracklet_features import compute_tracklet_features
    from asthunt.label.load_known_asteroids import load_known_asteroids
    from asthunt.config.load_config import load_config
    from asthunt.dataset.normalize_stamps import normalize_stamps
    from asthunt.features.compute_stamp_features import FEATURE_COLUMNS
    from asthunt.label.label_tracklets import label_tracklets
    from asthunt.link.link_detections_into_tracklets import link_detections_into_tracklets
    from asthunt.models.load_model_bundle import load_model_bundle
    from asthunt.models.predict_realbogus import predict_realbogus

    out, data = args.out, args.data
    (out / "scene").mkdir(parents=True, exist_ok=True)
    processed = data / "processed"
    v6, v4 = load_model_bundle(PROJECT / "models", "v6"), load_model_bundle(PROJECT / "models", "v4")
    trees, thresholds = v6["realbogus_model"], v6["thresholds"]
    cnn, cnn_cut = v4["realbogus_model"], v4["thresholds"]["realbogus"]
    link = load_config("link")

    # 1. The deployed classical model, and the CNN for comparison
    feature_names = FEATURE_COLUMNS + [f"pca_{k + 1}" for k in range(trees["pca_components"].shape[0])]
    (out / "trees.json").write_text(json.dumps(export_trees(trees, thresholds["realbogus"], feature_names), separators=(",", ":")))
    layers, blobs, offset = [], [], 0
    for name, tensor in cnn.state_dict().items():
        values = tensor.detach().cpu().numpy().astype(np.float32).ravel()
        layers.append({"name": name, "shape": list(tensor.shape), "offset": offset, "length": int(values.size)})
        blobs.append(values)
        offset += values.size
    np.concatenate(blobs).tofile(out / "cnn_weights.bin")
    (out / "cnn.json").write_text(json.dumps({"version": "v4", "parameters": offset, "threshold": cnn_cut, "layers": layers}, indent=1))

    # 2. Every blob of the night, scored by both models
    detections = pd.read_parquet(processed / f"detections__{UNIT}.parquet")
    normalized = normalize_stamps(np.asarray(np.load(processed / f"stamps__{UNIT}.npy", mmap_mode="r"), dtype=np.float32))
    detections["score"] = predict_realbogus(trees, normalized, detections=detections)
    detections["cnn_score"] = predict_realbogus(cnn, normalized)
    detections["kind"] = np.where(detections["injection_id"] >= 0, "fake", np.where(detections["known_asteroid"] != "", "known", "bogus"))

    def blob_numbers(row):
        return {c: float(row[c]) for c in BLOB_COLUMNS}

    rng = np.random.default_rng(7)
    picks = []
    fakes = detections[detections["kind"] == "fake"].sort_values("mag")
    picks += list(fakes.iloc[np.linspace(0, len(fakes) - 1, 14).astype(int)].index)
    known = detections[detections["kind"] == "known"]
    picks += list(known.drop_duplicates("known_asteroid").sample(min(10, known["known_asteroid"].nunique()), random_state=1).index)
    bogus = detections[detections["kind"] == "bogus"]
    picks += list(bogus.nlargest(8, "snr").index)                    # bright junk: star residuals, spikes
    picks += list(bogus.nlargest(6, "score").index)                  # junk the model finds hardest
    picks += list(rng.choice(bogus.index, 10, replace=False))        # typical junk: noise
    picks = list(dict.fromkeys(picks))
    sample = detections.loc[picks]
    normalized[picks].astype(np.float32).tofile(out / "stamps.bin")
    (out / "stamps.json").write_text(json.dumps([
        {"i": n, "kind": row["kind"], "label": int(row["label"]), "mag": round(float(row["mag"]), 2),
         "frame": int(row["frame_index"]) + 1, "name": row["known_asteroid"] or None, **blob_numbers(row),
         "score_python": round(float(row["score"]), 6), "cnn_score": round(float(row["cnn_score"]), 6)}
        for n, (_, row) in enumerate(sample.iterrows())], indent=1))

    # 3. Link the blobs v6 keeps into tracks (the deployed settings: 3 of 5 photos, on the common sky grid)
    kept = detections[detections["score"] >= thresholds["realbogus"]]
    kept = kept.assign(x=kept["x_common"], y=kept["y_common"])
    tracklets, members = link_detections_into_tracklets(
        kept, pixel_scale=link["pixel_scale_arcsec"], min_rate=link["min_rate_arcsec_per_hour"],
        max_rate=link["max_rate_arcsec_per_hour"], match_radius=link["match_radius_pixels"],
        min_detections=thresholds["min_detections"], max_residual=link["max_residual_pixels"])
    # ... and score every track with the deployed track model, the same steps as pipeline/run_night_pipeline.py
    tracklets = compute_tracklet_features(tracklets, members, detections)
    frame_list = list(detections.groupby("exposure")["obsjd"].first().sort_values().items())
    unit = {"field": 618, "ccd": 3, "quadrant": 4, "night": "2019-02-08"}
    images = load_night_images(unit, [name for name, _ in frame_list], {"raw": data / "raw", "interim": data / "interim"},
                               [0, 2, 3, 4, 7, 8, 9, 10, 12], with_injections=True)
    path_stamps = cut_tracklet_stamps(tracklets, frame_list, images[frame_list[0][0]]["wcs"], images)
    path_stamps = normalize_stamps(path_stamps.reshape(-1, *path_stamps.shape[2:])).reshape(path_stamps.shape)
    del images
    tracklets = add_path_features(tracklets, members, detections, path_stamps)
    tracklets["tracklet_score"] = v6["tracklet_model"].predict_proba(tracklets[v6["info"]["feature_columns"]].to_numpy())[:, 1]
    tracklets = label_tracklets(tracklets, members, detections)
    known = {exposure: load_known_asteroids(exposure, data / "interim") for exposure in detections["exposure"].unique()}
    tracklets = crossmatch_known_asteroids(tracklets, members, detections, known)       # ALL catalogued asteroids, any brightness
    members = members.join(detections[["x_common", "y_common", "frame_index", "known_asteroid", "injection_id", "score", "mag"]], on="detection_id")
    tracklets["has_fake"] = tracklets["tracklet_id"].map(members.groupby("tracklet_id")["injection_id"].max() >= 0)

    # 4. A window around one clear known asteroid track with planted fakes nearby
    exposures = detections.drop_duplicates("frame_index").sort_values("frame_index")[["frame_index", "exposure", "obsjd"]]
    raw = data / "raw" / "ztf" / "2019-02-08"
    first_reference = load_exposure(raw / f"{exposures.iloc[0]['exposure']}_sciimg.fits", [0, 2, 3, 4, 7, 8, 9, 10, 12])["reference"]
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

    planted = data / "interim" / "night_images" / UNIT
    frames = []
    for _, row in exposures.iterrows():
        k, name = int(row["frame_index"]), row["exposure"]
        reference = load_exposure(raw / f"{name}_sciimg.fits", [0, 2, 3, 4, 7, 8, 9, 10, 12])["reference"]
        science = np.load(planted / f"{name}__science.npy").astype(np.float64)
        difference = np.load(planted / f"{name}__difference.npy").astype(np.float64)
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
    scene_tracks = []
    for tid in members[members["detection_id"].isin(inside.index)]["tracklet_id"].unique():
        track = tracklets[tracklets["tracklet_id"] == tid].iloc[0]
        pts = members[members["tracklet_id"] == tid].sort_values("frame_index")
        scene_tracks.append({"id": int(tid), "known_name": track["known_name"], "fake": bool(track["has_fake"]),
                             "mag": round(float(track["mag_mean"]), 2), "rate": round(float(track["rate_arcsec_per_hour"]), 1),
                             "residual": round(float(track["max_residual"]), 3), "score_mean": round(float(track["score_mean"]), 4),
                             "track_score": round(float(track["tracklet_score"]), 4), "accepted": bool(track["tracklet_score"] >= thresholds["tracklet"]),
                             "points": [{"frame": int(p["frame_index"]) + 1, "x": round(float(p["x_common"]) - x0, 2),
                                         "y": round(float(p["y_common"]) - y0, 2)} for _, p in pts.iterrows()]})
    scene = {"unit": UNIT, "window": {"x0": x0, "y0": y0, "size": WINDOW}, "frames": frames, "tracks": scene_tracks,
             "blobs": [{"frame": int(r["frame_index"]) + 1, "x": round(float(r["x_common"]) - x0, 2), "y": round(float(r["y_common"]) - y0, 2),
                        **blob_numbers(r), "score": round(float(r["score"]), 6), "cnn_score": round(float(r["cnn_score"]), 6),
                        "kind": r["kind"], "name": r["known_asteroid"] or None}
                       for _, r in inside.iterrows()]}
    (out / "scene.json").write_text(json.dumps(scene, indent=1))

    # 5. Whole night: funnel, metrics of both models, scoreboard, model comparison, junk clusters
    labels, n_frames = detections["label"].to_numpy(), len(exposures)
    accepted = tracklets["tracklet_score"] >= thresholds["tracklet"]
    keep = detections["score"] >= thresholds["realbogus"]
    reports = PROJECT / "reports"
    comparison = pd.read_csv(reports / "realbogus_comparison.csv")
    clusters = pd.read_csv(reports / "junk_clusters.csv")
    shutil.copy(reports / "junk_clusters.png", out / "junk_clusters.png")
    night = {
        "unit": UNIT, "frames": n_frames, "blobs": int(len(detections)), "blobs_real": int(labels.sum()),
        "kept_by_cnn": int(keep.sum()), "kept_real": int((keep & (labels == 1)).sum()), "real_total": int(labels.sum()),
        "tracklets": int(len(tracklets)), "tracklets_known": int((tracklets["known_name"] != "").sum()),
        "tracklets_fake": int(tracklets["has_fake"].sum()),
        "accepted": int(accepted.sum()), "accepted_known": int((accepted & (tracklets["known_name"] != "")).sum()),
        "accepted_fake": int((accepted & tracklets["has_fake"]).sum()),
        "accepted_unexplained": int((accepted & (tracklets["known_name"] == "") & ~tracklets["has_fake"]).sum()),
        "fakes_planted": int(pd.read_parquet(processed / f"injections__{UNIT}.parquet")["injection_id"].nunique()),
        "cnn_threshold": thresholds["realbogus"], "tracklet_threshold": thresholds["tracklet"],
        "scoreboard": pd.read_csv(reports / "scoreboard.csv").to_dict("records"),
        "metrics": detection_metrics(labels, detections["score"].to_numpy(), thresholds["realbogus"], n_frames),
        "cnn_metrics": detection_metrics(labels, detections["cnn_score"].to_numpy(), cnn_cut, n_frames),
        "comparison": comparison.to_dict("records"),
        "junk_clusters": clusters.to_dict("records"),
        "stage1_importance": sorted(([n, round(float(v), 4)] for n, v in zip(feature_names, trees["xgb"].feature_importances_)), key=lambda p: -p[1]),
        "feature_importance": sorted(([c, round(float(v), 4)] for c, v in zip(v6["info"]["feature_columns"], v6["tracklet_model"].feature_importances_)), key=lambda p: -p[1]),
    }
    (out / "night.json").write_text(json.dumps(night, indent=1, default=float))
    print(json.dumps({k: v for k, v in night.items() if k in ("blobs", "kept_by_cnn", "kept_real", "real_total", "tracklets", "tracklets_known", "tracklets_fake")}),
          f"v6 AP {night['metrics']['average_precision']:.3f}, CNN AP {night['cnn_metrics']['average_precision']:.3f};",
          {k: night[k] for k in ("accepted", "accepted_known", "accepted_fake", "accepted_unexplained")},
          len(sample), "stamps;", len(scene["blobs"]), "blobs in window;", [(t["known_name"] or t["id"], t["track_score"]) for t in scene_tracks])


if __name__ == "__main__":
    main()
