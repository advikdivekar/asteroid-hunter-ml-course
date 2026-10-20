"""more_data phase: train bundle v5 (the v4 design) on every training unit. Run: python scripts/train_more_data_bundle.py (~35 min on an M3 Pro)."""
import json, time, numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
from asthunt.config.load_config import load_config
from asthunt.acquire.list_units import list_units
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.dataset.assemble_stamp_dataset import assemble_stamp_dataset
from asthunt.dataset.build_tracklet_stamp_dataset import build_tracklet_stamp_dataset
from asthunt.evaluate.choose_threshold import choose_threshold
from asthunt.features.add_path_features import add_path_features, ALL_FEATURE_COLUMNS
from asthunt.models.train_realbogus_cnn import train_realbogus_cnn
from asthunt.models.predict_realbogus import predict_realbogus
from asthunt.models.cross_fit_realbogus_scores import cross_fit_realbogus_scores
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier
from asthunt.models.save_model_bundle import save_model_bundle

project = load_config("project"); paths = project["paths"]; P = paths["processed"]
acq, link, dataset, settings = load_config("acquire"), load_config("link"), load_config("dataset"), load_config("train")
rng = np.random.default_rng(project["seed"]); device = "mps"
train_units, val_units = list_units(acq, "train"), list_units(acq, "validation")
others = list_units(acq, "test") + list_units(acq, "holdout")
log = lambda m: print(time.strftime("%H:%M"), m, flush=True)
log(f"{len(train_units)} training units, {len(val_units)} validation units")

train_stamps, train_labels, _ = assemble_stamp_dataset(train_units, P, rng, dataset["bogus_per_real"])
val_stamps, val_labels, val_table = assemble_stamp_dataset(val_units, P, rng)
log(f"stage-1 data: train {len(train_labels)} stamps ({train_labels.mean():.1%} real), validation {len(val_labels)}")
cnn, history = train_realbogus_cnn(train_stamps, train_labels, val_stamps, val_labels, settings, device, project["seed"])
log(f"stage-1 CNN: best validation AP {max(h['validation_ap'] for h in history):.4f}")
del train_stamps

scores = cross_fit_realbogus_scores(train_units, 4, val_stamps, val_labels, P, dataset["bogus_per_real"], settings, device, project["seed"])
for unit in val_units + others:
    stamps, _, _ = assemble_stamp_dataset([unit], P, rng)
    scores[make_unit_tag(unit)] = predict_realbogus(cnn, stamps, device)
for tag, s in scores.items():
    np.save(P / f"realbogus_scores_v5__{tag}.npy", s)
log(f"stage-1 scores saved for {len(scores)} units")

val_scores = np.concatenate([np.load(P / f"realbogus_scores_v5__{make_unit_tag(u)}.npy") for u in val_units])
realbogus_threshold = choose_threshold(val_labels, val_scores, 0.99)
tables = {}
for split, units in [("train", train_units), ("validation", val_units)]:
    rows = []
    for unit in units:
        t, m, s = build_tracklet_stamp_dataset(unit, paths, acq["bad_mask_bits"], link, realbogus_threshold, 3, version_suffix="_v5")
        d = pd.read_parquet(P / f"detections__{make_unit_tag(unit)}.parquet")
        d["score"] = np.load(P / f"realbogus_scores_v5__{make_unit_tag(unit)}.npy")
        rows.append(add_path_features(t, m, d, s.astype(np.float32)))
    tables[split] = pd.concat(rows, ignore_index=True)
    log(f"{split}: {len(tables[split])} tracklets, {int(tables[split]['label'].sum())} real")

X = lambda t: np.nan_to_num(t[ALL_FEATURE_COLUMNS].to_numpy())
xgb = train_tracklet_classifier(X(tables["train"]), tables["train"]["label"].to_numpy(), seed=project["seed"])
val_tracklet_scores = xgb.predict_proba(X(tables["validation"]))[:, 1]
log(f"tracklet classifier: validation AP {average_precision_score(tables['validation']['label'], val_tracklet_scores):.4f}")
thresholds = {"realbogus": float(realbogus_threshold),
              "tracklet": float(choose_threshold(tables["validation"]["label"], val_tracklet_scores, 0.95)), "min_detections": 3}
save_model_bundle(paths["models"], "v5", cnn, xgb, thresholds, {
    "description": "v5: v4 design retrained on all training units (3 CCD quadrants x 8 nights + field 567 near opposition)",
    "feature_columns": ALL_FEATURE_COLUMNS, "train_units": [make_unit_tag(u) for u in train_units],
    "validation_units": [make_unit_tag(u) for u in val_units], "stage1_history": history})
log(f"bundle v5 saved: {thresholds}")
print("V5 DONE", flush=True)
