"""Train bundle v6: the v4 design with CLASSICAL ML only (no neural network). Run: python scripts/train_classical_bundle.py

Stage 1 (real/bogus for every blob): XGBoost on 29 stamp features + 20 PCA components, instead of the CNN.
Stage 2 (tracklets): the same 17 features and XGBoost as v4, retrained on the new stage-1 scores.
Same nights, same quadrant, same thresholds rule as v4, so the scoreboard compares like with like.
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from asthunt.acquire.list_units import list_units
from asthunt.config.load_config import load_config
from asthunt.dataset.assemble_stamp_dataset import assemble_stamp_dataset
from asthunt.dataset.build_tracklet_stamp_dataset import build_tracklet_stamp_dataset
from asthunt.dataset.make_unit_tag import make_unit_tag
from asthunt.evaluate.choose_threshold import choose_threshold
from asthunt.features.add_path_features import ALL_FEATURE_COLUMNS, add_path_features
from asthunt.models.predict_realbogus_trees import predict_realbogus_trees
from asthunt.models.save_model_bundle import save_model_bundle
from asthunt.models.train_realbogus_trees import XGB_SETTINGS, train_realbogus_trees
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier

project = load_config("project"); paths = project["paths"]; P = paths["processed"]
acquire, link, dataset = load_config("acquire"), load_config("link"), load_config("dataset")
rng = np.random.default_rng(project["seed"])
VERSION, SUFFIX = "v6", "_v6"
log = lambda message: print(time.strftime("%H:%M:%S"), message, flush=True)

# Like v4: the original quadrant only, 8 training and 2 validation nights (configs/acquire.yaml)
train_units = list_units(acquire, "train", include_extra=False)
val_units = list_units(acquire, "validation", include_extra=False)

# ---- Stage 1: the classical real/bogus model ----
train_stamps, train_labels, train_table = assemble_stamp_dataset(train_units, P, rng, dataset["bogus_per_real"])
val_stamps, val_labels, val_table = assemble_stamp_dataset(val_units, P, rng)
model = train_realbogus_trees(train_stamps, train_table, train_labels, seed=project["seed"])
val_scores = predict_realbogus_trees(model, val_stamps, val_table)
log(f"stage 1: {len(train_labels)} training stamps; validation AP {average_precision_score(val_labels, val_scores):.4f}; "
    f"PCA keeps {model['pca_explained_variance']:.0%} of the pixel variance")

# Out-of-fold scores for the training nights (cross-fitting, 4 folds of 2 nights), so the tracklet model
# learns from scores like the ones it will see on new nights
scores = {}
for fold, held_out in enumerate(np.array_split(np.array(sorted({u["night"] for u in train_units})), 4)):
    fit_units = [u for u in train_units if u["night"] not in set(held_out)]
    stamps, labels, table = assemble_stamp_dataset(fit_units, P, rng, dataset["bogus_per_real"])
    fold_model = train_realbogus_trees(stamps, table, labels, seed=project["seed"] + fold)
    for unit in train_units:
        if unit["night"] in set(held_out):
            unit_stamps, _, unit_table = assemble_stamp_dataset([unit], P, rng)
            scores[make_unit_tag(unit)] = predict_realbogus_trees(fold_model, unit_stamps, unit_table)
    log(f"cross-fit fold {fold + 1}/4 done (held out {list(held_out)})")
for unit in val_units:
    unit_stamps, _, unit_table = assemble_stamp_dataset([unit], P, rng)
    scores[make_unit_tag(unit)] = predict_realbogus_trees(model, unit_stamps, unit_table)
for tag, unit_scores in scores.items():
    np.save(P / f"realbogus_scores{SUFFIX}__{tag}.npy", unit_scores.astype(np.float32))

# Same rule as v2/v4: keep 99% of real blobs on the validation nights (linking and stage 2 remove the junk)
realbogus_threshold = choose_threshold(val_labels, val_scores, 0.99)
log(f"stage-1 cut {realbogus_threshold:.4f} (99% recall on validation)")

# ---- Stage 2: link, describe and score tracklets (same 17 features as v4) ----
tables = {}
for split, units in [("train", train_units), ("validation", val_units)]:
    rows = []
    for unit in units:
        tracklets, members, stamps = build_tracklet_stamp_dataset(unit, paths, acquire["bad_mask_bits"], link,
                                                                  realbogus_threshold, 3, version_suffix=SUFFIX)
        detections = pd.read_parquet(P / f"detections__{make_unit_tag(unit)}.parquet")
        detections["score"] = np.load(P / f"realbogus_scores{SUFFIX}__{make_unit_tag(unit)}.npy")
        rows.append(add_path_features(tracklets, members, detections, stamps.astype(np.float32)))
    tables[split] = pd.concat(rows, ignore_index=True)
    log(f"{split}: {len(tables[split])} tracklets, {int(tables[split]['label'].sum())} real")

X = lambda table: np.nan_to_num(table[ALL_FEATURE_COLUMNS].to_numpy())
tracklet_model = train_tracklet_classifier(X(tables["train"]), tables["train"]["label"].to_numpy(), seed=project["seed"])
val_tracklet_scores = tracklet_model.predict_proba(X(tables["validation"]))[:, 1]
log(f"stage 2: validation tracklet AP {average_precision_score(tables['validation']['label'], val_tracklet_scores):.4f}")

thresholds = {"realbogus": float(realbogus_threshold),
              "tracklet": float(choose_threshold(tables["validation"]["label"], val_tracklet_scores, 0.95)), "min_detections": 3}
save_model_bundle(paths["models"], VERSION, model, tracklet_model, thresholds, {
    "description": "v6: classical ML only. Stage 1 = XGBoost on 29 stamp features + 20 PCA components (no CNN); "
                   "stage 2 = the v4 tracklet XGBoost (17 features), retrained on the new stage-1 scores",
    "realbogus_model": "xgboost + pca", "realbogus_settings": XGB_SETTINGS, "feature_columns": ALL_FEATURE_COLUMNS,
    "train_nights": acquire["nights"]["train"], "validation_nights": acquire["nights"]["validation"], "quadrants": [[3, 4]],
    "stage1_validation_ap": float(average_precision_score(val_labels, val_scores))})
log(f"bundle {VERSION} saved: {thresholds}")
