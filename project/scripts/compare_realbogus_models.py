"""Compare classical ML models (and the CNN, for reference) as the stage-1 real/bogus classifier; cluster the junk.

Run: python scripts/compare_realbogus_models.py      (about 2 minutes on a laptop)

Writes reports/realbogus_comparison.csv  one row per model: AP and false alarms at fixed recall, validation and test
       reports/junk_clusters.csv         k-means clusters of the junk blobs: size and what describes each one
       reports/junk_clusters.png         the average stamp of each cluster (science | reference | difference)

Fair-comparison rules (same as the CNN lesson): the same training nights and balanced training set, the realistic
validation nights for every choice (thresholds included), and the test night scored only once, at the end.
"""
import time

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from asthunt.acquire.list_units import list_units  # noqa: E402
from asthunt.config.load_config import load_config  # noqa: E402
from asthunt.dataset.assemble_stamp_dataset import assemble_stamp_dataset  # noqa: E402
from asthunt.evaluate.choose_threshold import choose_threshold  # noqa: E402
from asthunt.features.compute_stamp_features import DETECTION_COLUMNS, FEATURE_COLUMNS, compute_stamp_features  # noqa: E402
from asthunt.models.load_model_bundle import load_model_bundle  # noqa: E402
from asthunt.models.predict_realbogus import predict_realbogus  # noqa: E402
from asthunt.models.predict_realbogus_trees import core_pixels, predict_realbogus_trees  # noqa: E402
from asthunt.models.train_realbogus_trees import train_realbogus_trees  # noqa: E402

project = load_config("project"); paths = project["paths"]; P = paths["processed"]
acquire, dataset = load_config("acquire"), load_config("dataset")
rng = np.random.default_rng(project["seed"])
reports = paths["reports"]; reports.mkdir(exist_ok=True)

sets = {name: assemble_stamp_dataset(list_units(acquire, split, include_extra=False), P, rng,
                                     dataset["bogus_per_real"] if split == "train" else None)
        for name, split in [("train", "train"), ("validation", "validation"), ("test", "test")]}
features = {name: compute_stamp_features(s, t) for name, (s, _, t) in sets.items()}
labels = {name: y for name, (_, y, _) in sets.items()}
frames = {name: t.groupby(["night", "frame_index"]).ngroups for name, (_, _, t) in sets.items()}
print({name: f"{len(y)} blobs, {y.mean():.1%} real" for name, y in labels.items()}, flush=True)


def table(name, columns):
    return features[name][columns].to_numpy()


scores = {}
candidates = {
    "Logistic regression, 6 blob numbers (old baseline)": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), DETECTION_COLUMNS, None),
    "Logistic regression, 29 features": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), FEATURE_COLUMNS, None),
    "SVM (RBF kernel), 29 features": (make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, gamma="scale")), FEATURE_COLUMNS, 20000),
    "Random forest, 29 features": (RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=0), FEATURE_COLUMNS, None),
}
for name, (model, columns, subsample) in candidates.items():
    X, y = table("train", columns), labels["train"]
    if subsample:                          # an RBF SVM grows with n^2: train it on a random 20,000
        pick = np.random.default_rng(0).choice(len(y), subsample, replace=False)
        X, y = X[pick], y[pick]
    model.fit(X, y)
    score = model.decision_function if hasattr(model[-1], "decision_function") and isinstance(model[-1], SVC) else (lambda Z: model.predict_proba(Z)[:, 1])
    scores[name] = {split: score(table(split, columns)) for split in ("validation", "test")}
    print(time.strftime("%H:%M:%S"), name, flush=True)

trees = train_realbogus_trees(sets["train"][0], sets["train"][2], labels["train"], seed=project["seed"])
scores["XGBoost, 29 features + 20 PCA (chosen, bundle v6)"] = {split: predict_realbogus_trees(trees, sets[split][0], sets[split][2])
                                                               for split in ("validation", "test")}
cnn = load_model_bundle(paths["models"], "v4")["realbogus_model"]
scores["CNN (deep learning, comparison only)"] = {split: predict_realbogus(cnn, sets[split][0]) for split in ("validation", "test")}

rows = []
for name, s in scores.items():
    row = {"model": name}
    for split in ("validation", "test"):
        row[f"AP {split}"] = average_precision_score(labels[split], s[split])
    for target in (0.90, 0.95):
        threshold = choose_threshold(labels["validation"], s["validation"], target)      # chosen on validation only
        for split in ("validation", "test"):
            kept, y = s[split] >= threshold, labels[split]
            tp, fp = int((kept & (y == 1)).sum()), int((kept & (y == 0)).sum())
            row[f"recall @{target:.0%} {split}"] = tp / int((y == 1).sum())
            row[f"precision @{target:.0%} {split}"] = tp / max(tp + fp, 1)
            row[f"false alarms per photo @{target:.0%} {split}"] = fp / frames[split]
    rows.append(row)
comparison = pd.DataFrame(rows)
comparison.to_csv(reports / "realbogus_comparison.csv", index=False)
print(comparison[["model", "AP validation", "AP test", "precision @90% validation", "false alarms per photo @90% validation"]].round(3).to_string(index=False))

# ---- Clustering the junk (unsupervised): which kinds of bogus blobs are there? ----
train_stamps, _, train_table = sets["train"]
bogus = labels["train"] == 0
pixels = core_pixels(train_stamps[bogus])
projected = PCA(n_components=20, random_state=0).fit_transform(pixels)
kmeans = KMeans(n_clusters=6, n_init=10, random_state=0).fit(projected)
clusters = features["train"][bogus].assign(cluster=kmeans.labels_)
summary = clusters.groupby("cluster").agg(blobs=("snr", "size"), snr=("snr", "median"), sci_centre=("sci_centre", "median"),
                                          ref_centre=("ref_centre", "median"), diff_centre=("diff_centre", "median"),
                                          diff_min=("diff_min", "median"), diff_sharpness=("diff_sharpness", "median"),
                                          diff_psf_match=("diff_psf_match", "median"))
summary["share"] = summary["blobs"] / summary["blobs"].sum()
summary.to_csv(reports / "junk_clusters.csv")
print(summary.round(2).to_string(), flush=True)

fig, axes = plt.subplots(3, 6, figsize=(12, 6.4))
for k in range(6):
    mean_stamp = train_stamps[bogus][kmeans.labels_ == k].mean(axis=0)
    for channel, label in enumerate(["science", "reference", "difference"]):
        axes[channel, k].imshow(mean_stamp[channel], cmap="gray", vmin=-2, vmax=8, origin="lower")
        axes[channel, k].set_xticks([]); axes[channel, k].set_yticks([])
        if k == 0:
            axes[channel, k].set_ylabel(label)
    axes[0, k].set_title(f"cluster {k}: {summary.loc[k, 'share']:.0%}", fontsize=10)
fig.suptitle("k-means on PCA of junk blobs: the average stamp of each cluster", fontsize=12)
fig.tight_layout()
fig.savefig(reports / "junk_clusters.png", dpi=130)
print("written", reports / "realbogus_comparison.csv", reports / "junk_clusters.csv", reports / "junk_clusters.png")
