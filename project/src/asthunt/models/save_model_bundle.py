"""Save everything one pipeline version needs into models/<version>/."""

import json
import subprocess
from datetime import date

import numpy as np
import torch


def save_model_bundle(models_dir, version, realbogus_model, tracklet_model, thresholds, info):
    """Write a self-contained folder and return its path.

    models/<version>/
        realbogus_trees.json      classical real/bogus model (v6 on): XGBoost trees
        realbogus_pca.npz         ... and its PCA (mean and 20 directions)
        realbogus_cnn.pt          or, for v1-v5, the CNN weights
        tracklet_classifier.json  XGBoost model
        thresholds.json           the operating thresholds chosen on validation data
        bundle.json               version, date, git commit, training nights, notes

    One folder = one version of the whole system. To compare versions or to
    roll back, you load a different folder; nothing gets overwritten.
    """
    folder = models_dir / version
    if folder.exists():
        raise FileExistsError(f"{folder} already exists: bundles are never overwritten, pick a new version name")
    folder.mkdir(parents=True)

    if isinstance(realbogus_model, dict):                  # classical: models/train_realbogus_trees.py
        realbogus_model["xgb"].save_model(folder / "realbogus_trees.json")
        np.savez(folder / "realbogus_pca.npz", mean=realbogus_model["pca_mean"], components=realbogus_model["pca_components"])
    else:
        torch.save(realbogus_model.state_dict(), folder / "realbogus_cnn.pt")
    tracklet_model.save_model(folder / "tracklet_classifier.json")
    (folder / "thresholds.json").write_text(json.dumps(thresholds, indent=1))

    git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=models_dir.parent).stdout.strip()
    bundle_info = {"version": version, "created": str(date.today()), "git_commit": git_sha, **info}
    (folder / "bundle.json").write_text(json.dumps(bundle_info, indent=1))
    return folder
