"""Load one saved pipeline version from models/<version>/."""

import json

import torch
from xgboost import XGBClassifier

from asthunt.models.define_realbogus_cnn import define_realbogus_cnn


def load_model_bundle(models_dir, version):
    """Return a dict: realbogus_model, tracklet_model, thresholds, info.

    thresholds holds every operating setting of the version: "realbogus" and
    "tracklet" score cuts, and optionally "min_detections" for linking.
    """
    folder = models_dir / version
    if not folder.exists():
        raise FileNotFoundError(f"No model bundle '{version}' in {models_dir}")

    realbogus_model = define_realbogus_cnn()
    realbogus_model.load_state_dict(torch.load(folder / "realbogus_cnn.pt"))
    realbogus_model.eval()

    tracklet_model = XGBClassifier()
    tracklet_model.load_model(folder / "tracklet_classifier.json")

    return {
        "realbogus_model": realbogus_model,
        "tracklet_model": tracklet_model,
        "thresholds": json.loads((folder / "thresholds.json").read_text()),
        "info": json.loads((folder / "bundle.json").read_text()),
    }
