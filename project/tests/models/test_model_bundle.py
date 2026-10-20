import subprocess

import numpy as np
import pytest

from asthunt.models.define_realbogus_cnn import define_realbogus_cnn
from asthunt.models.load_model_bundle import load_model_bundle
from asthunt.models.save_model_bundle import save_model_bundle
from asthunt.models.train_tracklet_classifier import train_tracklet_classifier


def test_save_then_load_gives_the_same_models(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)            # bundles record the git commit
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cnn = define_realbogus_cnn()
    rng = np.random.default_rng(0)
    features, labels = rng.uniform(size=(100, 3)), (np.arange(100) % 2)
    xgb = train_tracklet_classifier(features, labels)

    save_model_bundle(models_dir, "v9", cnn, xgb, {"realbogus": 0.1, "tracklet": 0.9}, {"train_nights": ["n1"]})
    bundle = load_model_bundle(models_dir, "v9")

    assert bundle["thresholds"] == {"realbogus": 0.1, "tracklet": 0.9}
    assert bundle["info"]["version"] == "v9" and bundle["info"]["train_nights"] == ["n1"]
    assert np.allclose(bundle["tracklet_model"].predict_proba(features), xgb.predict_proba(features))
    first = next(iter(cnn.state_dict().values()))
    assert (next(iter(bundle["realbogus_model"].state_dict().values())) == first).all()


def test_bundles_are_never_overwritten(tmp_path):
    (tmp_path / "models" / "v1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        save_model_bundle(tmp_path / "models", "v1", None, None, {}, {})
