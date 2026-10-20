import numpy as np
import pandas as pd

from asthunt.evaluate.measure_injection_realism import measure_injection_realism


def detections_table(fakes_look_different, rng):
    rows = []
    for night in ["n1", "n2", "n3", "n4"]:
        for kind in ["real", "fake"]:
            for _ in range(150):
                width = 1.5 if (kind == "fake" and fakes_look_different) else 1.0
                rows.append({"night": night, "mag": rng.uniform(18, 20), "snr": rng.normal(10, 2),
                             "npix": rng.normal(12, 2), "a": rng.normal(1.2, 0.1) * width, "b": rng.normal(1.0, 0.1),
                             "peak": rng.normal(50, 5), "flux": rng.normal(500, 50),
                             "known_asteroid": "Anius" if kind == "real" else "",
                             "injection_id": -1 if kind == "real" else 1})
    return pd.DataFrame(rows)


def test_identical_populations_give_chance_level():
    auc, _ = measure_injection_realism(detections_table(False, np.random.default_rng(0)), np.arange(18, 20.5, 0.5))
    assert 0.4 < auc < 0.6


def test_a_giveaway_is_detected_and_named():
    auc, importances = measure_injection_realism(detections_table(True, np.random.default_rng(0)), np.arange(18, 20.5, 0.5))
    assert auc > 0.8
    assert importances.index[0] in ("a", "elongation")
