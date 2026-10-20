import numpy as np
import pandas as pd

from asthunt.models.cross_fit_realbogus_scores import cross_fit_realbogus_scores


def write_fake_night(processed_dir, night, rng, n=120):
    labels = (np.arange(n) % 3 == 0).astype(int)
    stamps = rng.normal(0, 1, size=(n, 3, 31, 31))
    stamps[labels == 1, 2, 13:18, 13:18] += 6.0
    pd.DataFrame({"label": labels}).to_parquet(processed_dir / f"detections__000618_c03_q4_{night}.parquet")
    np.save(processed_dir / f"stamps__000618_c03_q4_{night}.npy", stamps.astype(np.float16))
    return stamps, labels


def test_every_training_night_gets_out_of_fold_scores(tmp_path):
    rng = np.random.default_rng(0)
    nights = ["n1", "n2", "n3", "n4"]
    for night in nights:
        write_fake_night(tmp_path, night, rng)
    validation_stamps, validation_labels = write_fake_night(tmp_path, "val", rng)
    settings = {"learning_rate": 0.001, "batch_size": 64, "max_epochs": 2, "patience": 1}

    units = [{"field": 618, "ccd": 3, "quadrant": 4, "night": night} for night in nights]
    scores = cross_fit_realbogus_scores(units, 2, validation_stamps.astype(np.float32), validation_labels,
                                        tmp_path, bogus_per_real=2, settings=settings, device="cpu", seed=0)

    tags = [f"000618_c03_q4_{night}" for night in nights]
    assert sorted(scores) == tags
    assert all(len(scores[tag]) == 120 for tag in tags)
