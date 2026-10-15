import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import FEATURES, train  # noqa: E402


def test_train_writes_model_and_metrics(tmp_path):
    metrics = train(tmp_path, "test")
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == metrics
    assert saved["model_version"] == "test"


def test_rmse_is_reasonable(tmp_path):
    # Predicting the mean gives an RMSE of ~73 on this split; a real model must beat that.
    metrics = train(tmp_path, "test")
    assert 0 < metrics["rmse"] < 65


def test_training_is_reproducible(tmp_path):
    first = train(tmp_path / "a", "test")
    second = train(tmp_path / "b", "test")
    assert first == second


def test_model_predicts_from_feature_dataframe(tmp_path):
    train(tmp_path, "test")
    model = joblib.load(tmp_path / "model.joblib")
    row = pd.DataFrame(
        [[0.02, -0.044, 0.06, -0.03, -0.02, 0.03, -0.02, 0.02, 0.02, -0.001]],
        columns=FEATURES,
    )
    pred = model.predict(row)
    assert pred.shape == (1,)
    assert float(pred[0]) == float(pred[0])  # not NaN
