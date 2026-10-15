"""Train the diabetes progression model and save it with its metrics.

Usage:
    python src/train.py                      # trains the default model
    python src/train.py --out-dir models     # choose output folder

Outputs (contract with API/Docker/CI):
    models/model.joblib   - full sklearn Pipeline (preprocessing + model)
    models/metrics.json   - version, model type, RMSE and flag metrics
"""

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.2
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
# A patient counts as "high risk" if the progression index is in the top 25%
# of the training data. Used only for the extra precision/recall metrics.
HIGH_RISK_QUANTILE = 0.75


def set_seeds(seed: int = SEED) -> None:
    """Fix all random number generators so every run gives the same result."""
    random.seed(seed)
    np.random.seed(seed)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn diabetes dataset as a DataFrame."""
    data = load_diabetes(as_frame=True)
    X = data.frame[FEATURES]
    y = data.frame["target"]
    return X, y


def build_model() -> Pipeline:
    """v0.1 baseline: StandardScaler + LinearRegression in one pipeline.

    Keeping scaling inside the pipeline means the API applies exactly the
    same preprocessing as training (no training-serving skew).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def choose_flag_threshold(
    model: Pipeline, X_train, y_train, label_threshold: float
) -> float:
    """v0.1: flag a patient when the predicted score is above the label threshold."""
    return label_threshold


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_threshold: float,
    flag_threshold: float,
) -> dict:
    """Compute RMSE plus precision/recall of the high-risk flag.

    True high risk:      actual progression    >= label_threshold
    Predicted high risk: predicted progression >= flag_threshold
    """
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    true_flag = y_test >= label_threshold
    pred_flag = pred >= flag_threshold
    return {
        "rmse": round(rmse, 4),
        "high_risk_label_threshold": round(float(label_threshold), 2),
        "high_risk_flag_threshold": round(float(flag_threshold), 2),
        "precision_at_threshold": round(
            float(precision_score(true_flag, pred_flag, zero_division=0)), 4
        ),
        "recall_at_threshold": round(
            float(recall_score(true_flag, pred_flag, zero_division=0)), 4
        ),
    }


def train(out_dir: Path, version: str) -> dict:
    """Train, evaluate and save the model. Returns the metrics dict."""
    set_seeds()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = build_model()
    # 5-fold cross-validated RMSE on the training data: a more stable estimate
    # than the single held-out split, used to compare model versions fairly.
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rmse = -cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
    ).mean()
    model.fit(X_train, y_train)

    label_threshold = float(np.quantile(y_train, HIGH_RISK_QUANTILE))
    flag_threshold = choose_flag_threshold(model, X_train, y_train, label_threshold)
    metrics = {
        "model_version": version,
        "model_type": type(model.named_steps["model"]).__name__,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "seed": SEED,
        "cv_rmse": round(float(cv_rmse), 4),
        **evaluate(model, X_test, y_test, label_threshold, flag_threshold),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--version", default=os.getenv("MODEL_VERSION", "v0.1"))
    args = parser.parse_args()

    metrics = train(args.out_dir, args.version)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
