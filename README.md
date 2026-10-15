# maio-assignment3-groupK

# Diabetes Progression Triage Service

## Run & API
_Matthias_

## Reproduce training locally
Requirements: Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py              # trains the model
pytest tests/test_train.py       # training tests
```

`python src/train.py` writes two files:

| File | Content |
| --- | --- |
| `models/model.joblib` | Full scikit-learn pipeline (scaler + model), loaded by the API |
| `models/metrics.json` | Model version, model type, RMSE (test + 5-fold CV), high-risk precision/recall |

Options:

- `--out-dir <folder>`: save the files somewhere else (default `models/`)
- `--version <tag>`: version string stored in `metrics.json` (default: env var `MODEL_VERSION`, otherwise `v0.1`)

Training is deterministic: the seed is fixed to 42 and all package versions are pinned, so running the script twice gives identical metrics. `tests/test_train.py` checks this.

## CI/CD
_Tom_
