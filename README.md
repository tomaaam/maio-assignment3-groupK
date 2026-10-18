# maio-assignment3-groupK

# Diabetes Progression Triage Service

## Run & API


### Run with Docker
```bash

docker build -t diabetes-api .

docker run -d --name diabetes-api-container -p 8000:80 diabetes-api

```
API: `http://localhost:8000`  

Docs: `http://localhost:8000/docs`

### Health Check

`GET /health`

```json

{

  "status": "ok",

  "model_version": "v0.1"

}

```

### Predict

`POST /predict`

Example request:

```json

{

  "age": 0.02,

  "sex": -0.044,

  "bmi": 0.06,

  "bp": -0.03,

  "s1": -0.02,

  "s2": 0.03,

  "s3": -0.02,

  "s4": 0.02,

  "s5": 0.02,

  "s6": -0.001

}

```

Example response:

```json

{

  "prediction": 235.95

}

```

Missing required fields return HTTP `422` with a JSON validation error.

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
