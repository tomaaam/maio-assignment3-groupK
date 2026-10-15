# maio-assignment3-groupK
## Run & API
_Mattias_

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
### Continuous Integration (`.github/workflows/ci.yml`)

Runs on every pull request and on every push to `main`.

Steps: 
1. Install pinned dependencies (`pip install -r requirements-dev.txt`)
2. Lint with `ruff check .`
3. Train the model as a smoke test (`python src/train.py`)
4. Run unit tests with `pytest`
5. Upload `models/model.joblib` and `models/metrics.json` as the artifact `model-artifacts`

Training runs before the tests because the tests need a trained model.
If any step fails, the run fails and the PR cannot be merged.
Artifacts: Actions -> select a run -> "Artifacts" section.

### Release (`.github/workflows/release.yml`)

Runs when a tag matching `v*` is pushed (e.g. `v0.1`).

Steps: 
1. Build the Docker image, passing the tag as `MODEL_VERSION`
2. Start the container on port `80`
3. Wait until `/health` responds, then smoke-test:
   - `GET /health` returns status "ok" and the `model_version` matching the tag
   - `POST /predict` returns the numeric `prediction`.
   - `POST /predict` with a missing feature returns HTTP code 422`
4. Push the image to GitHub Container Registry (`ghcr.io`)
5. Create a GitHub Release with `metrics.json` and CHANGELOG.md` attached

Images:
- `ghcr.io/tomaaam/maio-assignment3-groupk:v0.1`
- `ghcr.io/tomaaam/maio-assignment3-groupk:v0.2`

### How to cut a release

```bash
git switch main && git pull
git tag v0.1
git push origin v0.1
```
