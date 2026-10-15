# maio-assignment3-groupK
## Run & API
_Mattias_

## Reproduce training locally
_Julian_

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
