# Changelog

All metrics come from `python src/train.py` (seed 42, 80/20 train/test split of the scikit-learn diabetes dataset, 353 train / 89 test rows). They are saved to `models/metrics.json` and attached to every GitHub Release.

## v0.1 – Baseline

- Model: `StandardScaler` + `LinearRegression` in one scikit-learn `Pipeline`, so the API uses exactly the same preprocessing as training.
- Reproducibility: fixed seed (42), pinned package versions in `requirements.txt`.
- Metrics:

| Metric | v0.1 |
| --- | --- |
| RMSE (held-out test set) | 53.85 |
| RMSE (5-fold CV on training set) | 55.39 |
| High-risk flag precision | 1.00 |
| High-risk flag recall | 0.29 |

"High risk" = progression index in the top 25 % of the training data (≥ 214). In v0.1 a patient is flagged when the predicted score is ≥ 214. The flag is very conservative: it only catches 6 of 21 high-risk test patients.
