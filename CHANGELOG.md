# Changelog

All metrics come from `python src/train.py` (seed 42, 80/20 train/test split of the scikit-learn diabetes dataset, 353 train / 89 test rows). They are saved to `models/metrics.json` and attached to every GitHub Release.

"High risk" = progression index in the top 25 % of the training data (≥ 214). The test set contains 21 high-risk patients.

## v0.2 – Regularised model + calibrated high-risk flag

### What changed

- **Model:** `LinearRegression` → `RidgeCV` (L2 regularisation). The strength alpha is chosen by cross-validation on the training data only (selected: alpha = 1.15).
- **High-risk flag calibration:** v0.1 flagged a patient when the predicted score was ≥ 214, the same cut-off as the label. Regression predictions are pulled towards the mean, so that rule missed most high-risk patients. v0.2 picks the flag threshold on out-of-fold predictions of the training set (5-fold CV, maximising F1). The test set is never used for this choice. Selected threshold: 186.5.
- **API response unchanged:** `/predict` still returns the continuous score. The flag threshold is stored in `metrics.json` for the dashboard.

### Metrics side by side

| Metric | v0.1 | v0.2 | Change |
| --- | --- | --- | --- |
| RMSE (held-out test set) | 53.85 | 53.77 | −0.08 |
| RMSE (5-fold CV on training set) | 55.39 | 55.35 | −0.05 |
| High-risk flag threshold | 214.0 | 186.5 | calibrated |
| Precision at threshold | 1.00 (6/6) | 0.75 (15/20) | −0.25 |
| Recall at threshold | 0.29 (6/21) | 0.71 (15/21) | **+0.43** |
| F1 at threshold | 0.44 | 0.73 | **+0.29** |

### Why v0.2 is an improvement

- **For triage, recall matters most.** A missed high-risk patient gets no follow-up call. v0.2 finds 15 of 21 high-risk patients instead of 6, at the cost of 5 extra calls for patients who turn out to be lower risk.
- **Score accuracy stays the same or slightly better.** RMSE improves marginally on both the test set and in cross-validation, so the ranking on the nurse dashboard is not worse.
- **Regularisation makes the model more robust.** Ridge shrinks unstable coefficients of correlated lab values (s1–s6), with no extra cost in latency or image size.

### Alternatives tried and rejected

| Model | RMSE (5-fold CV) | RMSE (test) | Decision |
| --- | --- | --- | --- |
| LinearRegression (v0.1) | 55.39 | 53.85 | baseline |
| **RidgeCV (v0.2)** | **55.35** | **53.77** | chosen |
| RandomForest (300 trees) | 58.68 | 54.92 | rejected: worse in CV, larger model |
| RandomForest (tuned: min_samples_leaf=5, max_features=0.5) | 57.27 | 52.92 | rejected: worse in CV; the test gain is noise on 89 rows |

Models were compared on the cross-validated RMSE, not on the test set, to avoid choosing a model that only fits this particular split.

## v0.1 – Baseline

- Model: `StandardScaler` + `LinearRegression` in one scikit-learn `Pipeline`, so the API uses exactly the same preprocessing as training.
- Reproducibility: fixed seed (42), pinned package versions in `requirements.txt`.
- Metrics: RMSE 53.85 (test), 55.39 (5-fold CV); high-risk flag precision 1.00, recall 0.29 (see table above).
