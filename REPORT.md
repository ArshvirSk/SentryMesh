# SentryMesh — Training Run Report

CSC701 Machine Learning mini-project · basis for the CSP701 project report.
All numbers below come from an actual training run recorded in
`artifacts/training_result.json`; nothing here is aspirational.

## 1. Dataset

| | |
|---|---|
| Source | Deterministic synthetic IoT-flow generator (offline fallback) |
| Why | TON_IoT `train_test_network.csv` and the CICIoT2023 bundle were not reachable from this machine at run time; the loaders in `sentrymesh/datasets.py` accept both real datasets as drop-in files and prefer them automatically when present |
| Rows | 45,000 flows (12 features) |
| Split | 75% train / 25% stratified test |
| Seed | 42 (fully reproducible) |

Class distribution as recorded (natural imbalance preserved — no oversampling):

| Class | Rows | Share |
|---|---|---|
| Benign | 32,513 | 72.3% |
| DoS | 7,092 | 15.8% |
| Probing | 3,604 | 8.0% |
| Authentication | 1,791 | 4.0% |

Feature groups: flow shape (byte/packet totals, per-packet bytes, byte & packet ratios,
inter-arrival time), response-side behaviour, and protocol code (TCP/UDP/ICMP).
Attack profiles follow real traffic patterns: DoS = floods with near-zero duration,
Probing = many tiny short-lived connections, Authentication = repeated moderate
credential exchanges.

## 2. Comparison table (uniform evaluation, 5-fold stratified CV)

Test set = 11,250 held-out records. Latency = mean over a 500-record batch,
5 repetitions. Size = serialized joblib (compression 3). Kappa = Cohen's kappa,
MCC = Matthews correlation coefficient; MAE/RMSE/RAE/RRSE are the Weka-style error
suite computed from predicted probabilities (RAE/RRSE normalised by the zero-R
class-prior predictor), following the source paper's error analysis.

| Candidate | Module | Accuracy | Macro-F1 | Kappa | MCC | ROC-AUC | Size | Latency (ms/rec) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | M2 | 0.9738 | 0.8917 | 0.9394 | 0.9405 | 0.9912 | 1.5 KB | 0.0041 |
| Decision Tree (CART/Gini) | M2 | 0.9862 | 0.9500 | 0.9688 | 0.9689 | 0.9687 | 15.7 KB | 0.0032 |
| Random Forest (bagging) | M3 | 0.9893 | 0.9600 | 0.9758 | 0.9759 | 0.9982 | 2,571 KB | 0.1195 |
| **Gradient Boosting (XGBoost)** | **M3** | **0.9921** | **0.9706** | **0.9821** | **0.9822** | **0.9991** | **424 KB** | **0.0096** |
| SVM (RBF kernel) | M4 | 0.9746 | 0.9002 | 0.9412 | 0.9423 | 0.9806 | 269 KB | 0.5514 |
| Voting Ensemble (LR+DT+XGB) | M3 | 0.9901 | 0.9629 | 0.9776 | 0.9778 | 0.9976 | 441 KB | 0.0188 |

Error suite of the deployed winner (XGBoost, full features): MAE 0.0048, RMSE 0.0571,
RAE 0.0216, RRSE 0.1713 — an order of magnitude better than the zero-R baseline on RAE.
The SVM's comparatively poor probability calibration (MAE 0.1638 at validation) is why
its error cells dominate the suite even where its hard-label metrics look acceptable.

*(Table matches the run log; see `training_run.log` for the raw training output.)*

## 3. Selection reasoning (the tolerance-band rule)

> Among all candidates whose macro-F1 is within **2 percentage points** of the best,
> select the one with the **lowest inference latency**.

Recorded reasoning:

- Best macro-F1 of any candidate: **0.9706** (Gradient Boosting / XGBoost).
- Tolerance band: ≥ 0.9506 → **3 of 6 candidates qualified**:
  Random Forest (0.9600), Gradient Boosting / XGBoost (0.9706), and the Voting Ensemble
  (0.9629).
- Latency tiebreaker inside the band: XGBoost 0.0096 ms/record vs Voting 0.0188 ms and
  Random Forest 0.1195 ms → **XGBoost wins** (~2× cheaper than the vote, ~12× cheaper
  than the forest, at equal-band quality).
- Excluded by the band: Logistic Regression (0.8917), Decision Tree (0.9500 — just
  0.0006 short), SVM (0.9002).
- Winner: **Gradient Boosting (XGBoost)** — best macro-F1 overall *and* lowest latency
  among the qualified.

This run shows both sides of the rule: the Decision Tree was the cheapest classifier of
all (0.0032 ms) but missed the band by 0.0006 macro-F1, so cheapness alone did not win;
Random Forest and the Voting Ensemble matched the band on quality but lost the latency
tiebreaker. The Voting Ensemble — the source paper's core technique — lands 3rd on
latency among the qualified and 2nd on kappa, confirming the paper's own finding that
combining classifiers buys robustness at extra execution cost. The band exists to reward
light models that are *nearly as accurate* — not to crown the cheapest model regardless
of accuracy.

## 4. PCA full-vs-reduced comparison (Module 6)

| | Full features | PCA-reduced |
|---|---|---|
| Features | 12 | 1 (99.4% variance retained) |
| Macro-F1 | 0.9706 | 0.7280 |
| Latency (ms/rec) | 0.0106 | 0.0092 (−13.2%) |
| Serialized size | 424 KB | 413 KB (−2.6%) |

- With only 12 correlated features, a single principal component already retained 99.4%
  of the variance — but compressing to 1 dimension collapsed macro-F1 from 0.9706 to
  0.7280 (cost 0.2426) for only a 13.2% latency gain and a 2.6% size reduction. The
  reduction is not worth a quarter of the model's F-score. The full 12-component
  cumulative-variance curve is rendered in Model Lab with the 95% target line and the
  chosen k marked.
- Rule applied: reduction is adopted only if macro-F1 cost ≤ 0.01 **and** latency does not
  regress. It failed decisively, so the **full-feature variant was deployed**.
- This matches the requirement that the choice be data-driven, not assumed: PCA would have
  been kept had it paid for itself, and will be re-evaluated on every future retraining run.

## 5. Fog/edge deployment

- Deployed artifact: `artifacts/deployed_model.joblib` (≈424 KB) — pipeline, class map,
  trained-on metadata, and the selection reasoning, all traceable. The Model Lab page can
  deploy any of the **six** candidates (manual override, shown with provenance) or reset
  to the rule's automatic pick; the PCA variant toggle switches the deployed pipeline
  between full_features and pca_reduced.
- Edge engine loads only this winner; per-record inference measured at ~0.01 ms plus
  DBSCAN backstop amortized over every 25th record (standardized rolling window,
  eps=1.8, min_samples=8; anomaly records are surfaced as ANOMALOUS, and the Live
  Detection page renders the window's 2D projection with cluster/anomaly coloring).
- Fog-node resource telemetry (added after the source-paper review): a psutil sampler
  reports CPU % of one core, RSS memory, and thread count on the Overview page. Typical
  live-streaming load: ~15% of one core, ~314–335 MB RSS (the process includes the Flask
  server, dataset, and live charts). The paper reports "<10% CPU" on a dedicated
  3 GB-RAM fog box; on such hardware the same artifact would sit well under that ceiling
  since our per-record inference is ~0.01 ms and the sampler isolates the inference
  process share.
- Cloud/fog data separation (added after the source-paper review, mirroring its
  KDDTrain+ / KDDTest+ design): the cloud trains on the baseline regime while the fog
  stream replays a **separate, drifted regime** (different seed, attack parameters
  perturbed at d = 0.25, benign tail broadened) that the model never saw. The simulator
  carries true labels, so the edge tier scores itself continuously: **live accuracy
  99.3% on the first 141 unseen streamed records** (per-class tracking included),
  against a held-out test accuracy of 99.21% — the deployed model generalizes to the
  drifted stream rather than replaying seen rows. With real datasets in `data/`, the
  pair loader instead uses two disjoint subsamples of the source CSVs. Live accuracy is
  shown on Overview and in the Live Detection tally.
- The traffic simulator replays the dataset as a live gateway stream (50 ms pacing),
  classifies record-by-record, and feeds the Overview / Live Detection / Investigate pages.

## 6. Directory structure

```
SentryMesh/
├── run.py                      # entry point (python run.py)
├── train.py                    # standalone cloud-role training run
├── README.md
├── REPORT.md                   # this file
├── training_run.log            # raw output of the recorded run
├── SentryMesh-ML-PRD.md        # original PRD (background)
├── sentrymesh/
│   ├── __init__.py
│   ├── datasets.py             # TON_IoT / CICIoT2023 / synthetic + subsampling
│   ├── cloud.py                # training, evaluation, selection, PCA, export
│   ├── edge.py                 # deployed-model engine, simulator, DBSCAN backstop
│   └── app.py                  # Flask REST API + static hosting
├── static/
│   ├── index.html              # SPA shell
│   ├── app.js                  # five pages, client-rendered
│   └── style.css               # monochrome design system
├── artifacts/
│   ├── deployed_model.joblib   # winning pipeline (edge loads only this)
│   └── training_result.json    # full run record (tables, reasoning, PCA)
└── data/                       # optional drop-in for real dataset CSVs
```

## 7. Reproducing

```bash
python train.py     # cloud role: retrains everything, rewrites artifacts/*
python run.py       # serves the app on http://127.0.0.1:5000
```

Or trigger the identical pipeline from the UI: **Model Lab → Train Models**.
