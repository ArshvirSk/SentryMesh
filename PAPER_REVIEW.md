# SentryMesh vs. the Source Paper — Review & Recommended Changes

Reference: Tomer, V.; Sharma, S. "Detecting IoT Attacks Using an Ensemble Machine Learning
Model." *Future Internet* 2022, 14, 102 (local copy: `futureinternet-14-00102.pdf`,
extracted text: `paper_text.txt`). This is the paper the original SentryMesh PRD was based
on; this document compares the finished project against it and lists concrete, prioritized
improvements.

---

## 1. Where SentryMesh already goes beyond the paper

| Dimension | Paper | SentryMesh |
|---|---|---|
| Dataset | NSL-KDD (1999-era traffic, 41 features) | TON_IoT / CICIoT2023 loaders (IoT-era flows); synthetic fallback for reproducibility |
| Model search space | 10 hardcoded pairs of base classifiers × {voting, stacking} = 20 runs | 5 architecturally diverse candidates incl. a modern boosting library (XGBoost) |
| Selection criterion | lowest execution time among near-equal performers (implicit) | **explicit, interactive tolerance-band rule** with a user-adjustable band + 3 tiebreakers, live in the UI |
| Deployment economics | execution time + "<10% CPU" on the fog node | per-record latency (batch-measured) **and** serialized model size — both portable to a real gateway |
| Real-time validation | re-run metrics on the held-out test file | continuous live stream with per-record latency, confidence, DBSCAN backstop, device forensics |
| Unsupervised tier | none | DBSCAN rolling-window anomaly backstop for zero-day traffic |
| Dimensionality reduction | none ("no additional feature extraction... is used") | PCA full-vs-reduced evaluation with a data-driven deployment decision |
| Multiclass framing | NSL-KDD's 4 classes via Weka | native 4-class (Benign/DoS/Probing/Authentication) with per-class P/R/F1 |

Two claims in the paper are worth adopting *verbatim* in the project report because
SentryMesh's data supports them even better: (a) ensemble/boosting methods hit the highest
detection quality at modest footprint (their ref [27]: XGBoost ≈ 99.6 AUC — our XGBoost
candidate: ROC-AUC 0.9991); (b) the fog node's job must stay light — our deployed artifact
is 424 KB with ~0.01 ms per-record inference.

---

## 2. Gaps the paper exposes — recommended changes

### Priority 1 — cheap, high-value, do before the report

**P1-1. Add the paper's headline metrics: Cohen's kappa and MCC.**
The paper's central evidence is kappa / F-measure / ROC / MCC per model (Figs. 6, 8, 9).
SentryMesh computes macro-F1 and ROC-AUC but not kappa or Matthews correlation coefficient,
both of which are *more* informative than accuracy under our imbalanced classes. Both are
one-liners in the eval harness (`cohen_kappa_score`, `matthews_corrcoef` — verified working
multiclass) and slot naturally into the comparison table and the scatter plot's tooltip.

**P1-2. Report the error suite the paper uses (MAE / RMSE / RAE / RRSE).**
Figs. 7 and 10 build the paper's "errors associated" analysis from four error metrics.
These need only probability outputs, which every candidate already produces (`predict_scores`).
Adding them gives the Model Lab a fourth cost column family and mirrors the paper's tables
when the CSP701 report is written.

**P1-3. Fog-node resource instrumentation (CPU %).**
The paper's fog-tier claim is "<10% CPU" (Section 5.2.3). SentryMesh measures latency but
not CPU or memory. A 5-line sampler (e.g. `psutil.Process().cpu_percent()` sampled per
window in the edge engine) surfaced on Overview next to "Avg edge latency" turns our
"the edge is light" argument from asserted to measured — and gives the report a direct
comparison row against the paper.

**P1-4. A single true "voting ensemble" candidate.**
The paper's whole method is *voting/stacking over base classifiers*; our candidates are
individual model families. The syllabus (Module 3: "ways to combine classifiers") and the
paper both point at the same missing artifact. One `VotingClassifier(soft)` over
{LR, DT, XGBoost} — three already-trained factories — costs one more row in the sweep,
completes the story, and likely wins or ties the band (at which point the tolerance rule
has a genuinely interesting decision to make, since a 3-model vote costs ~3× latency).

### Priority 2 — worthwhile, medium effort

**P2-1. Cloud/fog data separation like the paper (train file vs. live file).**
The paper trains on NSL-KDD's train file and validates on the *separate* KDDTest+ file —
a genuinely unseen distribution. SentryMesh currently derives both splits from one
subsample, so the "fog" stream replays rows drawn from the same distribution the cloud
trained on (that's why live accuracy looks so clean). Even without a second real dataset,
splitting the synthetic generator into two seeds/regimes (train on regime A, stream
regime B with slightly drifted parameters) would make the live tier a real generalization
test instead of a replay. This is the paper's most under-appreciated methodological point.

**P2-2. Kappa/MCC in the tolerance rule's tiebreaker set.**
Once P1-1 lands, "highest kappa" is a natural 4th tiebreaker option (the paper implicitly
uses a composite "isResultBetter(r, time)" — our interactive rule makes that tradeoff
*visible*, which is the better version of the same idea).

**P2-3. Drift indicator on Overview.**
The paper re-runs the full sweep to refresh the model. A lightweight "distribution drift"
signal — e.g. population-stability index of the live window vs. training feature stats,
rendered as a single number/badge — tells the user *when* retraining is warranted, closing
the MLOps loop the Train button only half-finishes.

### Priority 3 — larger, optional

**P3-1. CPU-constrained fog simulation.** The paper runs on a 3 GB RAM single-core box.
We could cap the edge worker (`taskset`/`cgroup`-style limits, or simply document a
Raspberry Pi deployment of `edge.py` + artifact) to reproduce their constrained-hardware
claim in kind.

**P3-2. Real testbed capture.** The paper's future work is exactly this (Fed4Fire, COSMOS,
POWDER). Out of scope for CSC701 but a good "future work" paragraph.

**P3-3. XGBoost hyperparameter sweep in the cloud tier.** The paper searches combinations;
we search families. A small grid (depth × n_estimators × lr) for the winning family would
strengthen the "cloud does the expensive search" narrative.

---

## 3. Small factual alignments for the report

- The paper selects Model 8 (KNN+NB+DT, voting) primarily on **execution time** (1.15 s)
  with near-ceiling quality metrics — same shape as our tolerance-band rule. Worth citing
  as the precedent our UI makes explicit and interactive.
- Their stacking is *always* slower than voting (Fig. 5) — consistent with our latency
  column showing ensemble cost; if P1-4 adds a voting candidate, expect it mid-pack on
  latency.
- Their fog-node execution time of the *unselected* models (Fig. 11) is the same
  "what if we'd deployed X" analysis our Deploy-every-row button operationalizes.
- Citation: Tomer, V.; Sharma, S. Future Internet 2022, 14, 102. doi:10.3390/fi14040102.

---

## 4. Suggested order of work

1. ~~P1-1 + P1-2 (metrics) → retrain once → table/README/REPORT update.~~ **DONE.**
   Kappa, MCC and the MAE/RMSE/RAE/RRSE error suite are in the harness and the UI;
   see REPORT.md §2.
2. ~~P1-3 (CPU sampler) + P1-4 (voting candidate).~~ **DONE.**
   - Fog-node resources (CPU % of one core, RSS, threads) sampled every 2 s and shown on
     Overview; typical live-streaming load ~15% of one core / ~330 MB RSS including the
     Flask app and charts.
   - "Voting Ensemble (LR+DT+XGB)" is the 6th candidate: recorded at macro-F1 0.9629,
     kappa 0.9776, 0.0188 ms/rec, 441 KB — it **qualified in the 2pp band** and lost the
     latency tiebreaker to XGBoost (0.0096 ms), empirically matching the paper's finding
     that combining classifiers buys robustness at extra execution cost.
3. P2-1 (separate cloud/fog distributions) if time allows before submission; else
   document as known limitation.
4. P2-2 (kappa as 4th tiebreaker) — also done: the Model Lab tiebreaker selector now
   offers "highest Cohen's kappa".
5. P2-3 / P3-x as "future work" bullets, honestly labeled.
