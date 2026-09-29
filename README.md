# SentryMesh — Detecting IoT Attacks Using an Ensemble Machine Learning Model

CSC701 Machine Learning (Rev-2019 'C' Scheme, Sem VII, University of Mumbai) — elective mini-project.

SentryMesh simulates a **two-tier ML deployment** for IoT network security:

- **Cloud tier** (heavy, run once / on demand): trains six candidate classifiers on labeled
  attack data, evaluates every candidate with the same harness, and applies an explicit
  accuracy-vs-lightness rule to select a single deployment winner.
- **Fog/edge tier** (cheap, run forever): loads *only* the winning model and classifies a
  simulated live IoT gateway stream record-by-record, with a DBSCAN anomaly backstop for
  novel/zero-day traffic.

Attack categories: **Benign, DoS, Probing (recon), Authentication (credential)** — plus a
distinct **ANOMALOUS** flag from the unsupervised safety net.

The cloud and fog tiers see **separate distributions** (mirroring the source paper's
KDDTrain+ / KDDTest+ design): training uses the baseline regime, while the live simulator
replays a drifted regime (or a disjoint half of a real dataset) whose true labels are
carried through, so the edge tier reports **live accuracy on unseen data** on Overview.

## Run it

```bash
pip install flask scikit-learn pandas numpy joblib xgboost
python run.py            # http://127.0.0.1:5000
```

Open **Model Lab → Train Models** to launch the cloud-role sweep (a few minutes), then
**Live Detection → Start simulator** to watch the deployed winner classify traffic live.
The sidebar toggle switches between the default dark theme and an equivalent light theme
(monochrome in both; the choice persists).

Optional: drop a real dataset CSV into `data/` before starting the server —
TON_IoT `train_test_network.csv` (primary) or any CICIoT2023 partition CSV (fallback).
Without a file, a deterministic synthetic IoT-flow generator (fixed seed, realistic
72/16/8/4% class imbalance) is used so every feature of the system still runs.

## The six candidates (syllabus-mapped)

| Candidate | Family | Module |
|---|---|---|
| Logistic Regression | linear model | M2 — Regression & Trees |
| Decision Tree (CART/Gini) | single tree | M2 |
| Random Forest | bagging ensemble | M3 — Ensemble Learning |
| Gradient Boosting (XGBoost) | boosting ensemble | M3 |
| SVM (RBF kernel) | kernel method | M4 — Classification |
| Voting Ensemble (LR+DT+XGB) | soft-vote combination of three base learners | M3 — "ways to combine classifiers"; the source paper's core method |

Every candidate gets the identical evaluation: confusion matrix, per-class precision/recall/F1,
macro-F1, macro precision, macro recall (sensitivity), Cohen's kappa, MCC, ROC-AUC (ovR),
the Weka-style error suite (MAE/RMSE/RAE/RRSE),
**serialized model size**, and **batch-averaged per-record latency**.
Training uses stratified 5-fold cross-validation.

## Model selection rule (the project's core argument)

> Among all candidates whose **macro-F1 is within 2 percentage points of the best**,
> deploy the one with the **lowest inference latency**.

The full comparison table, the tolerance-band qualifiers, and the sentence-level reasoning
are stored in `artifacts/training_result.json` and rendered on the Model Lab page.

## Paper vs. SentryMesh model lineup

The source paper is Tomer & Sharma, *Future Internet* 2022, 14:102 (NSL-KDD, Weka).
SentryMesh replicates its **methodology** and deliberately extends its **model set**:

| Aspect | Paper (Future Internet 2022) | SentryMesh |
|---|---|---|
| Base classifiers | Decision Tree, Random Forest, KNN, Logistic Regression, Naïve Bayes | Logistic Regression, Decision Tree, Random Forest kept; KNN and Naïve Bayes replaced by XGBoost and RBF-SVM |
| Ensembles tested | Voting **and** stacking over all 5C2 = 10 base pairs (20 models) | One curated soft-voting trio (LR + DT + XGBoost) evaluated alongside the individual learners |
| Winning model | KNN+NB+DT voting — chosen mainly on execution time (1.15 s) | XGBoost or Voting, chosen by the interactive tolerance-band rule (latency as first tiebreaker) |
| Selection rule | Implicit inside Algorithm 1 (`isResultBetter`) | Explicit, interactive: F1 tolerance band (0–10 pp slider) + tiebreakers {latency, size, accuracy, kappa}, with manual deploy override |
| Metrics | Kappa, F-measure, ROC, MCC; Weka error suite (MAE/RMSE/RAE/RRSE) | All adopted + macro precision/recall (sensitivity) added |
| Dimensionality reduction | Explicitly skipped | Evaluated, not assumed: PCA 12→1 retains 99.4% variance but collapses macro-F1, so full features are deployed |
| Train/test separation | KDDTrain+ vs KDDTest+ files | Cloud trains on the baseline regime; the fog simulator replays a drifted regime / disjoint half, reporting **live accuracy on unseen records** |
| Zero-day handling | None — fully supervised | DBSCAN anomaly backstop on a rolling window (Module 5) |
| Fog cost report | "<10% CPU" | Live psutil telemetry: CPU %, RSS, thread count on Overview |
| Dataset | NSL-KDD (1999-era) | TON_IoT / CICIoT2023 loaders, deterministic synthetic fallback for reproducibility |

Why replace KNN and Naïve Bayes: KNN must retain the entire training set at inference,
which is a poor fit for a memory-constrained fog node (our table shows comparable models
at 10–100× its latency), and Naïve Bayes' feature-independence assumption underperforms on
correlated IoT flow features. XGBoost is not foreign to the paper — it cites XGBoost [27]
reaching 99.6% AUC — so the upgrade is defensible: in the recorded run XGBoost leads on
macro-F1 (0.9706) at 9.6 ms/record, and the soft-voting trio also qualifies inside the band.

## PCA (Module 6)

PCA (≥95% variance retained) is fitted on the training features and the winner is re-evaluated
on the reduced set. The deployed variant (full vs reduced) is chosen by the measured
accuracy-vs-latency tradeoff — never assumed.

## Page-by-page guide

### Overview — mission control
At-a-glance health of the whole pipeline: what is deployed, how much traffic the edge tier
has processed, and what it is catching right now.

| Element | Purpose / effect |
|---|---|
| Records processed · Attacks detected · Avg edge latency | Live counters from the edge engine; attacks use the single red accent |
| Deployed model card | Current edge model, variant, macro-F1, and provenance (rule-automatic vs manual override) |
| Start / Pause / Reset | Start resumes the traffic-simulator thread, Pause halts it, Reset halts it **and** wipes counters, feed, device history and time series |
| Speed presets 1x / 5x / 20x | Sets the replay rate to 20 / 100 / 400 records per second — the time charts visibly accumulate faster |
| Chart window: last 1 min / 5 min / since start | Refetches the windowed data for **both** the distribution and the time charts |
| Attack-type distribution | Per-class counts inside the chosen window (Benign gray, attacks red). Clicking an attack-class bar navigates to Live Detection pre-filtered to that class |
| Per-class live accuracy | One accuracy bar per class over all scored records (bar length = accuracy, value = % of n). Attack bars turn red when the class has any miss; clicking an attack bar jumps to Live Detection pre-filtered to that class, with the miss count in the tooltip |
| Traffic over time | 2-second buckets (last 60 shown): gray bar = total volume, red fill = attack share |
| Traffic over time by class | One thin line per class — Benign solid gray; DoS solid red; Probing dashed; Authentication dotted; ANOMALOUS long-dash faint — so a DoS spike is distinguishable from a Probing spike |

### Model Lab — the cloud role (the project's core)
Trains the five syllabus candidates, compares them uniformly, and makes the selection rule
something you can explore instead of a fixed fact.

| Element | Purpose / effect |
|---|---|
| **Train Models** | Triggers the full cloud sweep (6 candidates × 5-fold CV, ~2 min); the page polls status and re-renders when it finishes. Retraining replaces the stored metrics, reasoning, and automatic deployment |
| Sortable comparison table | Click any column header to sort by it (metrics descend, Size/Latency ascend; click again to flip). Sorting reorders rows only — RULE PICK / IN BAND tags, winner highlighting and Deploy state always stay attached to the correct models |
| Deployment banner | Who is on the edge and why: **RULE AUTOMATIC** or **MANUAL OVERRIDE** (red border) with a one-click **Reset to Automatic** |
| Tolerance band slider (0–10 pp, default 2) | Qualifies every candidate whose macro-F1 is within the band of the best. Recomputes — instantly, with **no retraining** — the IN BAND row outlines, RULE PICK badge, qualified list, scatter shading, and (debounced) the sentence-level reasoning |
| Tiebreaker selector | lowest inference latency / smallest model size / highest raw accuracy — picks the winner among the qualified candidates (e.g. a wide band + size tiebreaker crowns Logistic Regression; 2 pp + latency crowns XGBoost) |
| Deploy variant toggle | Deploys the selected model as `full_features` or `pca_reduced`; the reduced variant wraps the classifier behind a PCA fitted for ≥95% variance and costs measurable accuracy |
| Candidate comparison table | Accuracy, macro-F1, Cohen's kappa, MCC, ROC-AUC (ovR), CV macro-F1 ± std, serialized size, batch-averaged latency per candidate. Every row has a **Deploy** button: deploying a non-winner refits that candidate on the stored split, swaps the edge artifact, and flips the banner to manual override |
| Latency vs macro-F1 scatter | Log-scale latency (x) vs macro-F1 (y). The qualifying band is shaded above *best-F1 − tolerance* and moves live with the slider — you can see exactly why models fall in or out of contention. White = rule pick, red = manually deployed, gray = other candidates |
| PCA cumulative-variance chart | Explained variance vs component count (full curve), dashed 95% target line, red marker at the chosen k with its retained variance — the visible justification for the dimensionality decision |
| Selection reasoning panel | The rule's output in sentences: who qualified, who was excluded (with their numbers), which tiebreaker decided it; follows the slider/tiebreaker live. The kv block records best-F1, band size, tiebreaker, and deployed provenance |
| Confusion-matrix heatmap | The deployed model's held-out confusion matrix: diagonal cells in white at intensity ∝ share (correct), off-diagonal errors in red at intensity ∝ count; hover shows exact actual→predicted counts. Re-renders for whichever model/variant you deploy; the caption shows the deployed artifact's re-measured accuracy/macro-F1 |
| PCA: full vs reduced table | Side-by-side features, variance retained, accuracy, macro-F1, latency and size, plus the data-driven explanation of which variant was deployed and why |

### Live Detection — the fog/edge role
Watches the deployed winner classify a simulated gateway stream record-by-record, and makes
the DBSCAN anomaly backstop visible instead of just a text flag.

| Element | Purpose / effect |
|---|---|
| Start simulator / Stop + RUNNING badge | Controls the replay thread; the tally line totals benign / DoS / probing / auth / anomalous as they accumulate |
| **⏸ Pause feed / ▶ Resume** | Freezes the feed list so a record can actually be read at 400 rec/s — the simulator, counters, tally and DBSCAN chart keep running behind it; the next change (new record or filter edit) repaints on resume |
| DBSCAN window scatter | A viz-only 2D PCA projection (independent of any deployed PCA) of the rolling 1,500-record window the backstop clusters. Gray squares = points in dense clusters; larger red squares = unclustered/anomalous points. Updates live; the caption reports point count, anomaly count, `eps`/`min_samples`, and notes when labels come from the last backstop pass (it re-fits every 25 records) |
| Filter bar | Five class checkboxes + a device/IP substring search; both filter the feed instantly with no reload. Arriving via an Overview bar click pre-applies that single class |
| Live feed | Timestamp, source IP, predicted class (attack rows in red), confidence %, per-record latency. Records the backstop flagged carry a red **DBSCAN** tag; with only ANOMALOUS checked, flagged records display as ANOMALOUS |

### Investigate Device — per-device forensics
Answers "what has this one device been doing?" — a per-IP behaviour story rather than
aggregate counts.

| Element | Purpose / effect |
|---|---|
| Search box | Type or paste an IP and press Enter to load it; typing also narrows the device list |
| Devices seen | Top 40 devices by volume, flagged red when they have any detection; click a row to open it, or click **watch ▸** to jump to Live Detection with the search box pre-filled with that IP (only that device's traffic shows) |
| Watch in live feed (device detail) | Same jump from the detail panel's Actions row — go from a device's forensics story straight to watching its traffic live |
| Classification timeline | One dot per event at its timestamp (gray = benign, red = attack/anomaly), so patterns like "mostly benign, then repeated DoS" are visible at a glance; hover shows time, class, confidence |
| Class breakdown + recent events | Per-class counts and the last 30 events with confidence and latency |
| Fog-node resources (Overview) | Live CPU % of one core, RSS memory, and thread count of the edge process, sampled every 2 s — the same instrumentation the source paper reports ("<10% CPU" on its fog node) |

### How It Works
Static reference: six numbered cards, one per CSC701 (Rev-2019 'C' Scheme) module, mapping
each module's concepts to the part of SentryMesh that implements them.

### Global controls
Sidebar navigation; the tier note shows the currently deployed model; the ◐ toggle switches
dark/light — both are the same monochrome system (inverted gray ramp, adjusted single red)
and the choice persists across reloads. All charts re-render in place on toggle.

## Layout

```
sentrymesh/         Flask backend
  datasets.py       TON_IoT / CICIoT2023 / synthetic loaders + subsampling
  cloud.py          training, evaluation, selection, PCA, artifact export
  edge.py           deployed-model engine, simulator, DBSCAN backstop
  app.py            REST API + static hosting
static/             monochrome SPA (index.html, app.js, style.css)
artifacts/          deployed_model.joblib, training_result.json
data/               optional real dataset drop-in
train.py            standalone cloud-role run (same pipeline as the UI button)
REPORT.md           factual summary of the recorded training run
```

See [REPORT.md](REPORT.md) for the recorded run's actual numbers.
