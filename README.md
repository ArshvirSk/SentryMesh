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
| Traffic over time | 2-second buckets (last 60 shown): gray bar = total volume, red fill = attack share |
| Traffic over time by class | One thin line per class — Benign solid gray; DoS solid red; Probing dashed; Authentication dotted; ANOMALOUS long-dash faint — so a DoS spike is distinguishable from a Probing spike |

### Model Lab — the cloud role (the project's core)
Trains the five syllabus candidates, compares them uniformly, and makes the selection rule
something you can explore instead of a fixed fact.

| Element | Purpose / effect |
|---|---|
| **Train Models** | Triggers the full cloud sweep (6 candidates × 5-fold CV, ~2 min); the page polls status and re-renders when it finishes. Retraining replaces the stored metrics, reasoning, and automatic deployment |
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
| DBSCAN window scatter | A viz-only 2D PCA projection (independent of any deployed PCA) of the rolling 1,500-record window the backstop clusters. Gray squares = points in dense clusters; larger red squares = unclustered/anomalous points. Updates live; the caption reports point count, anomaly count, `eps`/`min_samples`, and notes when labels come from the last backstop pass (it re-fits every 25 records) |
| Filter bar | Five class checkboxes + a device/IP substring search; both filter the feed instantly with no reload. Arriving via an Overview bar click pre-applies that single class |
| Live feed | Timestamp, source IP, predicted class (attack rows in red), confidence %, per-record latency. Records the backstop flagged carry a red **DBSCAN** tag; with only ANOMALOUS checked, flagged records display as ANOMALOUS |

### Investigate Device — per-device forensics
Answers "what has this one device been doing?" — a per-IP behaviour story rather than
aggregate counts.

| Element | Purpose / effect |
|---|---|
| Search box | Type or paste an IP and press Enter to load it; typing also narrows the device list |
| Devices seen | Top 40 devices by volume, flagged red when they have any detection; click a row to open it |
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
