# SentryMesh cloud role: candidate training, uniform evaluation, model selection.
#
# Given (X, y) this module:
#   1. trains all five candidate classifiers on the same stratified split,
#      with stratified k-fold cross-validation during training (Module 3),
#   2. scores every candidate with the identical evaluation harness
#      (confusion matrix, per-class P/R/F1, macro-F1, ROC-AUC ovR -- Module 2),
#      plus the two non-accuracy costs: serialized model size and
#      batch-averaged per-record inference latency,
#   3. applies the explicit selection rule: among candidates within
#      MACRO_F1_TOLERANCE of the best macro-F1, pick the lowest latency,
#   4. fits PCA on the training features and re-evaluates the winner on
#      PCA-reduced features (>= PCA_VARIANCE retained), choosing the
#      full-vs-reduced version that is the better edge tradeoff (Module 6),
#   5. exports the chosen artifact for the fog/edge tier.

import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .datasets import CLASSES, CLASS_INDEX, feature_sha

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MACRO_F1_TOLERANCE = 0.02   # 2 percentage points
PCA_VARIANCE = 0.95         # retain >= 95% explained variance
RANDOM_STATE = 42

# Latency is measured over this many test records, averaged -- a single warm
# call is too noisy to drive a deployment decision.
LATENCY_BATCH = 500


# ---------------------------------------------------------------------------
# Candidates (one per syllabus module; see How It Works page)
# ---------------------------------------------------------------------------
def build_candidates(seed=RANDOM_STATE):
    cands = [
        ("Logistic Regression", "Module 2",
         Pipeline([("scaler", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])),
        ("Decision Tree (CART/Gini)", "Module 2",
         Pipeline([("clf", _decision_tree(seed))])),
        ("Random Forest (bagging)", "Module 3",
         Pipeline([("clf", RandomForestClassifier(
             n_estimators=120, max_depth=None, n_jobs=-1, random_state=seed))])),
        (f"Gradient Boosting ({'XGBoost' if HAS_XGB else 'sklearn GBM'})", "Module 3",
         Pipeline([("clf", _boosting(seed))])),
        ("SVM (RBF kernel)", "Module 4",
         Pipeline([("scaler", StandardScaler()),
                   ("clf", SVC(kernel="rbf", C=2.0, gamma="scale",
                               probability=False, random_state=seed))])),
        # Module 3 "ways to combine classifiers" + the source paper's core method:
        # a soft-voting ensemble over three diverse base learners (paper Section 2.3).
        ("Voting Ensemble (LR+DT+XGB)", "Module 3",
         Pipeline([
             ("clf", VotingClassifier(
                 estimators=[
                     ("lr", Pipeline([("scaler", StandardScaler()),
                                      ("lr", LogisticRegression(max_iter=2000, C=1.0,
                                                                random_state=seed))])),
                     ("dt", _decision_tree(seed)),
                     ("xgb", _boosting(seed)),
                 ],
                 voting="soft"),
             )
         ])),
    ]
    return cands


def _decision_tree(seed):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(criterion="gini", max_depth=None,
                                  min_samples_leaf=2, random_state=seed)


def predict_scores(pipe, X):
    """Uniform class-probability interface: predict_proba when available,
    else softmax-normalised decision_function (kernel SVM without Platt scaling,
    which is ~5x cheaper to fit on tens of thousands of rows)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    from scipy.special import softmax
    return softmax(pipe.decision_function(X), axis=1)


def _boosting(seed):
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.15,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="mlogloss", random_state=seed, verbosity=0,
        )
    return GradientBoostingClassifier(n_estimators=120, learning_rate=0.1,
                                      max_depth=3, random_state=seed)


# ---------------------------------------------------------------------------
# Uniform evaluation harness
# ---------------------------------------------------------------------------
def _round_or_none(x, nd=4):
    """Round, mapping NaN/inf to None so json.dumps emits valid JSON."""
    v = round(float(x), nd)
    return v if math.isfinite(v) else None


def evaluate_candidate(name, module, pipe, X_train, y_train, X_test, y_test,
                       kfold=5, with_cv=True):
    """Fit one candidate; score it with the exact same harness as every other.

    Labels arrive as integer codes (0..3 = CLASSES order) because XGBoost
    requires numeric targets; probability column i therefore corresponds to
    CLASSES[i] for every candidate.
    """
    t0 = time.perf_counter()
    if with_cv:
        cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    else:
        cv_scores = [float("nan")] * kfold
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    proba = predict_scores(pipe, X_test)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASSES))), zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    # Paper's headline quality metrics (Figs. 6/8/9): kappa + MCC -- both more
    # informative than accuracy under class imbalance.
    kappa = float(cohen_kappa_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred))

    # True-class indicator matrix (used by ROC-AUC and the error suite below).
    y_test_bin = np.zeros((len(y_test), len(CLASSES)))
    for i in range(len(CLASSES)):
        y_test_bin[:, i] = (y_test == i).astype(int)

    # Paper's error suite (Figs. 7/10), Weka-style, from predicted probabilities:
    # errors measured per (instance, class) cell against the true indicator
    # vector; RAE/RRSE normalised by the zero-R (class-prior) predictor.
    prior = np.bincount(np.asarray(y_train), minlength=len(CLASSES)) / len(y_train)
    A = y_test_bin
    diff = proba - A
    prior_mat = np.tile(prior, (len(y_test), 1))
    errors = {
        "mae": round(float(np.mean(np.abs(diff))), 4),
        "rmse": round(float(np.sqrt(np.mean(diff ** 2))), 4),
        "rae": round(float(np.abs(diff).sum() / np.abs(prior_mat - A).sum()), 4),
        "rrse": round(float(np.sqrt((diff ** 2).sum() / ((prior_mat - A) ** 2).sum())), 4),
    }

    # ROC-AUC, one-vs-rest, averaged (Module 2)
    try:
        roc_auc = float(roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc = float("nan")

    # Non-accuracy cost #1: serialized model size on disk
    path = ARTIFACT_DIR / "_tmp_candidate.joblib"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path, compress=3)
    size_bytes = path.stat().st_size
    path.unlink(missing_ok=True)

    # Non-accuracy cost #2: average per-record latency over a batch of test
    # predictions (not a single warm call).
    batch = X_test.iloc[:LATENCY_BATCH] if hasattr(X_test, "iloc") else X_test[:LATENCY_BATCH]
    pipe.predict(batch.iloc[:1] if hasattr(batch, "iloc") else batch[:1])  # warm-up
    t0 = time.perf_counter()
    for _ in range(5):
        pipe.predict(batch)
    lat_ms = (time.perf_counter() - t0) / (5 * len(batch)) * 1000.0

    return {
        "name": name, "module": module,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
        "errors": errors,
        "roc_auc_ovr": _round_or_none(roc_auc, 4),
        # NaN never reaches JSON: null when CV was skipped (PCA comparison).
        "cv_macro_f1_mean": _round_or_none(np.nanmean(cv_scores), 4),
        "cv_macro_f1_std": _round_or_none(np.nanstd(cv_scores), 4),
        "cv_folds": kfold,
        "confusion_matrix": cm.tolist(),
        "per_class": {c: {"precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4),
                          "f1": round(float(f1[i]), 4),
                          "support": int(support[i])} for i, c in enumerate(CLASSES)},
        "model_size_bytes": int(size_bytes),
        "model_size_kb": round(size_bytes / 1024, 1),
        "latency_ms_per_record": round(float(lat_ms), 4),
        "fit_seconds": round(fit_s, 2),
    }


# ---------------------------------------------------------------------------
# Selection rule (the project's core argument -- surfaced verbatim in the UI)
# The UI re-runs this function live over stored metrics as the user moves the
# tolerance slider / tiebreaker selector; no retraining is involved.
# ---------------------------------------------------------------------------
TIEBREAKERS = {
    "latency": "lowest inference latency",
    "size": "smallest model size",
    "accuracy": "highest raw accuracy",
    "kappa": "highest Cohen's kappa",
}


def _tiebreaker_key(tiebreaker):
    return {
        "latency": lambda r: r["latency_ms_per_record"],
        "size": lambda r: r["model_size_bytes"],
        "accuracy": lambda r: -r["accuracy"],
        "kappa": lambda r: -r["kappa"],
    }[tiebreaker]


def _tb_metric(r, tiebreaker):
    return {"latency": lambda: f"{r['latency_ms_per_record']:.3f} ms/record",
            "size": lambda: f"{r['model_size_kb']:.1f} KB",
            "accuracy": lambda: f"{r['accuracy']:.4f} accuracy",
            "kappa": lambda: f"kappa {r['kappa']:.4f}"}[tiebreaker]()


def select_winner(results, tolerance=MACRO_F1_TOLERANCE, tiebreaker="latency"):
    """Tolerance-band rule: qualify candidates within `tolerance` pp of the best
    macro-F1, then pick by the configured tiebreaker. Pure recomputation."""
    best_f1 = max(r["macro_f1"] for r in results)
    band = [r for r in results if r["macro_f1"] >= best_f1 - tolerance]
    winner = min(band, key=_tiebreaker_key(tiebreaker))

    band_names = ", ".join(f"{r['name']} (macro-F1 {r['macro_f1']:.3f})" for r in band)
    others = [r for r in results if r not in band]
    reason = (
        f"Best macro-F1 of any candidate was {best_f1:.4f}. With the "
        f"{tolerance * 100:.0f}-percentage-point tolerance band "
        f"(>= {best_f1 - tolerance:.4f}), {len(band)} of {len(results)} candidates qualified: "
        f"{band_names}. "
    )
    if others:
        reason += (
            "Excluded: "
            + "; ".join(f"{r['name']} (macro-F1 {r['macro_f1']:.3f}, "
                        f"{_tb_metric(r, tiebreaker)})" for r in others)
            + ". "
        )
    _why = {
        "latency": lambda r: f"its inference latency ({r['latency_ms_per_record']:.3f} ms/record) is the lowest in the band",
        "size": lambda r: f"its serialized size ({r['model_size_kb']:.1f} KB) is the smallest in the band",
        "accuracy": lambda r: f"its raw accuracy ({r['accuracy']:.4f}) is the highest in the band",
        "kappa": lambda r: f"its Cohen's kappa ({r['kappa']:.4f}) is the highest in the band",
    }[tiebreaker]
    reason += (
        f"Among the qualifying models, {winner['name']} was selected for fog/edge deployment "
        f"because {_why(winner)} -- the smallest accuracy sacrifice that buys the cheapest "
        f"real-time classifier for a resource-constrained IoT gateway."
    )
    return winner, {"best_macro_f1": best_f1, "tolerance": tolerance,
                    "tiebreaker": tiebreaker,
                    "qualified": [r["name"] for r in band], "reasoning": reason}


# ---------------------------------------------------------------------------
# PCA comparison on the winner (Module 6)
# ---------------------------------------------------------------------------
def pca_comparison(winner_name, module, X_train, y_train, X_test, y_test, seed=RANDOM_STATE):
    """Evaluate the winning model on full features vs PCA-reduced features."""
    n_full = X_train.shape[1]
    pca = PCA(n_components=PCA_VARIANCE, random_state=seed).fit(X_train)
    n_red = int(pca.n_components_)
    var_retained = float(pca.explained_variance_ratio_.sum())

    # Cumulative explained-variance curve (UI chart): fit a full PCA so the
    # curve spans every component, not just the ones the 95% cut kept.
    full_pca = PCA(random_state=seed).fit(X_train)
    cum = np.cumsum(full_pca.explained_variance_ratio_)
    cumvar_curve = [{"k": i + 1, "variance": round(float(v), 4)}
                    for i, v in enumerate(cum)]

    def _eval(label, pipe, Xe):
        # No CV here -- CV already validated the winner during selection.
        return evaluate_candidate(label, module, pipe, X_train, y_train, Xe, y_test,
                                  with_cv=False)

    # Rebuild a fresh instance of the winning pipeline
    for cand_name, _, pipe in build_candidates(seed):
        if cand_name == winner_name:
            full_res = _eval(winner_name, pipe, X_test)
            reduced_pipe = Pipeline([("pca", PCA(n_components=n_red, random_state=seed)),
                                     ("clf", pipe.named_steps["clf"])])
            reduced_res = _eval(winner_name + " (PCA-reduced)", reduced_pipe, X_test)
            break
    else:
        raise RuntimeError(f"winner {winner_name!r} not among candidates")

    # Data-driven deployment choice: reduction must cost almost nothing in
    # macro-F1 while buying a meaningful latency/size win at the edge.
    full_f1 = full_res["macro_f1"]
    red_f1 = reduced_res["macro_f1"]
    f1_cost = full_f1 - red_f1
    lat_gain_pct = (1 - reduced_res["latency_ms_per_record"] / max(full_res["latency_ms_per_record"], 1e-9)) * 100
    size_gain_pct = (1 - reduced_res["model_size_bytes"] / max(full_res["model_size_bytes"], 1)) * 100
    use_reduced = (f1_cost <= 0.01) and (reduced_res["latency_ms_per_record"] <= full_res["latency_ms_per_record"])

    return {
        "n_features_full": n_full,
        "n_features_reduced": n_red,
        "variance_target": PCA_VARIANCE,
        "variance_retained": round(var_retained, 4),
        "cumvar_curve": cumvar_curve,
        "full": full_res,
        "reduced": reduced_res,
        "macro_f1_cost": round(f1_cost, 4),
        "latency_gain_pct": round(float(lat_gain_pct), 1),
        "size_gain_pct": round(float(size_gain_pct), 1),
        "use_reduced": bool(use_reduced),
        "explanation": (
            f"PCA retained {var_retained * 100:.1f}% variance with {n_full} -> {n_red} features. "
            f"Macro-F1 moved {full_f1:.4f} -> {red_f1:.4f} (cost {f1_cost:.4f}); "
            f"latency changed to {reduced_res['latency_ms_per_record']:.3f} ms/record "
            f"({lat_gain_pct:+.1f}%); serialized size changed {size_gain_pct:+.1f}%. "
            + ("Deploying the PCA-reduced model: accuracy cost is within tolerance and the "
               "edge gains latency/size." if use_reduced else
               "Deploying the full-feature model: the reduction did not pay for itself "
               "(accuracy cost too high or no latency benefit), so the edge keeps full features.")
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def make_split(X, y, test_size=0.25, seed=RANDOM_STATE):
    """Deterministic stratified split with integer-coded labels (XGBoost needs
    numeric targets; CLASS_INDEX keeps probability columns aligned with CLASSES)."""
    y_codes = y.map(CLASS_INDEX).astype(int)
    return train_test_split(X, y_codes, test_size=test_size,
                            stratify=y_codes, random_state=seed)


def _train_candidate(name, module, X_train, y_train, X_test, y_test, kfold=5, seed=RANDOM_STATE):
    """Train one candidate with k-fold CV and evaluate it on the shared split."""
    for cand_name, _, pipe in build_candidates(seed):
        if cand_name == name:
            return evaluate_candidate(name, module, pipe, X_train, y_train,
                                      X_test, y_test, kfold=kfold)
    raise ValueError(f"unknown candidate: {name}")


def export_deployment(name, variant, X_train, y_train, X_test, y_test, seed=RANDOM_STATE):
    """Refit `name` (optionally behind a fresh PCA) on the training split and
    write the fog/edge artifact. Returns (bundle, eval_metrics, variant)."""
    module = dict((n, m) for n, m, _ in build_candidates(seed)).get(name, "?")
    metrics = None
    for cand_name, _, pipe in build_candidates(seed):
        if cand_name != name:
            continue
        if variant == "pca_reduced":
            # Keep >= PCA_VARIANCE variance; n_components inferred on this split.
            probe = PCA(n_components=PCA_VARIANCE, random_state=seed).fit(X_train)
            k = int(probe.n_components_)
            deployed = Pipeline([("pca", PCA(n_components=k, random_state=seed)),
                                 ("clf", pipe.named_steps["clf"])])
        else:
            deployed = pipe
        deployed.fit(X_train, y_train)
        y_pred = deployed.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }
        break
    else:
        raise ValueError(f"unknown candidate: {name}")

    bundle = {
        "pipeline": deployed,
        "classes": CLASSES,
        "class_index": CLASS_INDEX,
        "proba_classes": list(range(len(CLASSES))),  # proba col i == CLASSES[i]
        "variant": variant,
        "winner_name": name,
        "module": module,
        "trained_on": {"n_train": len(X_train), "n_test": len(X_test),
                       "columns": list(X_train.columns),
                       "feature_sha": feature_sha(X_train)},
        "metrics": {"test_accuracy": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"]},
        "seed": seed,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT_DIR / "deployed_model.joblib", compress=3)
    return bundle, metrics


def record_deployment(result, name, variant, source, metrics):
    """Record who is on the edge and why (rule pick vs manual override) into the
    persisted training_result.json so the UI can show provenance."""
    result["deployment"] = {
        "model": name,
        "variant": variant,
        "source": source,          # "automatic" | "manual"
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "ts": time.time(),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "training_result.json").write_text(json.dumps(result, indent=2))
    return result["deployment"]


def run_training(X, y, sample_cap=None, kfold=5, test_size=0.25, seed=RANDOM_STATE):
    """Full cloud-role pipeline: split, train all candidates, evaluate, select, PCA, export."""
    if sample_cap and len(X) > sample_cap:
        X, y = X.iloc[:sample_cap], y.iloc[:sample_cap]

    X_train, X_test, y_train, y_test = make_split(X, y, test_size, seed)

    results = []
    for name, module, _pipe in build_candidates(seed):
        print(f"[cloud] training {name} ...")
        results.append(_train_candidate(name, module, X_train, y_train, X_test, y_test,
                                        kfold=kfold, seed=seed))
        print(f"[cloud]   macro-F1 {results[-1]['macro_f1']:.4f}  "
              f"lat {results[-1]['latency_ms_per_record']:.3f} ms  "
              f"size {results[-1]['model_size_kb']:.0f} KB")

    winner, selection = select_winner(results)
    print(f"[cloud] winner: {winner['name']}")

    pca_res = pca_comparison(winner["name"], winner["module"], X_train, y_train, X_test, y_test, seed)

    variant = "pca_reduced" if pca_res["use_reduced"] else "full_features"
    _bundle, metrics = export_deployment(winner["name"], variant,
                                         X_train, y_train, X_test, y_test, seed)
    model_path = str(ARTIFACT_DIR / "deployed_model.joblib")

    result = {
        "results": results,
        "winner": winner,
        "selection": selection,
        "pca": pca_res,
        "deployed_variant": variant,
        "model_path": model_path,
        "deployment": {
            "model": winner["name"],
            "variant": variant,
            "source": "automatic",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "ts": time.time(),
        },
        "dataset": {"n_rows": int(len(X)), "n_features": int(X.shape[1]),
                    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "class_dist": {k: int(v) for k, v in y.value_counts().items()}},
    }

    # Persist the full run so the UI (and REPORT.md) read the same record.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "training_result.json").write_text(json.dumps(result, indent=2))

    return result
