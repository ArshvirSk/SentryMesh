# SentryMesh Flask application.
#
# One process, two logical roles, distinguished by API surface and code module:
#   cloud role (/api/train*, /api/models, /api/deploy)  -- sentrymesh.cloud
#   fog/edge role (/api/sim*, /api/feed, /api/device*, /api/window) -- sentrymesh.edge

import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .cloud import (ARTIFACT_DIR, MACRO_F1_TOLERANCE, make_split, record_deployment,
                    run_training, select_winner, export_deployment)
from .datasets import CLASSES, load_dataset
from .edge import get_engine, get_simulator, set_stream_data, get_stream_data

app = Flask(__name__, static_folder=None)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
RESULT_PATH = ARTIFACT_DIR / "training_result.json"

_train_thread = None
_train_state = {"running": False, "started_at": None, "finished_at": None,
                "error": None, "result": None}

_ds_lock = threading.Lock()
DATA_CACHE = {"X": None, "y": None, "meta": None, "info": None}


def ensure_data(force=False):
    with _ds_lock:
        if force or DATA_CACHE["X"] is None:
            X, y, meta, info = load_dataset(sample_rows=60000, seed=42)
            DATA_CACHE.update({"X": X, "y": y, "meta": meta, "info": info})
            set_stream_data(X, y, meta, info)
            get_simulator().load_data(X, meta)   # simulator replays these rows
    return DATA_CACHE["X"], DATA_CACHE["y"], DATA_CACHE["meta"], DATA_CACHE["info"]


def _load_result():
    return json.loads(RESULT_PATH.read_text()) if RESULT_PATH.exists() else None


# ---------------------------------------------------------------------------
# Cloud role endpoints
# ---------------------------------------------------------------------------
@app.post("/api/train")
def api_train():
    """Trigger the cloud-role training sweep (retrainable on demand, like MLOps)."""
    global _train_thread
    if _train_state["running"]:
        return jsonify({"status": "already_running"}), 409
    ensure_data()
    X, y = DATA_CACHE["X"], DATA_CACHE["y"]

    def _job():
        _train_state.update({"running": True, "started_at": time.time(),
                             "finished_at": None, "error": None, "result": None})
        try:
            result = run_training(X, y)
            _train_state["result"] = {"winner": result["winner"]["name"],
                                      "variant": result["deployed_variant"]}
            get_engine().load()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _train_state["error"] = str(exc)
        finally:
            _train_state["running"] = False
            _train_state["finished_at"] = time.time()

    _train_thread = threading.Thread(target=_job, daemon=True)
    _train_thread.start()
    return jsonify({"status": "started"})


@app.get("/api/train/status")
def api_train_status():
    out = dict(_train_state)
    out.pop("result", None)
    if out.get("started_at") and not out["running"] and out.get("finished_at"):
        out["elapsed_s"] = round(out["finished_at"] - out["started_at"], 1)
    return jsonify(out)


@app.get("/api/models")
def api_models():
    result = _load_result()
    if not result:
        return jsonify({"trained": False})
    result = dict(result)
    result["trained"] = True
    return jsonify(result)


@app.post("/api/deploy")
def api_deploy():
    """Deploy any candidate (manual override) or revert to the rule's automatic pick.

    Body: {"model": "<candidate name>", "variant": "full_features"|"pca_reduced",
           "source": "manual"|"automatic"}
    Pure cloud-role refit on the stored split -- no simulator state is touched.
    """
    body = request.get_json(force=True, silent=True) or {}
    name = body.get("model")
    variant = body.get("variant", "full_features")
    source = body.get("source", "manual")
    if variant not in ("full_features", "pca_reduced"):
        return jsonify({"error": "variant must be full_features or pca_reduced"}), 400

    result = _load_result()
    if not result:
        return jsonify({"error": "no training run yet"}), 400
    if source == "manual":
        valid = {r["name"] for r in result["results"]}
        if name not in valid:
            return jsonify({"error": f"unknown model: {name}"}), 400
    else:
        # Automatic: re-run the rule with the UI's current tolerance/tiebreaker
        # (falls back to the stored selection's settings).
        stored = result["selection"]
        tolerance = float(body.get("tolerance", stored.get("tolerance", MACRO_F1_TOLERANCE)))
        tiebreaker = body.get("tiebreaker", stored.get("tiebreaker", "latency"))
        if not (0 <= tolerance <= 0.10):
            return jsonify({"error": "tolerance must be 0..0.10"}), 400
        if tiebreaker not in ("latency", "size", "accuracy", "kappa"):
            return jsonify({"error": "tiebreaker must be latency|size|accuracy|kappa"}), 400
        winner, selection = select_winner(result["results"], tolerance=tolerance,
                                          tiebreaker=tiebreaker)
        name = winner["name"]
        result["selection"] = selection

    ensure_data()
    X, y = DATA_CACHE["X"], DATA_CACHE["y"]
    X_train, X_test, y_train, y_test = make_split(X, y, 0.25, seed=42)
    try:
        _bundle, metrics = export_deployment(name, variant, X_train, y_train, X_test, y_test)
    except Exception as exc:
        return jsonify({"error": f"deployment failed: {exc}"}), 500
    dep = record_deployment(result, name, variant, source, metrics)
    get_engine().load()
    return jsonify({"status": "deployed", "deployment": dep})


@app.post("/api/select")
def api_select():
    """Re-run the selection rule over stored metrics (no retraining, no deploy).
    Used by the Model Lab slider for instant what-if recomputation."""
    body = request.get_json(force=True, silent=True) or {}
    result = _load_result()
    if not result:
        return jsonify({"error": "no training run yet"}), 400
    tolerance = float(body.get("tolerance", MACRO_F1_TOLERANCE))
    tiebreaker = body.get("tiebreaker", "latency")
    if not (0 <= tolerance <= 0.10):
        return jsonify({"error": "tolerance must be 0..0.10"}), 400
    if tiebreaker not in ("latency", "size", "accuracy", "kappa"):
        return jsonify({"error": "tiebreaker must be latency|size|accuracy|kappa"}), 400
    _winner, selection = select_winner(result["results"], tolerance=tolerance,
                                       tiebreaker=tiebreaker)
    return jsonify({"selection": selection})


# ---------------------------------------------------------------------------
# Fog/edge role endpoints
# ---------------------------------------------------------------------------
@app.get("/api/state")
def api_state():
    engine = get_engine()
    ensure_data()
    X, y, meta, info = get_stream_data()
    result = _load_result()
    sim = get_simulator()
    return jsonify({
        "deployed": engine.is_deployed,
        "model": {
            "name": result["deployment"]["model"] if result else None,
            "variant": result["deployment"]["variant"] if result else None,
            "macro_f1": result["deployment"]["macro_f1"] if result else None,
            "accuracy": result["deployment"]["accuracy"] if result else None,
            "source": result["deployment"]["source"] if result else None,
        } if result and "deployment" in result else None,
        "sim_running": sim.is_running,
        "sim_delay": sim.delay,
        "sim_rps": round(1.0 / sim.delay) if sim.delay > 0 else None,
        "dataset": {"source": info["source"], "rows": int(len(X)),
                    "n_features": int(X.shape[1]),
                    "class_dist": {k: int(v) for k, v in info["class_dist"].items()}},
        "classes": CLASSES,
    })


@app.post("/api/sim/start")
def api_sim_start():
    ensure_data()
    sim = get_simulator()
    try:
        sim.start()
        return jsonify({"status": "started", "rps": round(1 / sim.delay) if sim.delay > 0 else None})
    except RuntimeError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400


@app.post("/api/sim/stop")
def api_sim_stop():
    get_simulator().stop()
    return jsonify({"status": "stopped"})


@app.post("/api/sim/speed")
def api_sim_speed():
    body = request.get_json(force=True, silent=True) or {}
    try:
        rps = float(body.get("rps", 20))
        if not (0.5 <= rps <= 500):
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "rps must be 0.5..500"}), 400
    get_simulator().set_speed(rps)
    return jsonify({"status": "ok", "rps": rps})


@app.post("/api/sim/reset")
def api_sim_reset():
    get_simulator().reset()
    return jsonify({"status": "reset"})


@app.get("/api/timeseries")
def api_timeseries():
    engine = get_engine()
    since = request.args.get("since", type=float)
    rows = engine.timeseries(since=since)
    return jsonify({"buckets": rows, "classes": CLASSES + ["ANOMALOUS"]})


@app.get("/api/window")
def api_window():
    return jsonify(get_engine().window_viz())


@app.get("/api/feed")
def api_feed():
    engine = get_engine()
    recent = []
    for ip, hist in engine.device_history.items():
        for rec in list(hist["events"])[-10:]:
            recent.append(rec)
    recent.sort(key=lambda r: r["ts"], reverse=True)
    recent = recent[:120]
    lat = list(engine.latencies)
    avg_lat = round(sum(lat) / len(lat), 3) if lat else None
    return jsonify({
        "feed": recent,
        "total": engine.total,
        "counts": dict(engine.counts),
        "anomalies": engine.anomaly_count,
        "avg_latency_ms": avg_lat,
        "resources": engine.resources(),
        "sim_running": get_simulator().is_running,
    })


@app.get("/api/devices")
def api_devices():
    engine = get_engine()
    devices = [{"src_ip": ip,
                "counts": dict(h["counts"]),
                "total": sum(h["counts"].values())}
               for ip, h in engine.device_history.items()]
    devices.sort(key=lambda d: -d["total"])
    return jsonify({"devices": devices[:200]})


@app.get("/api/device/<ip>")
def api_device(ip):
    engine = get_engine()
    hist = engine.device_history.get(ip)
    if not hist:
        return jsonify({"found": False, "src_ip": ip}), 404
    events = sorted(hist["events"], key=lambda r: r["ts"], reverse=True)
    return jsonify({"found": True, "src_ip": ip,
                    "counts": dict(hist["counts"]),
                    "total": sum(hist["counts"].values()),
                    "events": events[:100]})


# ---------------------------------------------------------------------------
# Static SPA
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


def create_app():
    return app


def main():
    ensure_data()
    engine = get_engine()
    if engine.is_deployed:
        print(f"[sentrymesh] deployed model loaded: {engine.bundle['winner_name']}")
    else:
        print("[sentrymesh] no deployed model yet -- open Model Lab and run 'Train Models'")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
