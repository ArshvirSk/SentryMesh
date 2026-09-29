# SentryMesh fog/edge role: real-time inference with the single deployed model.
#
# What lives here (and nowhere else): the winning model artifact, the traffic
# simulator that replays dataset rows as a live stream, the rolling-window
# DBSCAN anomaly backstop (Module 5) for novel/zero-day traffic, and the
# device-history index backing the Investigate page.

import threading
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .datasets import CLASSES

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "deployed_model.joblib"

FEED_LEN = 200           # records kept for the live feed
WINDOW_MAX = 1500        # rolling window size for the DBSCAN backstop
DBSCAN_EPS = 1.8
DBSCAN_MIN_SAMPLES = 8
BACKSTOP_EVERY = 25       # re-fit the backstop every N records (keeps latency honest)
ANOMALY_LABEL = "ANOMALOUS"   # distinct from the three named supervised categories

# Real-time pacing of the simulated gateway stream
STREAM_MIN_DELAY = 0.05  # seconds between records (tunable from the UI)


class EdgeEngine:
    """Holds the deployed model + rolling DBSCAN window; classifies one row at a time."""

    def __init__(self):
        self.bundle = None
        self.pipeline = None
        self.window = deque(maxlen=WINDOW_MAX)
        self.lock = threading.Lock()
        self.latencies = deque(maxlen=500)
        self.counts = Counter()
        self.anomaly_count = 0
        self.total = 0
        self.started_at = None
        self.device_history = {}
        self._window_scaler = StandardScaler()  # DBSCAN needs comparable feature scales
        self._scaler_ready = False
        self._since_backstop = 0
        self._ts_lock = threading.Lock()
        self.ts_buckets = []          # [{ts, classes: {class: n}, total: n}] 2s buckets
        self._last_bucket = None
        self._last_labels = None      # most recent DBSCAN assignment for window_viz
        # Fog-node resource instrumentation (paper Sec. 5.2.3 reports "<10% CPU")
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)               # prime the counter
        self._res_lock = threading.Lock()
        self._res = {"cpu_pct": 0.0, "mem_mb": 0.0, "threads": 0}
        self._res_thread = threading.Thread(target=self._resource_loop, daemon=True)
        self._res_thread.start()
        self.load()

    def _resource_loop(self):
        """Sample this process's CPU/memory every 2 s (paper Sec. 5.2.3)."""
        while True:
            try:
                cpu = self._proc.cpu_percent(interval=None) / max(psutil.cpu_count(logical=True), 1)
                mem = self._proc.memory_info().rss / (1024 * 1024)
                with self._res_lock:
                    self._res = {"cpu_pct": round(cpu, 2), "mem_mb": round(mem, 1),
                                 "threads": self._proc.num_threads()}
            except Exception:
                pass
            time.sleep(2)

    def resources(self):
        with self._res_lock:
            return dict(self._res)

    # -- deployment ---------------------------------------------------------
    def load(self):
        if ARTIFACT.exists():
            self.bundle = joblib.load(ARTIFACT)
            self.pipeline = self.bundle["pipeline"]
            self.window.clear()
            self._scaler_ready = False
            return True
        return False

    @property
    def is_deployed(self):
        return self.bundle is not None

    def reset(self):
        """Clear all session state: counters, feed, device history, DBSCAN window."""
        with self.lock:
            self.window.clear()
            self._scaler_ready = False
            self._since_backstop = 0
        self.latencies.clear()
        self.counts.clear()
        self.anomaly_count = 0
        self.total = 0
        self.started_at = None
        self.device_history.clear()
        with self._ts_lock:
            self.ts_buckets.clear()
        self._last_bucket = None
        self._last_labels = None

    # -- classification -----------------------------------------------------
    def classify(self, row, src_ip=None, proto=None):
        """Run one network-flow record through the deployed model + DBSCAN backstop."""
        if not self.is_deployed:
            raise RuntimeError("no model deployed -- train models first (cloud role)")
        X = row.drop(labels=["src_ip", "proto"], errors="ignore").to_frame().T
        cols = self.bundle["trained_on"]["columns"]
        X = X.reindex(columns=cols, fill_value=0.0).astype(np.float32)

        t0 = time.perf_counter()
        proba = self.pipeline.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred = CLASSES[pred_idx]
        conf = float(proba[pred_idx])

        # Module 5: density-based anomaly backstop over a standardised rolling
        # window (raw byte counts would otherwise dominate the distance metric).
        # Re-fits every BACKSTOP_EVERY records so per-record latency stays
        # dominated by the supervised model; intermediate records inherit the
        # last check, trading a little anomaly granularity for throughput.
        anomalous = False
        with self.lock:
            self.window.append(X.iloc[0].to_numpy())
            self._since_backstop += 1
            if len(self.window) >= 40 and self._since_backstop >= BACKSTOP_EVERY:
                self._since_backstop = 0
                W = np.vstack(self.window)
                if not self._scaler_ready:
                    self._window_scaler.fit(W)
                    self._scaler_ready = True
                Ws = self._window_scaler.transform(W)
                db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(Ws)
                self._last_labels = db.labels_.copy()
                anomalous = bool(db.labels_[-1] == -1)

        lat_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies.append(lat_ms)

        rec_src = src_ip if src_ip is not None else str(row.get("src_ip", "unknown"))
        rec_proto = proto if proto is not None else (str(row["proto"]) if "proto" in row.index else "-")
        record = {
            "ts": time.time(),
            "src_ip": rec_src,
            "proto": rec_proto,
            "pred": pred,
            "confidence": round(conf, 4),
            "anomalous": anomalous,
            "latency_ms": round(lat_ms, 3),
        }
        return record

    def record_event(self, rec):
        """Update counters, device history, and feed after a classification."""
        self.counts[rec["pred"]] += 1
        if rec.get("anomalous"):
            self.anomaly_count += 1
            self.counts[ANOMALY_LABEL] += 1
        self.total += 1
        if len(self.device_history) > 5000:
            for k in list(self.device_history)[:1600]:   # keep memory bounded
                self.device_history.pop(k, None)
        hist = self.device_history.setdefault(rec["src_ip"], {"counts": Counter(), "events": deque(maxlen=200)})
        hist["counts"][rec["pred"]] += 1
        if rec.get("anomalous"):
            hist["counts"][ANOMALY_LABEL] += 1
        hist["events"].append(rec)
        # 2-second time buckets for the Overview charts
        with self._ts_lock:
            b = int(rec["ts"] // 2)
            if self._last_bucket is None or b > self._last_bucket:
                self.ts_buckets.append({"b": b, "classes": Counter(), "total": 0})
                self._last_bucket = b
                if len(self.ts_buckets) > 900:      # ~30 min of history
                    self.ts_buckets = self.ts_buckets[-600:]
            bucket = self.ts_buckets[-1]
            bucket["classes"][rec["pred"]] += 1
            if rec.get("anomalous"):
                bucket["classes"][ANOMALY_LABEL] += 1
            bucket["total"] += 1

    def timeseries(self, since=None):
        """Per-class time buckets since a bucket timestamp (or all)."""
        with self._ts_lock:
            rows = list(self.ts_buckets)
        if since is not None:
            rows = [r for r in rows if r["b"] >= since]
        return [{"t": r["b"] * 2, "total": r["total"],
                 "classes": dict(r["classes"])} for r in rows]

    def window_viz(self, max_points=600):
        """Viz-only 2D projection of the rolling DBSCAN window.
        Independent of any deployment PCA: fit a fresh 2-component PCA on the
        standardised window and attach the most recent DBSCAN labels."""
        with self.lock:
            W = np.vstack(self.window) if len(self.window) >= 10 else None
            labels = None if self._last_labels is None or len(self._last_labels) != len(self.window) \
                else self._last_labels.copy()
        if W is None:
            return {"points": [], "ready": False}
        Ws = self._window_scaler.transform(W) if self._scaler_ready else StandardScaler().fit_transform(W)
        pts = PCA(n_components=2, random_state=42).fit_transform(Ws)
        if len(pts) > max_points:                       # keep the newest records
            pts = pts[-max_points:]
            labels = labels[-max_points:] if labels is not None else None
        step = max(1, len(pts) // max_points)
        pts = pts[::step][-max_points:]
        labels = labels[::step][-max_points:] if labels is not None else None
        return {
            "ready": True,
            "n_window": int(len(self.window)),
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_SAMPLES,
            "labels_stale": labels is None,
            "points": [{"x": round(float(p[0]), 3), "y": round(float(p[1]), 3),
                        "c": (int(l) if l is not None and l == l else -2)}
                       for p, l in zip(pts, labels if labels is not None else [None] * len(pts))],
        }


class Simulator:
    """Replays dataset rows as a live IoT gateway stream on a background thread."""

    def __init__(self, engine: EdgeEngine):
        self.engine = engine
        self.thread = None
        self.running = False
        self.delay = STREAM_MIN_DELAY
        self.X = None
        self.meta = None
        self.cursor = 0
        self.lock = threading.Lock()

    def load_data(self, X, meta):
        self.X = X.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True) if meta is not None else None

    def start(self, delay=None):
        if not self.engine.is_deployed:
            raise RuntimeError("no deployed model -- run cloud training first")
        if delay:
            self.delay = float(delay)
        if self.running:
            return
        self.running = True
        if self.engine.started_at is None:
            self.engine.started_at = time.time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_speed(self, rps):
        """Set replay rate in records/second (UI speed presets)."""
        self.delay = max(0.0, 1.0 / float(rps))

    def reset(self):
        """Stop the stream and clear all edge-tier session state."""
        self.stop()
        self.cursor = 0
        self.engine.reset()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        self.thread = None

    @property
    def is_running(self):
        return self.running

    def _loop(self):
        while self.running:
            with self.lock:
                if self.X is None or len(self.X) == 0:
                    time.sleep(0.2)
                    continue
                i = self.cursor % len(self.X)
                row = self.X.iloc[i].copy()
                src_ip = proto = None
                if self.meta is not None:
                    if "src_ip" in self.meta.columns:
                        src_ip = str(self.meta.iloc[i]["src_ip"])
                        row["src_ip"] = src_ip
                    if "proto" in self.meta.columns:
                        proto = str(self.meta.iloc[i]["proto"])
                self.cursor += 1
            try:
                rec = self.engine.classify(row, src_ip=src_ip, proto=proto)
                self.engine.record_event(rec)
            except Exception as exc:
                print(f"[edge] classify error: {exc}")
            time.sleep(self.delay)


# ---------------------------------------------------------------------------
# App-level singleton
# ---------------------------------------------------------------------------
_state = {"engine": None, "sim": None, "X": None, "y": None, "meta": None, "info": None}


def get_engine():
    if _state["engine"] is None:
        _state["engine"] = EdgeEngine()
    return _state["engine"]


def get_simulator():
    if _state["sim"] is None:
        _state["sim"] = Simulator(get_engine())
    return _state["sim"]


def set_stream_data(X, y, meta, info):
    _state["X"], _state["y"], _state["meta"], _state["info"] = X, y, meta, info


def get_stream_data():
    return _state["X"], _state["y"], _state["meta"], _state["info"]
