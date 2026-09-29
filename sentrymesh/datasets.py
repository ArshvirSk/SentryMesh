# SentryMesh dataset layer.
#
# Primary source : TON_IoT network telemetry (UNSW Canberra Cyber), train_test_network.csv
# Fallback source: CICIoT2023 merged CSV (Canadian Institute for Cybersecurity)
# Last resort    : deterministic synthetic IoT traffic generator (offline fallback)
#
# All loaders converge on one schema:
#   X : float32 DataFrame  -- numeric network-flow features (finite, imputed, standardised later)
#   y : string Series      -- one of CLASSES = ["Benign", "DoS", "Probing", "Authentication"]
#   meta : dict            -- provenance info surfaced in the UI / report
#
# Subsampling preserves the dataset's natural class imbalance (attack traffic is a
# minority of real traffic); we never oversample to balance classes.

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASSES = ["Benign", "DoS", "Probing", "Authentication"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# TON_IoT (train_test_network.csv)
# ---------------------------------------------------------------------------
# Official column map (UNSW Canberra Cyber, TON_IoT Networks dataset):
#   ts, src_ip, src_port, dst_ip, dst_port, proto, service, duration,
#   src_bytes, dst_bytes, conn_state, missed_bytes, src_pkts, src_ip_bytes,
#   dst_pkts, dst_ip_bytes, dns_* / http_* / ssl_* / ftp_* behaviours,
#   typed attitudes, label, type
TONIOT_NUMERIC = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qdcount", "dns_ancount", "dns_nscount", "dns_arcount",
    "dns_rejection", "http_request_body_len", "http_response_body_len",
    "http_status_code", "ssl_num_ciphers", "ftp_command_ret_code",
]
TONIOT_CATEGORICAL = ["proto", "service", "conn_state", "http_method"]

TONIOT_URL = "https://research.unsw.edu.au/sites/default/files/documents/TON_IoT-Datasets.zip"

# TON_IoT 'type' -> SentryMesh category
TONIOT_TYPE_MAP = {
    "normal": "Benign",
    "dos": "DoS",
    "ddos": "DoS",
    "scan": "Probing",
    "scanning": "Probing",
    "recon": "Probing",
    "reconnaissance": "Probing",
    "mitm": "Probing",
    "password": "Authentication",
    "bruteforce": "Authentication",
    "xss": "Benign",       # not one of SentryMesh's named categories -> mapped to background
    "injection": "Benign",
    "backdoor": "Benign",
    "ransomware": "Benign",
    "keylog": "Benign",
    "fingerprinting": "Benign",
}


def _toniot_derive_network_shape(df):
    """Derive flow-shape features TON_IoT does not provide directly."""
    out = pd.DataFrame(index=df.index)
    out["f_bytes_total"] = df["src_bytes"] + df["dst_bytes"]
    out["f_pkts_total"] = df["src_pkts"] + df["dst_pkts"]
    denom = out["f_pkts_total"].replace(0, np.nan)
    out["f_bytes_per_pkt"] = (out["f_bytes_total"] / denom).fillna(0.0)
    out["f_byte_ratio"] = df["dst_bytes"] / df["src_bytes"].replace(0, np.nan).fillna(1.0)
    out["f_pkt_ratio"] = df["dst_pkts"] / df["src_pkts"].replace(0, np.nan).fillna(1.0)
    out["f_inter_arrival"] = df["duration"] / denom.fillna(1.0)
    out["f_connectivity"] = df["conn_state"].map(
        df["conn_state"].value_counts(normalize=True).to_dict()
    ) if "conn_state" in df.columns else 0.0
    out["f_resp_bytes_distinct"] = df["dst_ip_bytes"]
    out["f_dst_pkts_total"] = df["dst_pkts"]
    out["f_src_bytes_max"] = df["src_bytes"]
    return out


def load_toniot(path, sample_rows=60000, seed=42):
    """Load TON_IoT train_test_network.csv, map labels, subsample preserving imbalance."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    if "type" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} does not look like TON_IoT (missing label/type columns)")

    keep = [c for c in TONIOT_NUMERIC if c in df.columns] + list(TONIOT_CATEGORICAL)
    meta_cols = {"src_ip": "src_ip", "dst_ip": "dst_ip", "proto": "proto"}
    df = df[[c for c in keep if c in df.columns] +
            [c for c in meta_cols if c in df.columns] +
            ["label", "type"]].copy()

    df["type_norm"] = df["type"].astype(str).str.strip().str.lower()
    df["category"] = df["type_norm"].map(TONIOT_TYPE_MAP).fillna("Benign")
    df.loc[df["label"] != 1, "category"] = "Benign"

    shape = _toniot_derive_network_shape(df)
    X_num = df[[c for c in TONIOT_NUMERIC if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([X_num, shape], axis=1).astype(np.float32)

    for c in TONIOT_CATEGORICAL:
        if c in df.columns:
            codes = df[c].astype(str).astype("category").cat.codes
            X[c + "_code"] = codes.astype(np.float32)

    y = df["category"]
    return _subsample(X, y, sample_rows, seed, meta=df[[c for c in meta_cols if c in df.columns]],
                      source="TON_IoT", raw_rows=len(df))


# ---------------------------------------------------------------------------
# CICIoT2023 (merged CSV)
# ---------------------------------------------------------------------------
CICIOT_URL = "https://cicresearch.ca/CICDataset/CICIoT2023/Dataset/CSVs/CICIoT2023.zip"

# CICIoT2023 label -> SentryMesh category
CICIOT_MAP = {
    "benignTraffic": "Benign",
    "ddos-icmp_flood": "DoS", "ddos-udp_flood": "DoS", "ddos-tcp_flood": "DoS",
    "ddos-psack_flood": "DoS", "ddos-syn_flood": "DoS", "ddos-synonymousack_flood": "DoS",
    "ddos-synonymous_flood": "DoS", "ddos-fin_flood": "DoS", "ddos-rst_flood": "DoS",
    "ddos-slowite": "DoS", "ddos-synflooding": "DoS", "ddos-http_flood": "DoS",
    "ddos-udp_fragmentation": "DoS", "ddos-icmp_fragmentation": "DoS",
    "ddos-ack_fragmentation": "DoS", "ddos-http_flood": "DoS",
    "dos-udp_flood": "DoS", "dos-syn_flood": "DoS", "dos-tcp_flood": "DoS",
    "dos-icmp_flood": "DoS", "dos-synonymous_flood": "DoS", "dos-synonymousack_flood": "DoS",
    "dos-ack_flood": "DoS", "dos-slowite": "DoS", "dos-slowhttptest": "DoS",
    "dos-synflooding": "DoS", "dos-http_flood": "DoS",
    "dos-udp_fragmentation": "DoS", "dos-icmp_fragmentation": "DoS",
    "dos-ack_fragmentation": "DoS",
    "recon-pingsweep": "Probing", "recon-xmasscan": "Probing", "recon-portscan": "Probing",
    "recon-sshscan": "Probing", "recon-hostsweep": "Probing", "recon-nil": "Probing",
    "mirai-ackflooding": "DoS", "mirai-udpplain": "DoS", "mirai-synonymousack_flood": "DoS",
    "mirai-greeth_flood": "DoS", "mirai-udppositional": "DoS",
    "bruteforce-web": "Authentication", "bruteforce-xss": "Authentication",
    "bruteforce-xml": "Authentication", "bruteforce-xxx": "Authentication",
    "sql_injection": "Benign", "commandinjection": "Benign",
    "uploading_attack": "Benign", "backdoor_malware": "Benign",
    "dns_spoofing": "Benign", "mitm-arpspoofing": "Benign",
    "vulnerabilityscan": "Probing",
}

CICIOT_NUMERIC = [
    "flow_duration", "header_size", "duration", "rate", "srate", "drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number", "ack_count",
    "syn_count", "rst_count", "fin_count", "udp", "tcp", "icmp", "arp", "ipv",
    "dhcp", "bootp", "totsum", "mag", "avgr", "avgsz", "iopheader",
    "spkts", "dpkts", "sbytes", "dbytes", "totbytes", "rate",
]


def load_ciciot(data_dir, sample_rows=60000, seed=42):
    """Download (if needed) CICIoT2023, map labels, subsample preserving imbalance."""
    zip_path = data_dir / "CICIoT2023.zip"
    extract_dir = data_dir / "CICIoT2023"
    sentinel = data_dir / ".ciciot_download_failed"
    csvs = sorted(extract_dir.glob("*.csv")) if extract_dir.exists() else []
    if not csvs and sentinel.exists():
        raise RuntimeError("previous CICIoT2023 download failed (sentinel present); "
                           "drop a partition CSV in data/ manually and delete the sentinel")
    if not csvs:
        data_dir.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(CICIOT_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_dir)
            sentinel.unlink(missing_ok=True)
        except Exception:
            sentinel.write_text("download failed")
            zip_path.unlink(missing_ok=True)
            raise
        csvs = sorted(extract_dir.glob("*.csv"))
    frames = []
    for csv in csvs:
        frames.append(pd.read_csv(csv, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError("CICIoT2023 CSVs do not contain a 'label' column")
    df["cat"] = df["label"].astype(str).str.strip().map(CICIOT_MAP).fillna("Benign")

    numeric = [c for c in CICIOT_NUMERIC if c in df.columns and df[c].dtype != object]
    if len(numeric) < 8:
        raise ValueError("CICIoT2023: too few numeric columns found")
    X = df[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X = X.loc[:, ~X.columns.duplicated()]
    if "proto" in df.columns:
        X["proto_code"] = df["proto"].astype(str).astype("category").cat.codes.astype(np.float32)

    y = df["cat"]
    return _subsample(X, y, sample_rows, seed, meta=df.get(["src_ip", "dst_ip"], None),
                      source="CICIoT2023", raw_rows=len(df))


# ---------------------------------------------------------------------------
# Synthetic fallback (deterministic, class-imbalance aware)
# ---------------------------------------------------------------------------
_SYNTH_PROFILE = {
    # class: (weight, feature means / offsets)
    "Benign": 0.72,
    "DoS": 0.16,
    "Probing": 0.08,
    "Authentication": 0.04,
}


def load_synthetic(n_rows=48000, seed=42, drift=0.0):
    """Deterministic synthetic IoT flow generator used when no real dataset is present.

    Class weights mimic TON_IoT's natural imbalance (benign majority, attack minority).
    A fixed RNG seed makes runs reproducible so training results are comparable.

    drift: 0.0 = the baseline regime; >0 perturbs the attack-class parameters
    (scale/shape shifts and benign-tail mixing) to simulate a distribution the
    model was not trained on -- used to build a separate cloud-train / fog-stream
    pair (the paper's KDDTrain+ / KDDTest+ separation).
    """
    rng = np.random.RandomState(seed)
    cats = list(_SYNTH_PROFILE.keys())
    weights = [_SYNTH_PROFILE[c] for c in cats]
    y = rng.choice(cats, size=n_rows, p=weights)

    n_benign = int((y == "Benign").sum())
    cols = {}
    # Base benign distributions
    cols["f_bytes_total"] = rng.gamma(3.0, 1200.0, n_rows)
    cols["f_pkts_total"] = rng.gamma(3.0, 20.0, n_rows)
    cols["f_duration"] = rng.exponential(2.0, n_rows)
    cols["f_bytes_per_pkt"] = cols["f_bytes_total"] / np.maximum(cols["f_pkts_total"], 1)
    cols["f_byte_ratio"] = rng.lognormal(0.0, 0.6, n_rows)
    cols["f_pkt_ratio"] = rng.lognormal(0.0, 0.5, n_rows)
    cols["f_inter_arrival"] = rng.exponential(0.01, n_rows)
    cols["f_connectivity"] = rng.normal(0.5, 0.15, n_rows)
    cols["f_resp_bytes_distinct"] = rng.gamma(2.5, 800.0, n_rows)
    cols["f_dst_pkts_total"] = cols["f_pkts_total"] * rng.uniform(0.6, 1.4, n_rows)
    cols["f_src_bytes_max"] = cols["f_bytes_total"] * rng.uniform(0.3, 1.0, n_rows)
    cols["proto_code"] = rng.choice([0, 1, 2], n_rows, p=[0.72, 0.24, 0.04])

    X = pd.DataFrame(cols, index=np.arange(n_rows))

    d = float(drift)

    # DoS: huge floods, near-zero duration, tiny packets, single proto burst
    dos = y == "DoS"
    X.loc[dos, "f_bytes_total"] = rng.gamma(8.0 * (1 + 0.4 * d), 9000.0 * (1 + 0.5 * d), int(dos.sum()))
    X.loc[dos, "f_pkts_total"] = rng.gamma(9.0 * (1 + 0.3 * d), 120.0 * (1 + 0.4 * d), int(dos.sum()))
    X.loc[dos, "f_duration"] = rng.exponential(0.02 * (1 + 2.0 * d), int(dos.sum()))
    X.loc[dos, "f_bytes_per_pkt"] = X.loc[dos, "f_bytes_total"] / np.maximum(X.loc[dos, "f_pkts_total"], 1)
    X.loc[dos, "f_pkt_ratio"] = rng.lognormal(1.2 * (1 + 0.25 * d), 0.3 * (1 + 0.5 * d), int(dos.sum()))
    X.loc[dos, "f_inter_arrival"] = rng.exponential(0.0002 * (1 + 3.0 * d), int(dos.sum()))
    X.loc[dos, "proto_code"] = rng.choice([1, 0], int(dos.sum()), p=[0.92 - 0.15 * d, 0.08 + 0.15 * d])

    # Probing: many small connections, tiny payloads, low duration
    prb = y == "Probing"
    X.loc[prb, "f_bytes_total"] = rng.gamma(2.0 * (1 + 0.5 * d), 90.0 * (1 + 0.6 * d), int(prb.sum()))
    X.loc[prb, "f_pkts_total"] = rng.gamma(2.5 * (1 + 0.4 * d), 6.0 * (1 + 0.5 * d), int(prb.sum()))
    X.loc[prb, "f_duration"] = rng.exponential(0.4 * (1 + 1.5 * d), int(prb.sum()))
    X.loc[prb, "f_bytes_per_pkt"] = X.loc[prb, "f_bytes_total"] / np.maximum(X.loc[prb, "f_pkts_total"], 1)
    X.loc[prb, "f_byte_ratio"] = rng.lognormal(-0.4 * (1 - 0.5 * d), 0.3 * (1 + 0.6 * d), int(prb.sum()))
    X.loc[prb, "f_inter_arrival"] = rng.exponential(0.004 * (1 + 2.0 * d), int(prb.sum()))
    X.loc[prb, "proto_code"] = rng.choice([0, 2], int(prb.sum()), p=[0.7 - 0.2 * d, 0.3 + 0.2 * d])

    # Authentication: few packets, moderate bytes, distinctive byte/pkt ratios
    aut = y == "Authentication"
    X.loc[aut, "f_bytes_total"] = rng.gamma(3.5 * (1 - 0.3 * d), 500.0 * (1 + 0.7 * d), int(aut.sum()))
    X.loc[aut, "f_pkts_total"] = rng.gamma(3.0 * (1 - 0.25 * d), 9.0 * (1 + 0.6 * d), int(aut.sum()))
    X.loc[aut, "f_duration"] = rng.exponential(1.8 * (1 - 0.4 * d), int(aut.sum()))
    X.loc[aut, "f_bytes_per_pkt"] = X.loc[aut, "f_bytes_total"] / np.maximum(X.loc[aut, "f_pkts_total"], 1)
    X.loc[aut, "f_byte_ratio"] = rng.lognormal(0.5 * (1 - 0.6 * d), 0.35 * (1 + 0.8 * d), int(aut.sum()))
    X.loc[aut, "f_inter_arrival"] = rng.exponential(0.02 * (1 - 0.5 * d), int(aut.sum()))
    X.loc[aut, "proto_code"] = rng.choice([0, 2], int(aut.sum()), p=[0.5 + 0.2 * d, 0.5 - 0.2 * d])

    # Drift also blurs the benign tail toward attack-like volumes, the way real
    # traffic shifts (bulk backups, firmware updates) look attack-ish.
    if d > 0:
        ben = y == "Benign"
        n_ben = int(ben.sum())
        X.loc[ben, "f_bytes_total"] *= rng.lognormal(0.0, 0.5 * d, n_ben)
        X.loc[ben, "f_pkts_total"] *= rng.lognormal(0.0, 0.4 * d, n_ben)
        X.loc[ben, "f_duration"] *= rng.lognormal(0.0, 0.6 * d, n_ben)

    X = X.astype(np.float32)

    # Device/IP labels for the simulator + investigate view
    rng_ips = np.random.RandomState(seed + 1)
    octets = [f"10.0.{rng_ips.randint(1, 6)}.{rng_ips.randint(2, 200)}" for _ in range(n_rows)]
    meta = pd.DataFrame({"src_ip": octets, "proto": X["proto_code"].map({0: "TCP", 1: "UDP", 2: "ICMP"})})
    return X.reset_index(drop=True), pd.Series(y, name="category"), meta, {
        "source": "Synthetic (deterministic fallback)" + (f" · drifted regime d={drift}" if drift else ""),
        "raw_rows": n_rows,
        "class_dist": pd.Series(y).value_counts().to_dict(),
        "drift": float(drift),
    }


# ---------------------------------------------------------------------------
# Subsampling + validation
# ---------------------------------------------------------------------------
def _subsample(X, y, sample_rows, seed, meta=None, source="?", raw_rows=0):
    """Random subsample that preserves the dataset's natural class proportions."""
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    if meta is not None and not meta.empty:
        meta = meta[mask].reset_index(drop=True)
    else:
        meta = pd.DataFrame({"src_ip": ["unknown"] * len(X)})

    n = min(sample_rows, len(X))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))
    X, y, meta = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True), meta.iloc[idx].reset_index(drop=True)

    X = X.fillna(X.median(numeric_only=True)).astype(np.float32)
    class_dist = y.value_counts().to_dict()
    return X, y, meta, {
        "source": source, "raw_rows": int(raw_rows), "sampled_rows": int(n),
        "class_dist": class_dist, "n_features": X.shape[1],
        "columns": list(X.columns), "seed": seed,
    }


def load_dataset_pair(sample_rows=60000, stream_rows=60000, seed=42, drift=0.25,
                      prefer=None, data_dir=None):
    """Separate cloud-train and fog-stream distributions (the paper's
    KDDTrain+ / KDDTest+ separation, Section 4.3).

    Synthetic : train on the baseline regime, stream a *drifted* regime (different
    RNG seed + perturbed attack parameters) so live data is genuinely unseen.
    Real data : two disjoint row-disjoint subsamples of the source CSVs (half the
    pool each); drift does not apply.

    Returns (X_tr, y_tr, meta_tr, info_tr, X_st, y_st, meta_st, info_st).
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    real = None
    for src in ([prefer] if prefer in ("toniot", "ciciot") else ["toniot", "ciciot"]):
        try:
            if src == "toniot":
                candidates = list(data_dir.glob("**/train_test_network.csv")) + \
                             list(data_dir.glob("**/*train_test_network*.csv"))
                if candidates:
                    real = load_toniot(candidates[0], sample_rows + stream_rows, seed)
                    break
            elif src == "ciciot":
                real = load_ciciot(data_dir, sample_rows + stream_rows, seed)
                break
        except Exception as exc:
            print(f"[datasets] pair/{src} failed: {exc}")

    if real is not None:
        X, y, meta, info = real
        cut = min(sample_rows, len(X) // 2)
        X_tr, y_tr, meta_tr = X.iloc[:cut], y.iloc[:cut], meta.iloc[:cut]
        X_st, y_st, meta_st = X.iloc[cut:cut + stream_rows], y.iloc[cut:cut + stream_rows], meta.iloc[cut:cut + stream_rows]
        info_tr = dict(info, source=info["source"] + " (cloud-train half)", sampled_rows=len(X_tr))
        info_st = dict(info, source=info["source"] + " (fog-stream half, disjoint)", sampled_rows=len(X_st))
        return (X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), meta_tr.reset_index(drop=True), info_tr,
                X_st.reset_index(drop=True), y_st.reset_index(drop=True), meta_st.reset_index(drop=True), info_st)

    X_tr, y_tr, meta_tr, info_tr = load_synthetic(sample_rows, seed=seed, drift=0.0)
    X_st, y_st, meta_st, info_st = load_synthetic(stream_rows, seed=seed + 777, drift=drift)
    return X_tr, y_tr, meta_tr, info_tr, X_st, y_st, meta_st, info_st


def load_dataset(sample_rows=60000, seed=42, prefer=None, data_dir=None):
    """Try TON_IoT, then CICIoT2023, then synthetic. Returns (X, y, meta, info)."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    order = [prefer] if prefer in ("toniot", "ciciot", "synthetic") else ["toniot", "ciciot", "synthetic"]

    for src in order:
        try:
            if src == "toniot":
                candidates = list(data_dir.glob("**/train_test_network.csv")) + \
                             list(data_dir.glob("**/*train_test_network*.csv"))
                if candidates:
                    return load_toniot(candidates[0], sample_rows, seed)
            elif src == "ciciot":
                return load_ciciot(data_dir, sample_rows, seed)
            elif src == "synthetic":
                return load_synthetic(sample_rows, seed)
        except Exception as exc:  # fall through to the next source
            print(f"[datasets] {src} failed: {exc}")

    # Guaranteed fallback
    return load_synthetic(sample_rows, seed)


def feature_sha(X):
    """Stable fingerprint of the feature matrix so model artifacts are traceable."""
    h = hashlib.sha256()
    h.update(",".join(X.columns).encode())
    arr = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    h.update(arr.tobytes())
    return h.hexdigest()[:16]
