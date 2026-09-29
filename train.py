#!/usr/bin/env python
"""Standalone cloud-role training run (writes artifacts/deployed_model.joblib +
artifacts/training_result.json). Mirrors exactly what the UI 'Train Models'
button triggers via POST /api/train."""
import warnings
import time

warnings.filterwarnings("ignore")

from sentrymesh.datasets import load_synthetic
from sentrymesh.cloud import run_training

X, y, meta, info = load_synthetic(n_rows=45000, seed=42)
print("dataset:", X.shape, info["class_dist"], flush=True)
t0 = time.time()
res = run_training(X, y, kfold=5)
print(f"training finished in {time.time() - t0:.1f}s", flush=True)
print("winner:", res["winner"]["name"], "| variant:", res["deployed_variant"])
