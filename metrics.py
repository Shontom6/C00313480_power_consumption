"""Evaluation metrics and convenience printers shared by all models."""
from __future__ import annotations
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ────────────────────────────────── Metrics ────────────────────────────────────

def evaluate(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {f"{prefix}MAE": mae, f"{prefix}RMSE": rmse,
            f"{prefix}MAPE": mape, f"{prefix}R2": r2}


def print_metrics(metrics: Dict[str, float]):
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
# metrics.py
