"""Feature engineering helpers: time‑based, lagged and categorical encodings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import TARGET_COLUMN, LAG_STEPS

# ────────────────────────────── Time‑Based Features ────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    return df

# ───────────────────────────────── Lag Features ────────────────────────────────

def add_lag_features(df: pd.DataFrame, lags: int | list[int] = LAG_STEPS,
                     target: str = TARGET_COLUMN) -> pd.DataFrame:
    df = df.copy()
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df.dropna()
