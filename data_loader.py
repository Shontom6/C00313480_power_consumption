"""Data ingest, cleaning and splitting utilities."""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Tuple

from config import DATE_COLUMN, TARGET_COLUMN, RAW_DATA_PATH, RANDOM_STATE


# ──────────────────────────────────── Loading ──────────────────────────────────

def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the CSV, parse the date column and guarantee a DateTimeIndex."""
    df = pd.read_csv(path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.set_index(DATE_COLUMN).sort_index()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # split numeric vs others
    num_cols = df.select_dtypes(include="number").columns
    other_cols = df.columns.difference(num_cols)

    # aggregate
    df_num   = df[num_cols].groupby(df.index).mean()
    df_other = df[other_cols].groupby(df.index).first()   # or .mode().iloc[0]

    df = pd.concat([df_num, df_other], axis=1).sort_index()

    # hourly grid + interpolation
    df = df.asfreq("H")
    df[num_cols] = df[num_cols].interpolate("time")
    return df.dropna(subset=[TARGET_COLUMN])



def load_and_clean(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


# ───────────────────────────── Train/Val/Test Split ────────────────────────────

def train_val_test_split(df: pd.DataFrame, test_fraction: float = 0.2,
                         val_fraction: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split so future data never leaks into the past."""
    n_total = len(df)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    train = df.iloc[: -(n_val + n_test)]
    val = df.iloc[-(n_val + n_test): -n_test]
    test = df.iloc[-n_test:]
    return train, val, test
