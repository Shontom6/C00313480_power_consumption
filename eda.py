"""Exploratory Data Analysis utilities.

Call these helpers from a notebook to quickly understand patterns in the raw
signal before committing to modelling.
"""
from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config import TARGET_COLUMN

# ────────────────────────────── Quick Summary ─────────────────────────────────

def summary(df: pd.DataFrame):
    """Print basic stats & missing‑value overview."""
    print("\n===== DATA SHAPE =====")
    print(df.shape)
    print("\n===== MISSING VALUES (top 20) =====")
    print(df.isna().sum().sort_values(ascending=False).head(20))
    print("\n===== DESCRIPTIVE STATS (target) =====")
    print(df[TARGET_COLUMN].describe())

# ─────────────────────────── Time‑Series Visuals ──────────────────────────────

def plot_series(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None,
                column: str = TARGET_COLUMN):
    """Line plot for a date‑range with optional zoom."""
    df[column].loc[start:end].plot(figsize=(12, 4))
    plt.title(f"{column} over time")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


def stl_decompose(df: pd.DataFrame, period: int = 24, column: str = TARGET_COLUMN):
    """Seasonal‑Trend decomposition via Loess (STL)."""
    stl = STL(df[column], period=period)
    res = stl.fit()
    res.plot()
    plt.tight_layout()
    plt.show()


def autocorrelation_plots(df: pd.DataFrame, lags: int = 48, column: str = TARGET_COLUMN):
    """ACF & PACF up to *lags* steps."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df[column], ax=axes[0], lags=lags)
    plot_pacf(df[column], ax=axes[1], lags=lags)
    axes[0].set_title("Autocorrelation (ACF)")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    plt.show()

# ───────────────────── Weather vs. Demand Correlations ────────────────────────

def weather_correlation_heatmap(df: pd.DataFrame, weather_cols: list[str]):
    """Heatmap of Pearson correlations between weather vars & demand."""
    corr = df[[TARGET_COLUMN] + weather_cols].corr()
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.index)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Matrix: Weather vs Demand")
    plt.tight_layout()
    plt.show()
