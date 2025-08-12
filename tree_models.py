"""Random Forest and Extra Trees models with time‑series CV."""
from __future__ import annotations
from typing import List, Dict
import joblib
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from config import MODEL_DIR, RANDOM_STATE, RF_PARAM_GRID, ET_PARAM_GRID, TARGET_COLUMN
from metrics import evaluate, print_metrics

# ──────────────────────────────── Pipeline ─────────────────────────────────────

def _build_pipeline(cat_features: List[str], num_features: List[str], model):
    transformers = [
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("pre", preprocessor), ("model", model)])

# ───────────────────────────────‑ Training ─────────────────────────────────────

def train_tree_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    model_type: str = "rf",
) -> Dict[str, float]:
    """
    Fit a Random‑Forest or Extra‑Trees model with time‑series CV and save it.

    *Any non‑numeric columns that slipped into `features` are automatically
    moved to `cat_features`, so you never hit the “could not convert string
    to float” error.*
    """
    # --- make sure no stray text columns sit in the numeric list -------------
    cat_features = list(set(cat_features) | {c for c in features if train_df[c].dtype == "O"})
    num_features = [c for c in features if c not in cat_features]

    # pick estimator & param grid
    if model_type == "rf":
        estimator, param_grid = (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            RF_PARAM_GRID,
        )
    elif model_type == "et":
        estimator, param_grid = (
            ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            ET_PARAM_GRID,
        )
    else:
        raise ValueError("model_type must be 'rf' or 'et'")

    pipe = _build_pipeline(cat_features, num_features, estimator)

    ts_cv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipe,
        param_grid,
        n_iter=25, # <-- for quick testing, increase for real runs - Set to 25
        cv=ts_cv,
        scoring="neg_mean_absolute_error",
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",        # <-- easier debugging if something blows up
    )

    X_train, y_train = train_df[features], train_df[TARGET_COLUMN]
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    joblib.dump(best_model, MODEL_DIR / f"{model_type}_model.joblib")

    y_pred_val = best_model.predict(val_df[features])
    metrics = evaluate(val_df[TARGET_COLUMN], y_pred_val, prefix=f"{model_type}_")
    print_metrics(metrics)
    return metrics


