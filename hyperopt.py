"""
Hyper‑parameter optimisation utilities using Optuna.

Currently includes optimisation for Random Forest and Extra Trees. Extend the
`objective_*` functions or add new ones for other model families as needed.
"""
from __future__ import annotations
from typing import List

import optuna
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
from config import MODEL_DIR, TARGET_COLUMN, RANDOM_STATE
from metrics import evaluate, print_metrics   # optional: print at the end


# ─────────────────────────── Shared Pre‑processor ────────────────────────────
def _preprocessor(cat_features: List[str], num_features: List[str]):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )


# ─────────────────── Random Forest optimisation objective ────────────────────
def optimise_rf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    n_trials: int = 25, # increase to 25
):
    """Optuna search for the best Random‑Forest hyper‑parameters."""
    # ensure clean split between numeric & categorical
    cat_features = list(set(cat_features) | {c for c in features if train_df[c].dtype == "O"})
    num_features = [f for f in features if f not in cat_features]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 40, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        pipe = Pipeline(
            [("pre", _preprocessor(cat_features, num_features)), ("model", model)]
        )

        ts_cv = TimeSeriesSplit(n_splits=3)
        mae_scores = []
        for train_idx, val_idx in ts_cv.split(train_df):
            X_tr, X_val = train_df.iloc[train_idx][features], train_df.iloc[val_idx][features]
            y_tr, y_val = train_df.iloc[train_idx][TARGET_COLUMN], train_df.iloc[val_idx][TARGET_COLUMN]
            pipe.fit(X_tr, y_tr)
            mae_scores.append(abs(pipe.predict(X_val) - y_val).mean())
        return np.mean(mae_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best RF params:", study.best_params)

    # fit best model on full training set
    best_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **study.best_params)
    best_pipe = Pipeline([("pre", _preprocessor(cat_features, num_features)), ("model", best_model)])
    best_pipe.fit(train_df[features], train_df[TARGET_COLUMN])
    joblib.dump(best_pipe, MODEL_DIR / "opt_rf_model.joblib")

    # validation metrics
    y_pred = best_pipe.predict(val_df[features])
    metrics = evaluate(val_df[TARGET_COLUMN], y_pred, prefix="opt_rf_")
    print_metrics(metrics)
    return metrics


# ───────────────── ExtraTrees optimisation objective ────────────────────────
def optimise_et(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
    n_trials: int = 25, #increase to 25
):
    num_features = [f for f in features if f not in cat_features]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 5, 40, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        pipe = Pipeline(
            [("pre", _preprocessor(cat_features, num_features)), ("model", model)]
        )

        ts_cv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in ts_cv.split(train_df):
            X_tr, X_val = (
                train_df.iloc[train_idx][features],
                train_df.iloc[val_idx][features],
            )
            y_tr, y_val = (
                train_df.iloc[train_idx][TARGET_COLUMN],
                train_df.iloc[val_idx][TARGET_COLUMN],
            )
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            scores.append(abs(preds - y_val).mean())  # MAE
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best ET params:", study.best_params)

    best_model = ExtraTreesRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **study.best_params
    )
    pipe = Pipeline(
        [("pre", _preprocessor(cat_features, num_features)), ("model", best_model)]
    )
    pipe.fit(train_df[features], train_df[TARGET_COLUMN])
    joblib.dump(pipe, MODEL_DIR / "opt_et_model.joblib")

    y_pred = pipe.predict(val_df[features])
    metrics = evaluate(val_df[TARGET_COLUMN], y_pred, prefix="opt_et_")
    print_metrics(metrics)
    return metrics
