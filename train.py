"""
CLI entry‑point: EDA, model training, hyper‑opt and XAI.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from joblib import load
from metrics import evaluate, print_metrics
from config import TARGET_COLUMN, LAG_STEPS,MODEL_DIR,LSTM_CONFIG,MODEL_DIR  
from data_loader import load_and_clean, train_val_test_split
from feature_engineering import add_time_features, add_lag_features
from tree_models import train_tree_model
from xai_utils import tft_interpret_global_v3
from dl_models import train_lstm, train_tft
from dl_models import evaluate_lstm_on_test
from dl_models import evaluate_tft_on_test
from xai_utils import shap_tree_summary, permutation_imp_plot
from eda import (
    summary as eda_summary,
    plot_series,
    stl_decompose,
    autocorrelation_plots,
    weather_correlation_heatmap,
)
from hyperopt import optimise_rf, optimise_et


def evaluate_saved_model_on_test(model_type: str, test_df, feature_cols, target_col):
    """
    Load the saved RF/ET pipeline and print final test metrics.
    model_type: "rf" or "et"
    """
    model_path = MODEL_DIR / f"{model_type}_model.joblib"
    model = load(model_path)
    y_pred = model.predict(test_df[feature_cols])
    metrics = evaluate(test_df[TARGET_COLUMN], y_pred, prefix=f"{model_type}_test_")
    print_metrics(metrics)
    
# ─────────────────────────────── Main Pipeline ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Energy Demand Prediction Pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        help="Subset: all | eda | rf | et | lstm | tft | xai | opt_rf | opt_et",
    )
    args = parser.parse_args()

    # --- Data ingest & feature engineering
    df = load_and_clean()
    df = add_time_features(df)
    df = add_lag_features(df, LAG_STEPS)

    # Encode weather description via one‑hot
    df["weather_desc"] = df["description"].astype("category")
    df = pd.get_dummies(df, columns=["weather_desc"], drop_first=True)

    # Extra columns for TFT
    df["time_idx"] = np.arange(len(df))
    df["all"] = 0  # single household group

    # Feature sets
    # 1) make sure no raw text columns survive
    df = df.drop(columns=df.select_dtypes(include="object").columns)

    # 2) rebuild feature lists
    cat_features = [c for c in df.columns if c.startswith("weather_desc_")]
    num_features = df.select_dtypes(include="number").columns.difference(
        cat_features + [TARGET_COLUMN, "time_idx", "all"]
    )
    feature_cols = num_features.tolist() + cat_features

    train_df, val_df,test_df = train_val_test_split(df)  # test_df unused here

    # --- EDA
    if args.stage in ("all", "eda"):
        eda_summary(df)
        plot_series(df)
        stl_decompose(df)
        autocorrelation_plots(df)
        weather_cols = [
            c
            for c in df.columns
            if any(prefix in c for prefix in ["temp", "humidity", "pressure", "speed"])
        ]
        weather_correlation_heatmap(df, weather_cols)

    # --- Model training
    if args.stage in ("all", "rf"):
        train_tree_model(train_df, val_df, feature_cols, cat_features, model_type="rf")
        print("Evaluating saved RF model on test set...")
        evaluate_saved_model_on_test("rf", test_df, feature_cols, TARGET_COLUMN)
    if args.stage in ("all", "et"):
        train_tree_model(train_df, val_df, feature_cols, cat_features, model_type="et")
        print("Evaluating saved ET model on test set...")
        evaluate_saved_model_on_test("et", test_df, feature_cols, TARGET_COLUMN)
    if args.stage in ("all", "lstm"):
        train_lstm(train_df, val_df, feature_cols)
        # NEW: final test evaluation
        print("Evaluating saved LSTM model on test set...")
        evaluate_lstm_on_test(test_df, feature_cols)

    if args.stage in ("all", "tft"):
        print("train tft")
        train_tft(train_df, val_df, feature_cols)
        print("train tft completed")
        print("Evaluating saved TFT model on test set...")
        evaluate_tft_on_test(train_df, test_df, feature_cols)
    # --- Hyper‑opt
    if args.stage in ("all", "opt_rf"):
        optimise_rf(train_df, val_df, feature_cols, cat_features)
        print("Evaluating saved Opt_RF model on test set...")
        evaluate_saved_model_on_test("opt_rf", test_df, feature_cols, TARGET_COLUMN)
    if args.stage in ("all", "opt_et"):
        optimise_et(train_df, val_df, feature_cols, cat_features)
        print("Evaluating saved Opt_et model on test set...")
        evaluate_saved_model_on_test("opt_et", test_df, feature_cols, TARGET_COLUMN)
    



    print("Pipeline completed successfully!")
    print("Starting XAI...")
    # ───────────────────────────── XAI ────────────────────────────────

    # --- XAI
    
    if args.stage in ("all", "xai"):
        import joblib
        import torch
        from dl_models import EnergyLSTM
        from xai_utils import shap_tree_summary, permutation_imp_plot, shap_lstm_summary

    # ───── SHAP for Random Forest ─────
        try:
            rf_model = joblib.load(MODEL_DIR / "rf_model.joblib")
            print("SHAP: Random Forest")
            shap_tree_summary(rf_model, val_df[feature_cols].iloc[:500], feature_cols)
            permutation_imp_plot(rf_model, val_df[feature_cols], val_df[TARGET_COLUMN])
        except Exception as e:
            print(f"[WARN] RF SHAP failed: {e}")

    # ───── SHAP for Extra Trees ─────
        try:
            et_model = joblib.load(MODEL_DIR / "et_model.joblib")
            print("SHAP: Extra Trees")
            shap_tree_summary(et_model, val_df[feature_cols].iloc[:500], feature_cols)
            permutation_imp_plot(et_model, val_df[feature_cols], val_df[TARGET_COLUMN])
        except Exception as e:
            print(f"[WARN] ET SHAP failed: {e}")

    # ───── SHAP for LSTM ─────
        try:
            lstm_model = EnergyLSTM(n_features=len(feature_cols), cfg=LSTM_CONFIG)
            lstm_model.load_state_dict(torch.load(MODEL_DIR / "lstm_pl.ckpt")["state_dict"])
            lstm_model.eval()
            print("SHAP: LSTM")
            shap_lstm_summary(lstm_model, val_df, feature_cols, LAG_STEPS, TARGET_COLUMN)
        except Exception as e:
            print(f"[WARN] LSTM SHAP failed: {e}")

    # ───── SHAP for TFT ─────

        try:
            print("TFT XAI (global interpretation)")
            tft_interpret_global_v3(
                train_df=train_df,
                val_df=val_df,
                feature_cols=feature_cols,
                target_col=TARGET_COLUMN,
                lag_steps=LAG_STEPS,
                horizon=1,          # or your FORECAST_HORIZON
                model_dir=MODEL_DIR,
            )
        except Exception as e:
            print(f"[WARN] TFT interpretation failed: {e}")


    #    import joblib
    #    from config import MODEL_DIR

    #    rf_model = joblib.load(MODEL_DIR / "rf_model.joblib")
    #    shap_tree_summary(rf_model, val_df[feature_cols].iloc[:500], feature_cols)
    #    permutation_imp_plot(rf_model, val_df[feature_cols], val_df[TARGET_COLUMN])


if __name__ == "__main__":
    main()


l = []
print(sorted(l))
