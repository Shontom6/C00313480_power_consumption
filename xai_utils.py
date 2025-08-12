"""Explainable AI helpers: SHAP, permutation importance, partial dependence."""
from __future__ import annotations
from sklearn.inspection import permutation_importance

from config import RANDOM_STATE

# ──────────────────────────────── SHAP Trees ───────────────────────────────────
import scipy.sparse as sp       # add to top‑level imports
import matplotlib.pyplot as plt
import shap
import pandas as pd

# -------------------------------------------------------------------------
def shap_tree_summary(model_pipeline, X_sample: pd.DataFrame, feature_names):
    """
    Generate a SHAP summary plot for a tree‑based *pipeline* that contains a
    ColumnTransformer (e.g. OneHotEncoder).

    • Transforms X_sample through the pre‑processor so its shape matches
      shap_values.
    • Extracts post‑encoding feature names via get_feature_names_out().
    """
    # 1) run the sample through the ColumnTransformer
    pre = model_pipeline.named_steps["pre"]
    X_proc = pre.transform(X_sample)
    if sp.issparse(X_proc):
        X_proc = X_proc.toarray()

    # 2) feature names AFTER one‑hot / scaling
    try:
        proc_feature_names = pre.get_feature_names_out()
    except AttributeError:      # scikit‑learn < 1.0 fallback
        proc_feature_names = feature_names

    # 3) SHAP values from the underlying tree model
    explainer   = shap.TreeExplainer(model_pipeline.named_steps["model"])
    shap_values = explainer.shap_values(X_proc)

    shap.summary_plot(
        shap_values,
        X_proc,
        feature_names=proc_feature_names,
        show=False,
    )
    plt.title("SHAP Summary Plot — Tree Model")
    plt.tight_layout()
    plt.show()

import shap
import numpy as np
import torch

def shap_lstm_summary(model, df, feature_cols, lag_steps, target_col):
    model.eval()

    # Generate sequences (samples, lag_steps, features)
    sequences = []
    for i in range(len(df) - lag_steps):
        seq = df[feature_cols].iloc[i : i + lag_steps].values
        sequences.append(seq)

    X = np.stack(sequences)  # shape: (samples, lag_steps, features)

    background = X[:50]
    test_data = X[:10]

    # Model wrapper: Input must be reshaped back to 3D inside
    def model_predict(x):
        x_reshaped = x.reshape((x.shape[0], lag_steps, len(feature_cols)))
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x_tensor).detach().numpy()
        return preds

    # Flatten for SHAP (samples, lag_steps * features)
    background_flat = background.reshape(background.shape[0], -1)
    test_data_flat = test_data.reshape(test_data.shape[0], -1)

    # Flat feature names
    flat_feature_names = [
        f"{feat}_t-{lag}"
        for lag in range(lag_steps, 0, -1)
        for feat in feature_cols
    ]

    # Use KernelExplainer with NumPy inputs
    explainer = shap.KernelExplainer(model_predict, background_flat)
    shap_values = explainer.shap_values(test_data_flat)

    # Plot
    shap.summary_plot(shap_values, test_data_flat, feature_names=flat_feature_names)



# ──────────────────────────── Permutation Importance ───────────────────────────

def permutation_imp_plot(model_pipeline, X_val: pd.DataFrame, y_val: pd.Series):
    result = permutation_importance(model_pipeline, X_val, y_val, n_repeats=20,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), X_val.columns[sorted_idx])
    plt.title("Permutation Importance — Decrease in MAE")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# xai_utils.py
def tft_interpret_global_v3(train_df, val_df, feature_cols, target_col, lag_steps, horizon, model_dir):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import pandas as pd
    import pytorch_forecasting as ptf

    def _np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)): return np.array(x)
        return x

    def _dedup(seq):
        seen=set(); out=[]
        for s in seq:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    out_dir = Path(model_dir) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(model_dir) / "tft.ckpt"
    if not ckpt.exists():
        print(f"[WARN] No TFT checkpoint at {ckpt}"); return

    # --- rebuild datasets like training ---
    training = ptf.TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=target_col,
        group_ids=["all"],
        max_encoder_length=lag_steps,
        max_prediction_length=horizon,
        time_varying_known_reals=feature_cols,
        time_varying_unknown_reals=[target_col],
    )
    val_ds = ptf.TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=64)

    tft = ptf.TemporalFusionTransformer.load_from_checkpoint(ckpt)

    # ---- try to get true feature-name order from model hparams ----
    hp = getattr(tft, "hparams", {})
    if hasattr(hp, "items"):  # Namespace or dict-like
        try:
            # some versions hold a dict under .__dict__
            hp_dict = dict(hp) if isinstance(hp, dict) else {k: getattr(hp, k) for k in dir(hp) if not k.startswith("_")}
        except Exception:
            hp_dict = {}
    else:
        hp_dict = {}

    name_keys = [
        "x_reals", "x_categoricals",
        "time_varying_known_reals", "time_varying_unknown_reals",
        "time_varying_known_categoricals", "time_varying_unknown_categoricals",
        "static_reals", "static_categoricals",
    ]
    names_from_model = []
    for k in name_keys:
        v = hp_dict.get(k, [])
        if isinstance(v, (list, tuple)):
            names_from_model.extend(list(v))
    names_from_model = _dedup([str(n) for n in names_from_model])

    # fallback to dataset order if model didn't expose names
    if not names_from_model:
        try:
            names_from_model = _dedup(list(getattr(training, "reals", [])) + list(getattr(training, "categoricals", [])))
        except Exception:
            names_from_model = []

    # IMPORTANT: use RAW outputs for interpretation
    raw, x = tft.predict(val_dl, mode="raw", return_x=True)

    try:
        interp = tft.interpret_output(raw, reduction=None)
    except Exception as e:
        print(f"[INFO] interpret_output(reduction=None) failed: {e}. Falling back to reduction='sum'.")
        interp = tft.interpret_output(raw, reduction="sum")

    # ===== Variable Importance (decoder) =====
    vi = interp.get("decoder_variables", None)
    names, scores = [], []
    if vi is not None:
        if isinstance(vi, dict):
            names = list(vi.keys())
            scores = [float(vi[n]) for n in names]
        else:
            arr = _np(vi)
            if arr is not None:
                if arr.ndim == 1:
                    scores = arr.tolist()
                else:
                    scores = arr.mean(axis=0).tolist()
                # Align names length to scores length
                if names_from_model and len(names_from_model) >= len(scores):
                    names = names_from_model[:len(scores)]
                else:
                    names = [f"var_{i}" for i in range(len(scores))]

    if scores:
        df_vi = pd.DataFrame({"variable": names, "importance": scores}).sort_values("importance", ascending=False)
        df_vi.to_csv(out_dir / "tft_decoder_variable_importance.csv", index=False)

        order = np.argsort(scores)[::-1]
        plt.figure()
        plt.barh([names[i] for i in order][::-1], [scores[i] for i in order][::-1])
        plt.title("TFT Decoder Variable Importance (global)")
        plt.tight_layout()
        plt.savefig(out_dir / "tft_decoder_variable_importance.png", dpi=200)
        plt.close()
        print(f"[OK] Saved variable importance to {out_dir}")
    else:
        print("[INFO] 'decoder_variables' not found or empty; skipping VI plot.")

    # ===== Attention =====
    attn = _np(interp.get("attention", None))
    if attn is None:
        print("[INFO] 'attention' not found in interpretation."); return

    print(f"[DEBUG] attention shape: {getattr(attn, 'shape', None)}")
    if attn.ndim == 1:
        plt.figure()
        plt.plot(np.arange(len(attn)), attn)
        plt.xlabel("Decoder time step"); plt.ylabel("Attention")
        plt.title("TFT Decoder Attention")
        plt.tight_layout()
        plt.savefig(out_dir / "tft_decoder_attention.png", dpi=200)
        plt.close()
    elif attn.ndim == 2:
        plt.figure()
        plt.imshow(attn, aspect="auto")
        plt.xlabel("Decoder time step"); plt.ylabel("Sample")
        plt.title("TFT Decoder Attention (batch)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / "tft_decoder_attention_heatmap.png", dpi=200)
        plt.close()
    elif attn.ndim == 3:
        attn_mean = attn.mean(axis=1)
        plt.figure()
        plt.imshow(attn_mean, aspect="auto")
        plt.xlabel("Decoder time step"); plt.ylabel("Sample")
        plt.title("TFT Decoder Attention (heads-avg)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / "tft_decoder_attention_headsavg.png", dpi=200)
        plt.close()
    else:
        flat = attn.reshape(-1)
        plt.figure()
        plt.plot(np.arange(len(flat)), flat)
        plt.title("TFT Decoder Attention (flattened)")
        plt.tight_layout()
        plt.savefig(out_dir / "tft_decoder_attention_flat.png", dpi=200)
        plt.close()

    print(f"[OK] Saved attention plot(s) to {out_dir}")
