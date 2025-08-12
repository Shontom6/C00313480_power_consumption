"""
Deep‑learning models implemented in **PyTorch Lightning**:
  • LSTM for one‑step‑ahead forecasting
  • Optional Temporal Fusion Transformer (requires pytorch‑forecasting)

TensorBoard logs are stored in a `lightning_logs/` folder; models are
checkpointed into energy_weather/models/.
"""
# from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from config import (
    MODEL_DIR,
    RANDOM_STATE,
    LAG_STEPS,
    FORECAST_HORIZON,
    TARGET_COLUMN,
    LSTM_CONFIG,
)

from metrics import evaluate, print_metrics

# Optional TFT
try:
    import pytorch_forecasting as ptf
except ImportError:
    ptf = None


# ───────────────────────────── Dataset Utilities ──────────────────────────────
class WindowDataset(Dataset):
    """Sliding‑window dataset to feed the LSTM."""
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        lag_steps: int,
        horizon: int,
    ):
        # ensure only numeric dtypes
        block = df[features + [target]].copy()
        if block.select_dtypes("object").shape[1]:
            raise ValueError("WindowDataset received non‑numeric columns.")

        self.data = block.values.astype(np.float32)
        self.lag_steps = lag_steps
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.lag_steps - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lag_steps, :-1]              # features window
        y = self.data[idx + self.lag_steps + self.horizon - 1, -1]  # point forecast
        return torch.tensor(x), torch.tensor(y)


# ───────────────────────────── LSTM LightningModule ────────────────────────────
class EnergyLSTM(pl.LightningModule):
    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg["units"],
            batch_first=True,
        )
        self.fc1 = nn.Linear(cfg["units"], cfg["dense_units"])
        self.relu = nn.ReLU()
        self.out = nn.Linear(cfg["dense_units"], 1)
        self.loss_fn = nn.L1Loss()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return self(x)

    def forward(self, x):
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]          # last timestep
        return self.out(self.relu(self.fc1(last))).squeeze(1)

    # ---------- Lightning hooks ----------
    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_mae", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_mae", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ───────────────────────────── Training Function ───────────────────────────────
def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
) -> Dict[str, float]:
    """Train Lightning‑based LSTM, save checkpoint, return validation metrics."""
    train_ds = WindowDataset(train_df, features, TARGET_COLUMN, LAG_STEPS, FORECAST_HORIZON)
    val_ds   = WindowDataset(val_df,   features, TARGET_COLUMN, LAG_STEPS, FORECAST_HORIZON)

    train_dl = DataLoader(train_ds, batch_size=LSTM_CONFIG["batch_size"], shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=LSTM_CONFIG["batch_size"], shuffle=False)

    model = EnergyLSTM(len(features), LSTM_CONFIG)

    trainer = pl.Trainer(
        max_epochs=LSTM_CONFIG["epochs"],
        deterministic=True,
        accelerator="auto",
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_mae",
                mode="min",
                patience=LSTM_CONFIG["patience"],
            )
        ],
    )

    trainer.fit(model, train_dl, val_dl)
    MODEL_DIR.mkdir(exist_ok=True)
    trainer.save_checkpoint(str(MODEL_DIR / "lstm_pl.ckpt"))

    # ----- validation metrics -----
    preds = torch.cat(trainer.predict(model, val_dl)).cpu().numpy().ravel()
    y_true = np.array([y for _, y in val_ds])
    metrics = evaluate(y_true, preds, prefix="lstm_")
    print_metrics(metrics)
    return metrics


# ──────────────────── Temporal Fusion Transformer (optional) ───────────────────
def train_tft(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
):
    """Train a Temporal Fusion Transformer if pytorch‑forecasting is available."""
    if ptf is None:
        print("[WARN] pytorch‑forecasting not installed; skipping TFT.")
        return {}

    from config import TFT_CONFIG

    max_encoder_length = LAG_STEPS
    max_prediction_length = FORECAST_HORIZON

    training = ptf.TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["all"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=features,
        time_varying_unknown_reals=[TARGET_COLUMN],
    )
    validation = ptf.TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

    train_dl = training.to_dataloader(train=True,  batch_size=TFT_CONFIG["batch_size"])
    val_dl   = validation.to_dataloader(train=False, batch_size=TFT_CONFIG["batch_size"])

    tft = ptf.TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=TFT_CONFIG["learning_rate"],
        hidden_size=TFT_CONFIG["hidden_size"],
        attention_head_size=TFT_CONFIG["attention_head_size"],
        dropout=TFT_CONFIG["dropout"],
        hidden_continuous_size=TFT_CONFIG["hidden_continuous_size"],
        loss=ptf.metrics.MAE(),
        log_interval=10,
    )

    pl_trainer = pl.Trainer(max_epochs=TFT_CONFIG["max_epochs"], accelerator="auto")
    pl_trainer.fit(tft, train_dl, val_dl)
    MODEL_DIR.mkdir(exist_ok=True)
    pl_trainer.save_checkpoint(str(MODEL_DIR / "tft.ckpt"))
    # After creating your dataset
    #torch.save(training.dataset, MODEL_DIR / "tft_dataset.pt")  
    #torch.save(train_dataset, MODEL_DIR / "tft_dataset.pt")# Shon 
    
    preds = tft.predict(val_dl).detach().cpu().numpy()
    y_true  = np.concatenate([y[0].cpu().numpy() for _, y in val_dl])
    metrics = evaluate(y_true, preds, prefix="tft_")
    print_metrics(metrics)
    return metrics

# ========= Final test evaluation (LSTM) =========
def evaluate_lstm_on_test(
    test_df: pd.DataFrame,
    features: List[str],
):
    """Load saved LSTM and print final test metrics."""
    test_ds = WindowDataset(test_df, features, TARGET_COLUMN, LAG_STEPS, FORECAST_HORIZON)
    test_dl = DataLoader(test_ds, batch_size=LSTM_CONFIG["batch_size"], shuffle=False)

    # Load model
    model = EnergyLSTM(len(features), LSTM_CONFIG)
    ckpt = torch.load(MODEL_DIR / "lstm_pl.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Predict
    preds = []
    with torch.no_grad():
        for x, _ in test_dl:
            preds.append(model(x).cpu())
    preds = torch.cat(preds).numpy().ravel()

    # y_true from dataset
    y_true = np.array([y for _, y in test_ds])
    metrics = evaluate(y_true, preds, prefix="lstm_test_")
    print_metrics(metrics)
    return metrics

# ========= Final test evaluation (TFT) =========
def evaluate_tft_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
):
    """Rebuild TFT dataset config, load checkpoint, and evaluate on test."""
    if ptf is None:
        print("[WARN] pytorch-forecasting not installed; skipping TFT test eval.")
        return {}

    from config import TFT_CONFIG

    # Recreate the training TimeSeriesDataSet (same config as train_tft)
    training = ptf.TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["all"],
        max_encoder_length=LAG_STEPS,
        max_prediction_length=FORECAST_HORIZON,
        time_varying_known_reals=features,
        time_varying_unknown_reals=[TARGET_COLUMN],
    )
    test_dataset = ptf.TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    test_dl = test_dataset.to_dataloader(train=False, batch_size=TFT_CONFIG["batch_size"])

    # Load TFT checkpoint and predict
    tft = ptf.TemporalFusionTransformer.load_from_checkpoint(MODEL_DIR / "tft.ckpt")
    preds = tft.predict(test_dl).detach().cpu().numpy().ravel()

    # y_true (note: pytorch-forecasting dataloader returns tuple (x, (y, weight)))
    y_true = np.concatenate([y[0].cpu().numpy().ravel() for _, y in test_dl])

    metrics = evaluate(y_true, preds, prefix="tft_test_")
    print_metrics(metrics)
    return metrics
