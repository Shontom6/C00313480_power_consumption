"""Global configuration constants shared across the Energy‑Weather package.
Adjust any paths, hyper‑parameters or random seeds here and import from the
other modules so everything stays in sync.
"""
from pathlib import Path

# ──────────────────────────────────── Paths ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "energy_weather_raw_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────── Runtime ───────────────────────────────────
RANDOM_STATE = 42

# ────────────────────────────── Forecast Settings ──────────────────────────────
# Hours ahead to predict. Change to 24 for day‑ahead for example.
FORECAST_HORIZON = 1

# Number of historical steps fed into DL networks or used to create lag features
LAG_STEPS = 24

# ────────────────────────────────── Columns ────────────────────────────────────
TARGET_COLUMN = "active_power"
DATE_COLUMN = "date"

# Names of the dataframe columns that come from OpenWeather (prefixed to filter)
WEATHER_PREFIXES = [
    "temp", "humidity", "pressure", "speed",  # numeric weather vars
    "main", "description"                      # categorical
]

# ───────────────────────────── Hyper‑Parameter Grids ───────────────────────────
RF_PARAM_GRID = {
    "model__n_estimators": [200, 400, 600, 800],
    "model__max_depth": [None, 10, 20, 30, 40],
    "model__min_samples_leaf": [1, 2, 4],
}

ET_PARAM_GRID = {
    "model__n_estimators": [200, 400, 600, 800],
    "model__max_depth": [None, 10, 20, 30, 40],
    "model__min_samples_leaf": [1, 2, 4],
}

# ─────────────────────────────── Deep Learning ─────────────────────────────────
LSTM_CONFIG = {
    "units": 64,
    "dense_units": 32,
    "batch_size": 64,
    "epochs": 50,
    "patience": 5,
}

TFT_CONFIG = {
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 32,
    "learning_rate": 1e-3,
    "max_epochs": 50, # should be 50 for training 
    "batch_size": 64,
}
