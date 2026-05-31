"""
config.py
---------
Central store for all project-wide variables, paths and configuration.

Import this module anywhere in the project to get consistent, resolved paths
without hard-coding strings in individual scripts.
"""

import platform
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from this file: customer_churn/config.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

# Raw dataset
EXCEL_FILENAME: str = "E Commerce Dataset.xlsx"
EXCEL_PATH: Path = RAW_DIR / EXCEL_FILENAME
SHEET_NAME: str = "E Comm"

# Processed artefacts
INTERIM_CSV: Path = INTERIM_DIR / "ecommerce_interim.csv"
FEATURES_PATH: Path = PROCESSED_DIR / "features.csv"
TARGET_PATH: Path = PROCESSED_DIR / "target.csv"
PREDICTIONS_PATH: Path = PROCESSED_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Model directory
# ---------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "lgbm_churn_model.txt"  # LightGBM native format

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "CustomerID"

NUMERICAL_COLS: list[str] = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "SatisfactionScore",
    "CityTier",
    "Complain",
]

CATEGORICAL_COLS: list[str] = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


# ---------------------------------------------------------------------------
# GPU / device detection
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    """Auto-detect the best available device for LightGBM.

    Returns:
        ``"gpu"``  when an NVIDIA GPU is available (CUDA).
        ``"gpu"``  when running on Apple Silicon (OpenCL via Metal).
        ``"cpu"``  fallback.
    """
    system = platform.system()
    machine = platform.machine()

    # Apple Silicon (arm64 Darwin) -> try OpenCL GPU via LightGBM
    if system == "Darwin" and machine == "arm64":
        return "gpu"

    # NVIDIA GPU (Linux / Windows with CUDA)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            return "gpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return "cpu"


DEVICE: str = _detect_device()

# ---------------------------------------------------------------------------
# LightGBM hyper-parameters
# ---------------------------------------------------------------------------
LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "scale_pos_weight": 4,  # handles class imbalance (~83% vs 17%)
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbosity": -1,
    "random_state": 42,
    "device": DEVICE,
}

# Train / validation split
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Ensure output directories exist at import time
# ---------------------------------------------------------------------------
for _dir in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
