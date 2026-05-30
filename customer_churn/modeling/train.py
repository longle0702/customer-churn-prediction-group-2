"""
train.py
--------
Code to train the LightGBM classification model.

Pipeline
~~~~~~~~
1. Load feature matrix (``data/processed/features.csv``) and target
   (``data/processed/target.csv``).
2. Stratified train / validation split.
3. Train a **LightGBM** binary classifier (GPU-accelerated when available,
   falls back to CPU automatically).
4. Evaluate on the validation set and print:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC
   - Inference time (ms / sample)
5. Save the trained model to ``models/lgbm_churn_model.txt``.

Usage::

    python -m customer_churn.modeling.train
    # or via Makefile:
    make train
"""

import logging
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from customer_churn.config import (
    DEVICE,
    FEATURES_PATH,
    LGBM_PARAMS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_PATH,
    TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(
    features_path: Path = FEATURES_PATH,
    target_path: Path = TARGET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load pre-computed feature matrix and target vector.

    Args:
        features_path: Path to ``data/processed/features.csv``.
        target_path:   Path to ``data/processed/target.csv``.

    Returns:
        Tuple ``(X, y)``.

    Raises:
        FileNotFoundError: When either file is missing.
    """
    for p in (features_path, target_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run `python -m customer_churn.features` first."
            )
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    logger.info("Loaded X: %d×%d  |  y: %d samples", *X.shape, len(y))
    return X, y


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / validation split.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data reserved for validation.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_val, y_train, y_val)``
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        "Train: %d samples  |  Val: %d samples  (stratified, test_size=%.0f%%)",
        len(X_train), len(X_val), test_size * 100,
    )
    return X_train, X_val, y_train, y_val


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> lgb.Booster:
    """Train a LightGBM binary classifier.

    Attempts GPU acceleration (OpenCL on Apple Silicon, CUDA on NVIDIA)
    and automatically falls back to CPU if the GPU build is unavailable.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val:   Validation features (used for early stopping).
        y_val:   Validation labels.
        params:  Optional dict to override :data:`~customer_churn.config.LGBM_PARAMS`.

    Returns:
        Trained :class:`lightgbm.Booster`.
    """
    p = dict(LGBM_PARAMS)
    if params:
        p.update(params)

    # Extract fit-time-only params not passed directly to lgb.train
    n_estimators = p.pop("n_estimators", 500)
    early_stopping_rounds = p.pop("early_stopping_rounds", 50)
    p.pop("random_state", None)

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)

    logger.info(
        "Training LightGBM  device='%s'  n_estimators=%d …",
        p.get("device", "cpu"), n_estimators,
    )

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    try:
        booster = lgb.train(
            params=p,
            train_set=train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )
    except Exception as gpu_err:
        logger.warning("GPU training failed (%s). Retrying on CPU …", gpu_err)
        p["device"] = "cpu"
        booster = lgb.train(
            params=p,
            train_set=train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=callbacks,
        )

    logger.info("Training complete – best iteration: %d", booster.best_iteration)
    return booster


def evaluate(
    booster: lgb.Booster,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute evaluation metrics on the validation set.

    Metrics reported
    ~~~~~~~~~~~~~~~~
    * **Accuracy**
    * **Precision**
    * **Recall**
    * **F1-score**
    * **ROC-AUC**
    * **Inference time (ms/sample)**

    Args:
        booster:   Trained :class:`lightgbm.Booster`.
        X_val:     Validation feature matrix.
        y_val:     True labels.
        threshold: Probability threshold for binarisation (default 0.5).

    Returns:
        Dictionary mapping metric name → float value.
    """
    n_samples = len(X_val)
    t_start = time.perf_counter()
    y_prob = booster.predict(X_val, num_iteration=booster.best_iteration)
    t_end = time.perf_counter()

    inference_ms_total = (t_end - t_start) * 1_000
    inference_ms_per_sample = inference_ms_total / n_samples

    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy":                  accuracy_score(y_val, y_pred),
        "precision":                 precision_score(y_val, y_pred, zero_division=0),
        "recall":                    recall_score(y_val, y_pred, zero_division=0),
        "f1_score":                  f1_score(y_val, y_pred, zero_division=0),
        "roc_auc":                   roc_auc_score(y_val, y_prob),
        "inference_time_ms_total":   round(inference_ms_total, 4),
        "inference_time_ms_sample":  round(inference_ms_per_sample, 6),
    }

    logger.info("─" * 55)
    logger.info("Evaluation  (threshold=%.2f,  n=%d)", threshold, n_samples)
    logger.info("  %-32s  %.4f", "Accuracy",  metrics["accuracy"])
    logger.info("  %-32s  %.4f", "Precision", metrics["precision"])
    logger.info("  %-32s  %.4f", "Recall",    metrics["recall"])
    logger.info("  %-32s  %.4f", "F1-score",  metrics["f1_score"])
    logger.info("  %-32s  %.4f", "ROC-AUC",   metrics["roc_auc"])
    logger.info(
        "  %-32s  %.4f ms total  (%.6f ms/sample)",
        "Inference time",
        metrics["inference_time_ms_total"],
        metrics["inference_time_ms_sample"],
    )
    logger.info("─" * 55)
    logger.info(
        "\nClassification report:\n%s",
        classification_report(
            y_val, y_pred, target_names=["No Churn", "Churn"], zero_division=0
        ),
    )
    return metrics


def save_model(booster: lgb.Booster, model_path: Path = MODEL_PATH) -> None:
    """Save the trained model in LightGBM's native text format.

    Args:
        booster:    Trained :class:`lightgbm.Booster`.
        model_path: Destination path (default: ``models/lgbm_churn_model.txt``).
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    logger.info("Model saved → %s", model_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end training pipeline."""
    X, y = load_features()
    X_train, X_val, y_train, y_val = split(X, y)
    booster = train(X_train, y_train, X_val, y_val)
    evaluate(booster, X_val, y_val)
    save_model(booster)


if __name__ == "__main__":
    main()
