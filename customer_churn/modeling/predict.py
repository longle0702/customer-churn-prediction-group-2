"""
predict.py
----------
Code to run model inference with the trained LightGBM model.

Loads ``models/lgbm_churn_model.txt`` and a feature CSV, generates
churn probability scores and binary predictions, then saves the result to
``data/processed/predictions.csv``.

Also reports inference-time statistics (total ms and ms/sample).
Optionally evaluates accuracy, precision, recall, F1 and ROC-AUC when
ground-truth labels are available.

Usage::

    python -m customer_churn.modeling.predict
    python -m customer_churn.modeling.predict \\
        --input  data/processed/features.csv \\
        --output data/processed/predictions.csv \\
        --threshold 0.4
    # or via Makefile:
    make predict
"""

import argparse
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from customer_churn.config import (
    FEATURES_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    TARGET_PATH,
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
# Public API
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODEL_PATH) -> lgb.Booster:
    """Load a saved LightGBM model from disk.

    Args:
        model_path: Path to the ``.txt`` model file.

    Returns:
        :class:`lightgbm.Booster` ready for inference.

    Raises:
        FileNotFoundError: When the model file is missing.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python -m customer_churn.modeling.train` first."
        )
    booster = lgb.Booster(model_file=str(model_path))
    logger.info("Model loaded from %s", model_path)
    return booster


def load_features(input_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the feature matrix for inference.

    Args:
        input_path: Path to a CSV file with the same column schema used
            during training.

    Returns:
        :class:`pandas.DataFrame` of features.

    Raises:
        FileNotFoundError: When the file is missing.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")
    X = pd.read_csv(input_path)
    logger.info("Loaded %d samples for inference", len(X))
    return X


def predict(
    booster: lgb.Booster,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate churn predictions and probability scores.

    Measures and logs inference time (total ms and ms/sample).

    Args:
        booster:   Trained :class:`lightgbm.Booster`.
        X:         Feature matrix (same schema as training data).
        threshold: Probability threshold for binarisation (default 0.5).

    Returns:
        :class:`pandas.DataFrame` with columns:

        * ``churn_probability``  – P(churn=1)
        * ``churn_prediction``   – 0 or 1
        * ``inference_time_ms``  – per-sample inference time (ms, same for all rows)
    """
    n = len(X)
    t0 = time.perf_counter()
    y_prob = booster.predict(X, num_iteration=booster.best_iteration)
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1_000
    per_sample_ms = total_ms / n

    logger.info(
        "Inference  %d samples | total=%.2f ms | per-sample=%.4f ms",
        n, total_ms, per_sample_ms,
    )

    y_pred = (y_prob >= threshold).astype(int)
    n_churn = int(y_pred.sum())
    logger.info("Predicted %d churned / %d total  (%.1f%%)", n_churn, n, 100 * n_churn / n)

    return pd.DataFrame(
        {
            "churn_probability": y_prob,
            "churn_prediction":  y_pred,
            "inference_time_ms": per_sample_ms,
        }
    )


def evaluate_predictions(
    results: pd.DataFrame,
    target_path: Path = TARGET_PATH,
) -> dict[str, float] | None:
    """Optionally evaluate predictions against ground-truth labels.

    Prints accuracy, precision, recall, F1, ROC-AUC and inference time
    if ``target_path`` exists.

    Args:
        results:     DataFrame returned by :func:`predict`.
        target_path: Path to ``data/processed/target.csv``.

    Returns:
        Dictionary of metrics, or ``None`` when labels are unavailable.
    """
    if not target_path.exists():
        logger.info("No ground-truth labels found – skipping metric evaluation.")
        return None

    y_true = pd.read_csv(target_path).squeeze().astype(int)
    y_pred = results["churn_prediction"].astype(int)
    y_prob = results["churn_probability"].values

    metrics = {
        "accuracy":                 accuracy_score(y_true, y_pred),
        "precision":                precision_score(y_true, y_pred, zero_division=0),
        "recall":                   recall_score(y_true, y_pred, zero_division=0),
        "f1_score":                 f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":                  roc_auc_score(y_true, y_prob),
        "inference_time_ms_sample": results["inference_time_ms"].iloc[0],
    }

    logger.info("─" * 55)
    logger.info("Post-inference evaluation metrics")
    for name, val in metrics.items():
        logger.info("  %-35s %.6f", name, val)
    logger.info("─" * 55)

    return metrics


def save_predictions(
    results: pd.DataFrame,
    output_path: Path = PREDICTIONS_PATH,
) -> None:
    """Save prediction results to CSV.

    Args:
        results:     DataFrame from :func:`predict`.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Predictions saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run churn inference on a prepared feature CSV."
    )
    parser.add_argument("--model",     type=Path, default=MODEL_PATH,       help="Path to .txt model file")
    parser.add_argument("--input",     type=Path, default=FEATURES_PATH,    help="Input features CSV")
    parser.add_argument("--output",    type=Path, default=PREDICTIONS_PATH, help="Output predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.5,             help="Probability threshold (default 0.5)")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    booster = load_model(args.model)
    X = load_features(args.input)
    results = predict(booster, X, threshold=args.threshold)
    evaluate_predictions(results)
    save_predictions(results, args.output)


if __name__ == "__main__":
    main()
