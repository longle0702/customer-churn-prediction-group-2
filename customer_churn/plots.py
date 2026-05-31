"""
plots.py
--------
Code to create visualisations for the customer churn project.

All figures are saved to ``reports/figures/``.

Available plots
~~~~~~~~~~~~~~~
* :func:`plot_churn_distribution`  – class balance bar chart
* :func:`plot_feature_importance`  – LightGBM gain-based feature importance
* :func:`plot_confusion_matrix`    – heatmap of the confusion matrix
* :func:`plot_roc_curve`           – ROC curve with AUC annotation
* :func:`plot_precision_recall_curve` – Precision-Recall curve

Usage::

    python -m customer_churn.plots
    # or via Makefile:
    make plots
"""

import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
)

from customer_churn.config import (
    FIGURES_DIR,
    FEATURES_PATH,
    MODEL_PATH,
    TARGET_PATH,
)

matplotlib.use("Agg")  # non-interactive backend (safe for headless runs)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#4F46E5",  # indigo
    "secondary": "#F97316",  # orange
    "neutral": "#6B7280",
    "bg": "#F9FAFB",
}


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------


def plot_churn_distribution(
    y: pd.Series,
    output_path: Path = FIGURES_DIR / "churn_distribution.png",
) -> None:
    """Bar chart showing the class distribution (churned vs retained).

    Args:
        y:           Target series (0 = retained, 1 = churned).
        output_path: Destination PNG file.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PALETTE["bg"])
    counts = y.value_counts().sort_index()
    bars = ax.bar(
        ["Retained (0)", "Churned (1)"],
        counts.values,
        color=[PALETTE["primary"], PALETTE["secondary"]],
        edgecolor="white",
    )
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{count:,}\n({count / len(y) * 100:.1f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_title("Churn Class Distribution", pad=14)
    ax.set_ylabel("Number of Customers")
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_feature_importance(
    booster: lgb.Booster,
    top_n: int = 15,
    output_path: Path = FIGURES_DIR / "feature_importance.png",
) -> None:
    """Horizontal bar chart of the top-N LightGBM feature importances (gain).

    Args:
        booster:     Trained :class:`lightgbm.Booster`.
        top_n:       Number of top features to display.
        output_path: Destination PNG file.
    """
    importance = (
        pd.Series(
            booster.feature_importance(importance_type="gain"),
            index=booster.feature_name(),
        )
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)), facecolor=PALETTE["bg"])
    ax.barh(importance.index, importance.values, color=PALETTE["primary"], edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances (Gain)", pad=14)
    ax.set_xlabel("Gain")
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> None:
    """Heatmap of the confusion matrix.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Binary predictions.
        output_path: Destination PNG file.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=PALETTE["bg"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix", pad=12)
    ax.set_facecolor(PALETTE["bg"])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path = FIGURES_DIR / "roc_curve.png",
) -> None:
    """ROC curve with AUC score annotation.

    Args:
        y_true:      Ground-truth labels.
        y_prob:      Predicted probabilities for the positive class.
        output_path: Destination PNG file.
    """
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE["bg"])
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color=PALETTE["primary"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random classifier")
    ax.set_title(f"ROC Curve  (AUC = {auc:.4f})", pad=12)
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path = FIGURES_DIR / "precision_recall_curve.png",
) -> None:
    """Precision-Recall curve.

    Args:
        y_true:      Ground-truth labels.
        y_prob:      Predicted probabilities for the positive class.
        output_path: Destination PNG file.
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE["bg"])
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, color=PALETTE["secondary"])
    ax.set_title("Precision-Recall Curve", pad=12)
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point – generate all plots at once
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all project figures.

    Loads model and processed data, then renders and saves every plot to
    ``reports/figures/``.
    """
    # Load model
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # Load features and labels
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(TARGET_PATH).squeeze().astype(int)

    # Predict
    y_prob = booster.predict(X, num_iteration=booster.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)

    # Generate all figures
    plot_churn_distribution(y)
    plot_feature_importance(booster)
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_prob)
    plot_precision_recall_curve(y, y_prob)

    logger.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
