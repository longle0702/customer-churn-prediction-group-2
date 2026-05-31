"""
explainability.py
-----------------
SHAP explainability utilities for the customer churn LightGBM model.

KAN-9 scope
~~~~~~~~~~~
This module loads the trained LightGBM model and processed feature matrix,
builds a :class:`shap.TreeExplainer`, computes SHAP values, and generates
global model interpretation plots for the whole dataset.

Usage::

    python -m customer_churn.explainability --mode global

Outputs are written to ``reports/figures/xai/``.
"""

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from customer_churn.config import FEATURES_PATH, MODEL_PATH, XAI_FIGURES_DIR

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path = MODEL_PATH) -> lgb.Booster:
    """Load the trained LightGBM booster used in Part 2.

    Args:
        model_path: Path to the LightGBM native model file.

    Returns:
        Trained :class:`lightgbm.Booster`.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun `python -m customer_churn.modeling.train` first."
        )
    logger.info("Loading model from %s", model_path)
    return lgb.Booster(model_file=str(model_path))


def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    """Load the processed feature matrix used by the LightGBM model.

    Args:
        features_path: Path to ``data/processed/features.csv``.

    Returns:
        Feature matrix with the same columns used during training.

    Raises:
        FileNotFoundError: If the feature file does not exist.
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {features_path}\nRun `python -m customer_churn.features` first."
        )
    features = pd.read_csv(features_path)
    logger.info("Loaded features for SHAP: %d rows × %d columns", *features.shape)
    return features


def sample_features(
    features: pd.DataFrame,
    max_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows for efficient SHAP plotting while keeping reproducibility.

    Args:
        features: Full processed feature matrix.
        max_samples: Maximum number of rows to explain.
        random_state: Random seed used when sampling.

    Returns:
        Full feature matrix if smaller than ``max_samples``; otherwise a sample.
    """
    if len(features) <= max_samples:
        return features.copy()
    return features.sample(n=max_samples, random_state=random_state).copy()


def build_tree_explainer(model: lgb.Booster) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer for the LightGBM model.

    Args:
        model: Trained LightGBM booster.

    Returns:
        Configured :class:`shap.TreeExplainer`.
    """
    logger.info("Building SHAP TreeExplainer")
    return shap.TreeExplainer(model)


def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    features: pd.DataFrame,
) -> shap.Explanation:
    """Compute SHAP values for a feature matrix.

    Args:
        explainer: SHAP TreeExplainer.
        features: Processed feature matrix.

    Returns:
        SHAP explanation object containing values, base values and input data.
    """
    logger.info("Computing SHAP values for %d rows", len(features))
    explanation = explainer(features)
    logger.info("Computed SHAP values with shape %s", np.asarray(explanation.values).shape)
    return explanation


def save_summary_bar_plot(
    explanation: shap.Explanation,
    features: pd.DataFrame,
    output_path: Path = XAI_FIGURES_DIR / "shap_summary_bar_global.png",
) -> None:
    """Save a global mean-absolute SHAP feature importance bar plot.

    Args:
        explanation: SHAP explanation for all sampled rows.
        features: Feature matrix used to compute the explanation.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(explanation.values, features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP summary bar plot → %s", output_path)


def save_beeswarm_plot(
    explanation: shap.Explanation,
    output_path: Path = XAI_FIGURES_DIR / "shap_beeswarm_global.png",
) -> None:
    """Save a beeswarm plot showing SHAP values for all explained rows.

    Args:
        explanation: SHAP explanation for all sampled rows.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP beeswarm plot → %s", output_path)


def save_mean_shap_plot(
    explanation: shap.Explanation,
    output_path: Path = XAI_FIGURES_DIR / "shap_mean_global.png",
) -> None:
    """Save a custom global mean absolute SHAP feature-importance plot.

    Args:
        explanation: SHAP explanation for all sampled rows.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(explanation.values)
    importance = pd.Series(
        np.abs(values).mean(axis=0),
        index=explanation.feature_names,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(importance) * 0.35)))
    ax.barh(importance.index, importance.values, color="#4F46E5", edgecolor="white")
    ax.set_title("Mean Absolute SHAP Values — Global Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved global mean SHAP plot → %s", output_path)


def generate_global_explanations(
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
    output_dir: Path = XAI_FIGURES_DIR,
    max_samples: int = 500,
) -> list[Path]:
    """Generate all KAN-9 global SHAP interpretation artefacts.

    Args:
        model_path: Path to the trained LightGBM model.
        features_path: Path to processed features.
        output_dir: Directory where SHAP plots are saved.
        max_samples: Maximum number of rows used for SHAP computation.

    Returns:
        List of generated artefact paths.
    """
    model = load_model(model_path)
    features = sample_features(load_features(features_path), max_samples=max_samples)
    explainer = build_tree_explainer(model)
    explanation = compute_shap_explanation(explainer, features)

    outputs = [
        output_dir / "shap_summary_bar_global.png",
        output_dir / "shap_beeswarm_global.png",
        output_dir / "shap_mean_global.png",
    ]
    save_summary_bar_plot(explanation, features, outputs[0])
    save_beeswarm_plot(explanation, outputs[1])
    save_mean_shap_plot(explanation, outputs[2])
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for the churn model.")
    parser.add_argument("--mode", choices=["global"], default="global")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=XAI_FIGURES_DIR)
    parser.add_argument("--max-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    """CLI entry point for SHAP explanations."""
    args = _parse_args()
    if args.mode == "global":
        outputs = generate_global_explanations(
            model_path=args.model,
            features_path=args.features,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
        logger.info("Generated %d global SHAP artefacts", len(outputs))


if __name__ == "__main__":
    main()