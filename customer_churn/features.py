"""
features.py
-----------
Code to create features for modeling.

Reads ``data/interim/ecommerce_interim.csv``, applies imputation, label
encoding and (optionally) standard scaling, then writes:

* ``data/processed/features.csv``  – feature matrix  (X)
* ``data/processed/target.csv``    – binary target    (y)

Usage::

    python -m customer_churn.features
    # or via Makefile:
    make features
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from customer_churn.config import (
    CATEGORICAL_COLS,
    FEATURES_PATH,
    ID_COLUMN,
    INTERIM_CSV,
    NUMERICAL_COLS,
    PROCESSED_DIR,
    TARGET_COLUMN,
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


def load_interim(path: Path = INTERIM_CSV) -> pd.DataFrame:
    """Load the interim (cleaned) dataset.

    Args:
        path: Path to ``data/interim/ecommerce_interim.csv``.

    Returns:
        Cleaned :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: When the interim CSV is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Interim dataset not found: {path}\n" "Run `python -m customer_churn.dataset` first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded interim data: %d rows × %d columns", *df.shape)
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    - Numerical columns → **median** imputation.
    - Categorical columns → **mode** imputation.

    Args:
        df: DataFrame (must not include target or ID columns).

    Returns:
        DataFrame with no missing values.
    """
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed %d NaN in '%-30s' → median=%.4f", n, col, median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            n = df[col].isna().sum()
            df[col] = df[col].fillna(mode_val)
            logger.info("Imputed %d NaN in '%-30s' → mode='%s'", n, col, mode_val)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns in-place.

    Args:
        df: DataFrame after imputation.

    Returns:
        DataFrame with categorical columns replaced by integer codes.
    """
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info("Encoded  '%s'", col)
    return df


def scale_numericals(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise numerical columns (μ=0, σ=1).

    Args:
        df: DataFrame with encoded categoricals.

    Returns:
        Tuple of (scaled DataFrame, fitted :class:`~sklearn.preprocessing.StandardScaler`).
    """
    num_present = [c for c in NUMERICAL_COLS if c in df.columns]
    scaler = StandardScaler()
    df[num_present] = scaler.fit_transform(df[num_present])
    logger.info("Standardised %d numerical column(s)", len(num_present))
    return df, scaler


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Orchestrate the full feature engineering pipeline.

    Args:
        df: Interim (cleaned) DataFrame including target and ID columns.

    Returns:
        Tuple ``(X, y)`` where:

        * ``X`` – :class:`pandas.DataFrame` of ML-ready features.
        * ``y`` – binary :class:`pandas.Series` target (``Churn``).
    """
    y = df[TARGET_COLUMN].copy().astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    X = impute(X)
    X = encode_categoricals(X)
    X, _ = scale_numericals(X)

    logger.info("Feature matrix ready: %d rows × %d features", *X.shape)
    return X, y


def save_features(X: pd.DataFrame, y: pd.Series) -> None:
    """Persist the feature matrix and target vector.

    Args:
        X: Feature matrix.
        y: Target vector.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(FEATURES_PATH, index=False)
    y.to_csv(TARGET_PATH, index=False, header=True)
    logger.info("Features saved → %s", FEATURES_PATH)
    logger.info("Target   saved → %s", TARGET_PATH)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point invoked by ``python -m customer_churn.features`` or ``make features``."""
    df = load_interim()
    X, y = build_features(df)
    save_features(X, y)


if __name__ == "__main__":
    main()
