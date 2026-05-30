"""
dataset.py
----------
Scripts to download or generate data.

Responsibilities
~~~~~~~~~~~~~~~~
1. Load the raw Excel dataset from ``data/raw/``.
2. Apply basic cleaning (deduplication, type casting, whitespace stripping).
3. Save the result to ``data/interim/ecommerce_interim.csv`` for the feature
   engineering step.

Usage::

    python -m customer_churn.dataset
    # or via Makefile:
    make data
"""

import logging
from pathlib import Path

import pandas as pd

from customer_churn.config import (
    CATEGORICAL_COLS,
    EXCEL_PATH,
    ID_COLUMN,
    INTERIM_CSV,
    SHEET_NAME,
    TARGET_COLUMN,
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

def load_raw(excel_path: Path = EXCEL_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    """Load the raw dataset from the Excel workbook stored in ``data/raw/``.

    Args:
        excel_path: Absolute path to the ``.xlsx`` file.
        sheet_name: Worksheet name to read.

    Returns:
        Raw :class:`pandas.DataFrame` with all original columns.

    Raises:
        FileNotFoundError: When ``excel_path`` does not exist.
    """
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {excel_path}\n"
            f"Place '{excel_path.name}' inside data/raw/ before proceeding."
        )

    logger.info("Loading '%s'  sheet='%s' …", excel_path.name, sheet_name)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    logger.info("Raw dataset loaded: %d rows × %d columns", *df.shape)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning to the raw DataFrame.

    Steps
    ~~~~~
    1. Drop exact duplicate rows.
    2. Cast ``CustomerID`` to string (identifier, not a feature).
    3. Strip leading / trailing whitespace from all string columns.
    4. Cast the target to integer (0 / 1).

    Args:
        df: Raw DataFrame from :func:`load_raw`.

    Returns:
        Cleaned :class:`pandas.DataFrame`.
    """
    n_before = len(df)

    # 1. Deduplication
    df = df.drop_duplicates().copy()
    logger.info("Dropped %d duplicate row(s)  (%d → %d)", n_before - len(df), n_before, len(df))

    # 2. CustomerID as string identifier
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)

    # 3. Strip whitespace from object columns
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # 4. Ensure target is int
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    logger.info("Cleaning complete – %d rows retained", len(df))
    return df


def save_interim(df: pd.DataFrame, output_path: Path = INTERIM_CSV) -> None:
    """Persist the cleaned DataFrame to the interim data folder.

    Args:
        df: Cleaned DataFrame.
        output_path: Destination CSV path (default: ``data/interim/ecommerce_interim.csv``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Interim dataset saved → %s", output_path)


def make_dataset(
    excel_path: Path = EXCEL_PATH,
    output_path: Path = INTERIM_CSV,
) -> pd.DataFrame:
    """Full data-preparation pipeline: load → clean → save interim.

    Args:
        excel_path: Path to the raw ``.xlsx`` file.
        output_path: Destination for the interim CSV.

    Returns:
        Cleaned :class:`pandas.DataFrame`.
    """
    df = load_raw(excel_path)
    df = clean(df)
    save_interim(df, output_path)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point invoked by ``python -m customer_churn.dataset`` or ``make data``."""
    make_dataset()


if __name__ == "__main__":
    main()
