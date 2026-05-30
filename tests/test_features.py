"""
test_features.py
----------------
Unit tests for customer_churn.features
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from customer_churn.features import build_features, encode_categoricals, impute
from customer_churn.config import CATEGORICAL_COLS, NUMERICAL_COLS


@pytest.fixture()
def interim_df() -> pd.DataFrame:
    """Minimal interim DataFrame with missing values."""
    return pd.DataFrame(
        {
            "CustomerID": ["50001", "50002", "50003"],
            "Churn": [1, 0, 1],
            "Tenure": [4.0, None, 6.0],
            "PreferredLoginDevice": ["Mobile Phone", None, "Computer"],
            "CityTier": [1, 3, 2],
            "WarehouseToHome": [6.0, 8.0, None],
            "PreferredPaymentMode": ["Debit Card", "UPI", "Credit Card"],
            "Gender": ["Male", "Female", "Male"],
            "HourSpendOnApp": [3.0, None, 2.0],
            "NumberOfDeviceRegistered": [3, 4, 2],
            "PreferedOrderCat": ["Laptop & Accessory", "Mobile", "Fashion"],
            "SatisfactionScore": [2, 3, 4],
            "MaritalStatus": ["Single", "Divorced", "Married"],
            "NumberOfAddress": [9, 7, 3],
            "Complain": [1, 1, 0],
            "OrderAmountHikeFromlastYear": [11.0, None, 20.0],
            "CouponUsed": [1.0, 0.0, None],
            "OrderCount": [1.0, 2.0, 3.0],
            "DaySinceLastOrder": [5.0, None, 2.0],
            "CashbackAmount": [159.93, 120.90, 200.00],
        }
    )


def test_impute_no_nans(interim_df: pd.DataFrame) -> None:
    """All numerical columns must have zero NaNs after imputation."""
    X = interim_df.drop(columns=["CustomerID", "Churn"]).copy()
    X_imp = impute(X)
    for col in NUMERICAL_COLS:
        if col in X_imp.columns:
            assert X_imp[col].isna().sum() == 0, f"NaN found in '{col}' after imputation"


def test_encode_categoricals_are_integers(interim_df: pd.DataFrame) -> None:
    """All categorical columns must become integer dtype after encoding."""
    X = interim_df.drop(columns=["CustomerID", "Churn"]).copy()
    X = impute(X)
    X = encode_categoricals(X)
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            assert pd.api.types.is_integer_dtype(X[col]), f"'{col}' is not integer after encoding"


def test_build_features_shapes(interim_df: pd.DataFrame) -> None:
    """X and y must have the same length; target and ID must not be in X."""
    X, y = build_features(interim_df)
    assert len(X) == len(y) == len(interim_df)
    assert "Churn" not in X.columns
    assert "CustomerID" not in X.columns


def test_build_features_no_nans(interim_df: pd.DataFrame) -> None:
    """Feature matrix must have no NaN values."""
    X, _ = build_features(interim_df)
    assert X.isna().sum().sum() == 0, "NaN values found in feature matrix after build_features"
