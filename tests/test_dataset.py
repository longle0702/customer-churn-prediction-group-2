"""
test_dataset.py
---------------
Unit tests for customer_churn.dataset
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from customer_churn.dataset import clean  # noqa: E402


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame mimicking the Excel source."""
    return pd.DataFrame(
        {
            "CustomerID": [50001, 50002, 50002],  # duplicate row
            "Churn": [1, 0, 0],
            "Tenure": [4.0, None, None],
            "PreferredLoginDevice": [" Mobile Phone ", "Computer", "Computer"],
            "CityTier": [1, 3, 3],
            "WarehouseToHome": [6.0, None, None],
            "PreferredPaymentMode": ["Debit Card", "UPI", "UPI"],
            "Gender": ["Male", "Female", "Female"],
            "HourSpendOnApp": [3.0, 3.0, 3.0],
            "NumberOfDeviceRegistered": [3, 4, 4],
            "PreferedOrderCat": ["Laptop & Accessory", "Mobile", "Mobile"],
            "SatisfactionScore": [2, 3, 3],
            "MaritalStatus": ["Single", "Divorced", "Divorced"],
            "NumberOfAddress": [9, 7, 7],
            "Complain": [1, 1, 1],
            "OrderAmountHikeFromlastYear": [11.0, 15.0, 15.0],
            "CouponUsed": [1.0, 0.0, 0.0],
            "OrderCount": [1.0, 2.0, 2.0],
            "DaySinceLastOrder": [5.0, 0.0, 0.0],
            "CashbackAmount": [159.93, 120.90, 120.90],
        }
    )


def test_clean_removes_duplicates(raw_df: pd.DataFrame) -> None:
    """Exact duplicate rows must be dropped."""
    result = clean(raw_df)
    assert len(result) == 2


def test_clean_strips_whitespace(raw_df: pd.DataFrame) -> None:
    """Leading/trailing spaces in string columns must be removed."""
    result = clean(raw_df)
    assert result["PreferredLoginDevice"].iloc[0] == "Mobile Phone"


def test_clean_customer_id_is_string(raw_df: pd.DataFrame) -> None:
    """CustomerID must be cast to string after cleaning."""
    result = clean(raw_df)
    assert result["CustomerID"].dtype == object


def test_clean_churn_is_int(raw_df: pd.DataFrame) -> None:
    """Churn column must be integer dtype after cleaning."""
    result = clean(raw_df)
    assert pd.api.types.is_integer_dtype(result["Churn"])
