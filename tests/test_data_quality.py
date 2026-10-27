import pandas as pd
import pytest

from src.data_quality import validate_customer_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["C1", "C2"], "tenure_months": [1, 72],
        "monthly_charges": [10.0, 20.0], "total_charges": [10.0, 1440.0],
        "monthly_data_gb": [0.0, 10.0], "support_tickets_3m": [0, 1],
        "late_payments_6m": [0, 1], "senior_citizen": [0, 1], "churn": [0, 1],
    })


def test_valid_customer_data_has_no_issues():
    assert sum(validate_customer_data(valid_frame()).values()) == 0


def test_quality_checks_report_invalid_values_and_duplicates():
    df = valid_frame()
    df.loc[1, "customer_id"] = "C1"
    df.loc[1, "tenure_months"] = 0
    df.loc[1, "churn"] = 2
    result = validate_customer_data(df)
    assert result["duplicate_customer_ids"] == 1
    assert result["invalid_tenure_months"] == 1
    assert result["invalid_churn"] == 1


def test_quality_checks_require_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_customer_data(pd.DataFrame({"customer_id": ["C1"]}))
