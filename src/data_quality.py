"""Reusable validation for the synthetic customer source data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "customers.csv"


def validate_customer_data(df: pd.DataFrame) -> dict[str, int]:
    """Return a count for each configured data-quality rule.

    The function intentionally does not mutate its input so it can be used by
    the command-line workflow and by pytest.
    """
    required = {
        "customer_id", "tenure_months", "monthly_charges", "total_charges",
        "monthly_data_gb", "support_tickets_3m", "late_payments_6m",
        "senior_citizen", "churn",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    checks = {
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customer_id"].duplicated().sum()),
        "invalid_tenure_months": int((~df["tenure_months"].between(1, 72)).sum()),
        "invalid_monthly_charges": int((~df["monthly_charges"].gt(0)).sum()),
        "invalid_total_charges": int((~df["total_charges"].ge(0)).sum()),
        "invalid_monthly_data_gb": int((~df["monthly_data_gb"].ge(0)).sum()),
        "invalid_support_tickets": int((~df["support_tickets_3m"].ge(0)).sum()),
        "invalid_late_payments": int((~df["late_payments_6m"].ge(0)).sum()),
        "invalid_senior_citizen": int((~df["senior_citizen"].isin([0, 1])).sum()),
        "invalid_churn": int((~df["churn"].isin([0, 1])).sum()),
    }
    return checks


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    checks = validate_customer_data(df)
    total_issues = sum(checks.values())
    print("=" * 55)
    print("TELECOM CUSTOMER DATA QUALITY AUDIT")
    print("=" * 55)
    print(f"\nRows: {len(df)} | Columns: {len(df.columns)}")
    for name, count in checks.items():
        print(f"{name.replace('_', ' ').title()}: {'PASS' if count == 0 else 'FAIL'} ({count})")
    print(f"\nTotal detected quality issues: {total_issues}")
    print("RESULT: Dataset passed configured checks." if total_issues == 0 else "RESULT: Investigate reported issues.")


if __name__ == "__main__":
    main()
