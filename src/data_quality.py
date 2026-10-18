
from pathlib import Path

import pandas as pd


# Locate the project dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "customers.csv"

# Load the dataset
df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("TELECOM CUSTOMER DATA QUALITY AUDIT")
print("=" * 55)

# 1. Dataset dimensions
print("\n1. DATASET OVERVIEW")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")

# 2. Column names and data types
print("\n2. COLUMN DATA TYPES")
print(df.dtypes.to_string())

# 3. Missing values
print("\n3. MISSING VALUE CHECK")
missing_values = df.isnull().sum()
print(missing_values.to_string())
print(f"Total missing values: {missing_values.sum()}")

# 4. Duplicate checks
print("\n4. DUPLICATE CHECK")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Duplicate customer IDs: {df['customer_id'].duplicated().sum()}")

# 5. Churn distribution
print("\n5. CHURN DISTRIBUTION")
churn_counts = df["churn"].value_counts().reindex(
    [0, 1], fill_value=0
)
churn_rates = df["churn"].value_counts(normalize=True).reindex(
    [0, 1], fill_value=0
) * 100

print(f"Non-churn customers (0): {churn_counts[0]}")
print(f"Churned customers (1): {churn_counts[1]}")
print(f"Overall churn rate: {df['churn'].mean():.2%}")
print(f"Non-churn percentage: {churn_rates[0]:.2f}%")
print(f"Churn percentage: {churn_rates[1]:.2f}%")

# 6. Numeric range checks
print("\n6. NUMERIC VALIDATION")

checks = {
    "tenure_months": df["tenure_months"].between(1, 72),
    "monthly_charges": df["monthly_charges"].gt(0),
    "total_charges": df["total_charges"].ge(0),
    "monthly_data_gb": df["monthly_data_gb"].ge(0),
    "support_tickets_3m": df["support_tickets_3m"].ge(0),
    "late_payments_6m": df["late_payments_6m"].ge(0),
    "senior_citizen": df["senior_citizen"].isin([0, 1]),
    "churn": df["churn"].isin([0, 1]),
}

for column, valid_rows in checks.items():
    invalid_count = int((~valid_rows).sum())
    status = "PASS" if invalid_count == 0 else "FAIL"
    print(f"{column}: {status} | Invalid rows: {invalid_count}")

# 7. Final summary
print("\n7. FINAL AUDIT SUMMARY")

quality_issues = (
    int(df.isnull().sum().sum())
    + int(df.duplicated().sum())
    + int(df["customer_id"].duplicated().sum())
    + sum(int((~valid_rows).sum()) for valid_rows in checks.values())
)

print(f"Total detected quality issues: {quality_issues}")

if quality_issues == 0:
    print("RESULT: Dataset passed the configured quality checks.")
else:
    print("RESULT: Investigate the reported quality issues.")

print("\nAudit completed.")

