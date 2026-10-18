
from pathlib import Path

import pandas as pd


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "customers.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "model_ready.csv"

# Load the original dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 55)
print("TELECOM CUSTOMER FEATURE ENGINEERING")
print("=" * 55)

print(f"\nOriginal rows: {df.shape[0]}")
print(f"Original columns: {df.shape[1]}")

# 1. Create useful business features
df["average_monthly_spend"] = (
    df["total_charges"] / df["tenure_months"]
).round(2)

df["high_support_usage"] = (
    df["support_tickets_3m"] >= 3
).astype(int)

df["frequent_late_payments"] = (
    df["late_payments_6m"] >= 2
).astype(int)

df["short_tenure"] = (
    df["tenure_months"] <= 12
).astype(int)

print("\n1. NEW FEATURES CREATED")
print("- average_monthly_spend")
print("- high_support_usage")
print("- frequent_late_payments")
print("- short_tenure")

# 2. Separate target and input features
target = "churn"

# Customer ID identifies a customer but should not be a model feature.
X = df.drop(columns=["customer_id", target])
y = df[target]

# 3. Encode categorical features using one-hot encoding
X_encoded = pd.get_dummies(
    X,
    columns=[
        "contract_type",
        "internet_service",
        "payment_method",
    ],
    dtype=int,
)

# 4. Combine encoded features and target
model_ready = X_encoded.copy()
model_ready[target] = y

# 5. Check the prepared dataset
print("\n2. DATA PREPARATION CHECKS")
print(f"Rows: {model_ready.shape[0]}")
print(f"Model input features: {X_encoded.shape[1]}")
print(f"Target column: {target}")
print(f"Missing values: {model_ready.isnull().sum().sum()}")
print(f"Duplicate rows: {model_ready.duplicated().sum()}")

# 6. Save the model-ready dataset
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
model_ready.to_csv(OUTPUT_FILE, index=False)

print("\n3. DATASET SAVED")
print(f"File: {OUTPUT_FILE}")

print("\n4. ENCODED COLUMN NAMES")
print(model_ready.columns.tolist())

print("\n5. FIRST 5 ROWS")
print(model_ready.head().to_string(index=False))

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY")
