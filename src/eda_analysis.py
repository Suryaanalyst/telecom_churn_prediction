
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "customers.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Chart appearance
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

print("=" * 55)
print("TELECOM CUSTOMER EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# 1. Overall churn rate
churn_rate = df["churn"].mean() * 100

print("\n1. OVERALL CHURN")
print(f"Total customers: {len(df)}")
print(f"Churned customers: {df['churn'].sum()}")
print(f"Overall churn rate: {churn_rate:.2f}%")

plt.figure()
sns.countplot(data=df, x="churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
plt.close()

# 2. Churn rate by contract type
contract_churn = (
    df.groupby("contract_type")["churn"]
    .agg(["mean", "count"])
    .reset_index()
)
contract_churn["churn_rate_pct"] = contract_churn["mean"] * 100

print("\n2. CHURN BY CONTRACT TYPE")
print(
    contract_churn[
        ["contract_type", "count", "churn_rate_pct"]
    ].to_string(index=False, formatters={
        "churn_rate_pct": "{:.2f}%".format
    })
)

plt.figure()
sns.barplot(
    data=contract_churn,
    x="contract_type",
    y="churn_rate_pct",
)
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_by_contract.png", dpi=150)
plt.close()

# 3. Monthly charges and churn
print("\n3. MONTHLY CHARGES BY CHURN STATUS")
charge_summary = (
    df.groupby("churn")["monthly_charges"]
    .agg(["mean", "median", "count"])
)
print(charge_summary.round(2).to_string())

plt.figure()
sns.boxplot(data=df, x="churn", y="monthly_charges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn (0 = Retained, 1 = Churned)")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "charges_vs_churn.png", dpi=150)
plt.close()

# 4. Tenure and churn
print("\n4. CUSTOMER TENURE BY CHURN STATUS")
tenure_summary = (
    df.groupby("churn")["tenure_months"]
    .agg(["mean", "median", "count"])
)
print(tenure_summary.round(2).to_string())

plt.figure()
sns.histplot(
    data=df,
    x="tenure_months",
    hue="churn",
    bins=18,
    multiple="layer",
    element="step",
)
plt.title("Customer Tenure Distribution by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tenure_vs_churn.png", dpi=150)
plt.close()

# 5. Correlation between numeric variables
print("\n5. NUMERIC CORRELATION WITH CHURN")
numeric_df = df.select_dtypes(include="number")
correlations = (
    numeric_df.corr()["churn"]
    .drop("churn")
    .sort_values(ascending=False)
)
print(correlations.round(3).to_string())

print("\nEDA COMPLETED SUCCESSFULLY")
print(f"Charts saved in: {FIGURES_DIR}")

print("\nGenerated chart files:")
for file_path in sorted(FIGURES_DIR.glob("*.png")):
    print(f"- {file_path.name}")


