
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "customers.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
CHART_DIR = OUTPUT_DIR / "charts"

CHART_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Configure chart style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

# Convert churn labels to readable names for charts
df["churn_status"] = df["churn"].map({
    0: "Retained",
    1: "Churned",
})

print("=" * 60)
print("TELECOM CUSTOMER EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Overall business summary
total_customers = len(df)
churned_customers = int(df["churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate = df["churn"].mean() * 100

print("\n1. OVERALL CUSTOMER SUMMARY")
print(f"Total customers: {total_customers}")
print(f"Retained customers: {retained_customers}")
print(f"Churned customers: {churned_customers}")
print(f"Overall churn rate: {churn_rate:.2f}%")

# 2. Churn by contract type
contract_summary = (
    df.groupby("contract_type", observed=True)
    .agg(
        total_customers=("churn", "size"),
        churned_customers=("churn", "sum"),
        churn_rate=("churn", "mean"),
    )
    .reset_index()
)

contract_summary["churn_rate"] *= 100
contract_summary.to_csv(
    OUTPUT_DIR / "churn_by_contract.csv", index=False
)

print("\n2. CHURN BY CONTRACT TYPE")
print(contract_summary.round(2).to_string(index=False))

# 3. Churn by internet service
internet_summary = (
    df.groupby("internet_service", observed=True)
    .agg(
        total_customers=("churn", "size"),
        churn_rate=("churn", "mean"),
    )
    .reset_index()
)

internet_summary["churn_rate"] *= 100
internet_summary.to_csv(
    OUTPUT_DIR / "churn_by_internet_service.csv", index=False
)

print("\n3. CHURN BY INTERNET SERVICE")
print(internet_summary.round(2).to_string(index=False))

# 4. Churn by payment method
payment_summary = (
    df.groupby("payment_method", observed=True)
    .agg(
        total_customers=("churn", "size"),
        churn_rate=("churn", "mean"),
    )
    .reset_index()
)

payment_summary["churn_rate"] *= 100
payment_summary.to_csv(
    OUTPUT_DIR / "churn_by_payment_method.csv", index=False
)

print("\n4. CHURN BY PAYMENT METHOD")
print(payment_summary.round(2).to_string(index=False))

# 5. Compare numerical behavior by churn status
numeric_summary = (
    df.groupby("churn_status", observed=True)[
        [
            "tenure_months",
            "monthly_charges",
            "support_tickets_3m",
            "late_payments_6m",
        ]
    ]
    .mean()
    .round(2)
)

numeric_summary.to_csv(
    OUTPUT_DIR / "numeric_summary_by_churn.csv"
)

print("\n5. AVERAGE CUSTOMER METRICS BY CHURN STATUS")
print(numeric_summary.to_string())

# 6. Chart 1: Churn distribution
plt.figure()
sns.countplot(
    data=df,
    x="churn_status",
    hue="churn_status",
    order=["Retained", "Churned"],
    palette="Set2",
    legend=False,
)
plt.title("Customer Churn Distribution")
plt.xlabel("Customer status")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(CHART_DIR / "01_churn_distribution.png", dpi=150)
plt.close()

# 7. Chart 2: Churn rate by contract type
plt.figure()
sns.barplot(
    data=contract_summary,
    x="contract_type",
    y="churn_rate",
    hue="contract_type",
    palette="Blues",
    legend=False,
)
plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract type")
plt.ylabel("Churn rate (%)")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_churn_by_contract.png", dpi=150)
plt.close()

# 8. Chart 3: Tenure versus churn
plt.figure()
sns.boxplot(
    data=df,
    x="churn_status",
    y="tenure_months",
    hue="churn_status",
    order=["Retained", "Churned"],
    palette="Set2",
    legend=False,
)
plt.title("Customer Tenure by Churn Status")
plt.xlabel("Customer status")
plt.ylabel("Tenure (months)")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_tenure_by_churn.png", dpi=150)
plt.close()

# 9. Chart 4: Monthly charges versus churn
plt.figure()
sns.boxplot(
    data=df,
    x="churn_status",
    y="monthly_charges",
    hue="churn_status",
    order=["Retained", "Churned"],
    palette="Set2",
    legend=False,
)
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Customer status")
plt.ylabel("Monthly charges")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_charges_by_churn.png", dpi=150)
plt.close()

# 10. Identify the highest observed churn-rate categories
print("\n6. KEY OBSERVATIONS")

highest_contract = contract_summary.loc[
    contract_summary["churn_rate"].idxmax()
]
highest_internet = internet_summary.loc[
    internet_summary["churn_rate"].idxmax()
]
highest_payment = payment_summary.loc[
    payment_summary["churn_rate"].idxmax()
]

print(
    f"Highest contract churn rate: {highest_contract['contract_type']} "
    f"({highest_contract['churn_rate']:.2f}%)"
)
print(
    f"Highest internet-service churn rate: "
    f"{highest_internet['internet_service']} "
    f"({highest_internet['churn_rate']:.2f}%)"
)
print(
    f"Highest payment-method churn rate: "
    f"{highest_payment['payment_method']} "
    f"({highest_payment['churn_rate']:.2f}%)"
)

print("\n7. OUTPUT FILES")
print(f"Analysis tables: {OUTPUT_DIR}")
print(f"Charts: {CHART_DIR}")
print("\nEDA completed successfully.")
