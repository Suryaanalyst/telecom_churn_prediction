
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CUSTOMER_FILE = PROJECT_ROOT / "data" / "customers.csv"
MODEL_DATA_FILE = PROJECT_ROOT / "data" / "model_ready.csv"
TUNED_MODEL_FILE = PROJECT_ROOT / "models" / "best_model_tuned.joblib"
LEGACY_MODEL_FILE = PROJECT_ROOT / "models" / "best_model_phase6.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "risk_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RISK_REPORT = REPORT_DIR / "customer_churn_risk.csv"
IMPORTANCE_REPORT = REPORT_DIR / "feature_importance.csv"
IMPORTANCE_CHART = REPORT_DIR / "feature_importance.png"


# 2. Load the data and trained model
customers = pd.read_csv(CUSTOMER_FILE)
model_data = pd.read_csv(MODEL_DATA_FILE)
model_file = TUNED_MODEL_FILE if TUNED_MODEL_FILE.exists() else LEGACY_MODEL_FILE
model = joblib.load(model_file)

if len(customers) != len(model_data):
    raise ValueError(
        "Customer data and model-ready data have different row counts."
    )

if "customer_id" not in customers.columns:
    raise ValueError("customer_id missing from customer dataset.")

if "churn" not in model_data.columns:
    raise ValueError("churn target missing from model-ready dataset.")

X = model_data.drop(columns=["churn"])
y = model_data["churn"]

print("=" * 60)
print("TELECOM CUSTOMER RISK ANALYSIS")
print("=" * 60)

print(f"\nTotal customers: {len(customers)}")
print(f"Scoring model loaded: {model_file.name}")


# 3. Predict churn probabilities for ALL customers
# This is for customer prioritization, not unbiased model evaluation.
probabilities = model.predict_proba(X)[:, 1]

risk_report = customers[
    [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "support_tickets_3m",
        "late_payments_6m",
        "churn",
    ]
].copy()

risk_report["churn_probability"] = probabilities

risk_report["risk_category"] = pd.cut(
    risk_report["churn_probability"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
    right=False,
).astype(str)

risk_report = risk_report.sort_values(
    "churn_probability",
    ascending=False,
).reset_index(drop=True)


# 4. Risk distribution
print("\n1. CUSTOMER RISK DISTRIBUTION")
print(
    risk_report["risk_category"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .to_string()
)


# 5. Display the top 10 highest-risk customers
print("\n2. HIGHEST-RISK CUSTOMERS")

print(
    risk_report[
        [
            "customer_id",
            "churn_probability",
            "risk_category",
            "contract_type",
            "monthly_charges",
        ]
    ]
    .head(10)
    .to_string(
        index=False,
        formatters={
            "churn_probability": "{:.4f}".format,
            "monthly_charges": "{:.2f}".format,
        },
    )
)


# 6. Calculate feature importance on a held-out split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_report = pd.DataFrame(
    {
        "feature": X.columns,
        "importance_mean": importance_result.importances_mean,
        "importance_std": importance_result.importances_std,
    }
).sort_values(
    "importance_mean",
    ascending=False,
).reset_index(drop=True)

print("\n3. TOP 10 MODEL FEATURES")
print(importance_report.head(10).round(4).to_string(index=False))


# 7. Save all customer predictions
risk_report.to_csv(RISK_REPORT, index=False)

# Save feature importance
importance_report.to_csv(IMPORTANCE_REPORT, index=False)

# Create a feature importance chart
top_features = importance_report.head(10).sort_values(
    "importance_mean"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
)
plt.xlabel("Mean decrease in ROC-AUC after shuffling")
plt.title("Top 10 Churn Model Features")
plt.tight_layout()
plt.savefig(IMPORTANCE_CHART, dpi=150)
plt.close()


# 8. Final status
print("\n4. REPORTS SAVED")
print(f"Customer risk report: {RISK_REPORT}")
print(f"Feature importance: {IMPORTANCE_REPORT}")
print(f"Feature chart: {IMPORTANCE_CHART}")

print(f"\nCustomers evaluated: {len(risk_report)}")
print("CUSTOMER RISK ANALYSIS COMPLETED SUCCESSFULLY")
