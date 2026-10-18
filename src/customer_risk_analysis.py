
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "model_ready.csv"
CUSTOMER_FILE = PROJECT_ROOT / "data" / "customers.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "best_model_phase6.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "risk_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RISK_REPORT_FILE = REPORT_DIR / "customer_churn_risk.csv"
IMPORTANCE_FILE = REPORT_DIR / "feature_importance.csv"
CHART_FILE = REPORT_DIR / "feature_importance.png"


# 2. Load the dataset and trained model
df = pd.read_csv(DATA_FILE)
customer_data = pd.read_csv(CUSTOMER_FILE)
model = joblib.load(MODEL_FILE)

X = df.drop(columns=["churn"])
y = df["churn"]

# Reproduce the same split used in Phase 6
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Preserve original customer IDs using the dataset row indices.
customer_ids = customer_data.loc[
    X_test.index, "customer_id"
].reset_index(drop=True)

print("=" * 60)
print("TELECOM CHURN - CUSTOMER RISK ANALYSIS")
print("=" * 60)

print(f"\nCustomers evaluated: {len(X_test)}")
print("Using the Phase 6 selected model.")


# 3. Predict churn probabilities on the held-out test set
churn_probabilities = model.predict_proba(X_test)[:, 1]
predictions = model.predict(X_test)

risk_report = pd.DataFrame({
    "customer_id": customer_ids,
    "actual_churn": y_test.reset_index(drop=True),
    "churn_probability": churn_probabilities,
    "predicted_churn": predictions,
})

# 4. Assign business-friendly risk categories
def assign_risk(probability):
    if probability >= 0.60:
        return "High"
    elif probability >= 0.30:
        return "Medium"
    return "Low"


risk_report["risk_category"] = (
    risk_report["churn_probability"].apply(assign_risk)
)

risk_report["churn_probability"] = (
    risk_report["churn_probability"].round(4)
)

# 5. Add useful customer details to the risk report
customer_features = customer_data[
    [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "support_tickets_3m",
        "late_payments_6m",
    ]
]

risk_report = risk_report.merge(
    customer_features,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

# Sort highest-risk customers first
risk_report = risk_report.sort_values(
    "churn_probability",
    ascending=False,
).reset_index(drop=True)

risk_report.to_csv(RISK_REPORT_FILE, index=False)


# 6. Explain predictions using permutation importance
# This measures how much test-set ROC-AUC changes when
# each feature is randomly shuffled.
importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": importance_result.importances_mean,
    "importance_std": importance_result.importances_std,
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=False,
).reset_index(drop=True)

importance_df.to_csv(IMPORTANCE_FILE, index=False)


# 7. Save a feature-importance chart
plot_data = importance_df.head(10).sort_values(
    "importance_mean",
    ascending=True,
)

plt.figure(figsize=(10, 6))
plt.barh(
    plot_data["feature"],
    plot_data["importance_mean"],
)
plt.xlabel("Mean decrease in ROC-AUC when shuffled")
plt.ylabel("Feature")
plt.title("Top 10 Features - Permutation Importance")
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150)
plt.close()


# 8. Print the business summary
print("\n1. CUSTOMER RISK DISTRIBUTION")
print(
    risk_report["risk_category"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .to_string()
)

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
    ].head(10).to_string(index=False)
)

print("\n3. TOP 10 MODEL FEATURES")
print(
    importance_df.head(10).round(4).to_string(index=False)
)

print("\n4. REPORTS SAVED")
print(f"Customer risk report: {RISK_REPORT_FILE}")
print(f"Feature importance: {IMPORTANCE_FILE}")
print(f"Feature chart: {CHART_FILE}")

print("\nCUSTOMER RISK ANALYSIS COMPLETED SUCCESSFULLY")

