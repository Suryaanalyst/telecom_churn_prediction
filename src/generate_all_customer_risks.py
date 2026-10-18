
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

customers = pd.read_csv(ROOT / "data" / "customers.csv")
model_data = pd.read_csv(ROOT / "data" / "model_ready.csv")
model = joblib.load(ROOT / "models" / "best_model_phase6.joblib")

if len(customers) != len(model_data):
    raise ValueError("Customer rows and model-ready rows do not match.")

X = model_data.drop(columns=["churn"])
probabilities = model.predict_proba(X)[:, 1]

report = pd.DataFrame({
    "customer_id": customers["customer_id"],
    "tenure_months": customers["tenure_months"],
    "monthly_charges": customers["monthly_charges"],
    "contract_type": customers["contract_type"],
    "support_tickets_3m": customers["support_tickets_3m"],
    "late_payments_6m": customers["late_payments_6m"],
    "churn": customers["churn"],
    "churn_probability": probabilities,
})

report["risk_category"] = pd.cut(
    report["churn_probability"],
    bins=[-0.001, 0.4, 0.7, 1.001],
    labels=["Low", "Medium", "High"],
).astype(str)

output_dir = ROOT / "reports" / "risk_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "customer_churn_risk.csv"
report.to_csv(output_file, index=False)

print("Customers evaluated:", len(report))
print("\nRisk distribution:")
print(report["risk_category"].value_counts())
print("\nSaved report:", output_file)
