
from pathlib import Path

import pandas as pd


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CUSTOMER_FILE = PROJECT_ROOT / "data" / "customers.csv"
RISK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "risk_analysis"
    / "customer_churn_risk.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "business_impact"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEGMENT_REPORT = REPORT_DIR / "retention_segments.csv"
ROI_REPORT = REPORT_DIR / "retention_roi_scenarios.csv"


# 2. Check whether required files exist
print("=" * 60)
print("TELECOM CHURN - BUSINESS IMPACT ANALYSIS")
print("=" * 60)

if not CUSTOMER_FILE.exists():
    raise FileNotFoundError(
        f"Customer dataset not found: {CUSTOMER_FILE}"
    )

if not RISK_FILE.exists():
    raise FileNotFoundError(
        f"Risk report not found: {RISK_FILE}\n"
        "Run Phase 7 first: python src/customer_risk_analysis.py"
    )


# 3. Load customer and risk data
customers = pd.read_csv(CUSTOMER_FILE)
risk = pd.read_csv(RISK_FILE)

required_risk_columns = {
    "customer_id",
    "churn_probability",
    "risk_category",
}

missing_columns = required_risk_columns - set(risk.columns)

if missing_columns:
    raise ValueError(
        f"Risk report is missing columns: {sorted(missing_columns)}"
    )

if customers["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs found in customers.csv.")

if risk["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs found in the risk report.")

if risk["churn_probability"].isna().any():
    raise ValueError("Risk report contains missing churn probabilities.")

if not risk["churn_probability"].between(0, 1).all():
    raise ValueError("Churn probabilities must be between 0 and 1.")

print(f"\nCustomers in source dataset: {len(customers)}")
print(f"Customers with risk predictions: {len(risk)}")


# 4. Match predictions to customers
df = customers.merge(
    risk[
        ["customer_id", "churn_probability", "risk_category"]
    ],
    on="customer_id",
    how="left",
    validate="one_to_one",
    indicator=True,
)

unmatched = df["_merge"] == "left_only"
unmatched_count = int(unmatched.sum())

if unmatched_count > 0:
    print(
        f"\nERROR: {unmatched_count} customers have no risk prediction."
    )
    print("Business analysis cannot cover the full customer dataset.")
    print("Run Phase 7 again with predictions for all customers.")
    print(f"Missing customer examples: "
          f"{df.loc[unmatched, 'customer_id'].head(10).tolist()}")
    raise ValueError(
        "Risk report does not cover all customers. "
        "No business reports were generated."
    )

df = df.drop(columns=["_merge"])

print(f"Customers successfully matched: {len(df)}")


# 5. Summarize retention segments
segment_summary = (
    df.groupby("risk_category", observed=True)
    .agg(
        customers=("customer_id", "count"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_tenure_months=("tenure_months", "mean"),
        predicted_churn_probability=("churn_probability", "mean"),
        observed_churn_rate=("churn", "mean"),
    )
    .reset_index()
)

segment_summary["estimated_monthly_revenue"] = (
    df.groupby("risk_category", observed=True)["monthly_charges"]
    .sum()
    .reindex(segment_summary["risk_category"])
    .to_numpy()
)

segment_summary["estimated_annual_revenue"] = (
    segment_summary["estimated_monthly_revenue"] * 12
)

segment_summary["predicted_expected_monthly_revenue_at_risk"] = (
    df.assign(
        expected_revenue_at_risk=(
            df["monthly_charges"] * df["churn_probability"]
        )
    )
    .groupby("risk_category", observed=True)["expected_revenue_at_risk"]
    .sum()
    .reindex(segment_summary["risk_category"])
    .to_numpy()
)

segment_summary.to_csv(SEGMENT_REPORT, index=False)

print("\n1. RETENTION SEGMENT SUMMARY")
print(segment_summary.round(2).to_string(index=False))


# 6. Estimate the high-risk customer opportunity
high_risk = df[df["risk_category"] == "High"].copy()

high_risk_customers = len(high_risk)
high_risk_monthly_revenue = high_risk["monthly_charges"].sum()

high_risk_expected_monthly_revenue_at_risk = (
    high_risk["monthly_charges"]
    * high_risk["churn_probability"]
).sum()

print("\n2. HIGH-RISK CUSTOMER OPPORTUNITY")
print(f"High-risk customers: {high_risk_customers}")
print(
    "Current monthly revenue: "
    f"Rs. {high_risk_monthly_revenue:,.2f}"
)
print(
    "Predicted expected monthly revenue at risk: "
    f"Rs. {high_risk_expected_monthly_revenue_at_risk:,.2f}"
)


# 7. Estimate hypothetical retention campaign scenarios
# These are assumptions for planning, not measured outcomes.
contact_cost_per_customer = 100.0
monthly_contribution_margin = 0.60
retained_duration_months = 6

scenarios = []

for save_rate in [0.05, 0.10, 0.20, 0.30]:
    expected_customers_saved = (
        high_risk["churn_probability"] * save_rate
    ).sum()

    campaign_cost = (
        high_risk_customers * contact_cost_per_customer
    )

    expected_contribution_saved = (
        expected_customers_saved
        * high_risk["monthly_charges"].mean()
        * monthly_contribution_margin
        * retained_duration_months
    )

    net_benefit = expected_contribution_saved - campaign_cost

    roi_pct = (
        (net_benefit / campaign_cost) * 100
        if campaign_cost > 0
        else 0.0
    )

    scenarios.append(
        {
            "assumed_save_rate": save_rate,
            "high_risk_customers": high_risk_customers,
            "expected_customers_saved": expected_customers_saved,
            "campaign_cost_rs": campaign_cost,
            "expected_contribution_saved_rs": expected_contribution_saved,
            "net_benefit_rs": net_benefit,
            "roi_pct": roi_pct,
        }
    )

roi_df = pd.DataFrame(scenarios)
roi_df.to_csv(ROI_REPORT, index=False)

print("\n3. HYPOTHETICAL RETENTION CAMPAIGN SCENARIOS")
print(roi_df.round(2).to_string(index=False))


# 8. Print practical business recommendations
print("\n4. RECOMMENDED BUSINESS ACTIONS")

if high_risk_customers > 0:
    print(
        "1. Review high-risk customers and prioritize "
        "appropriate retention outreach."
    )
    print(
        "2. Investigate contract type, pricing, support issues, "
        "and late payments."
    )
    print(
        "3. Test retention offers using a control group "
        "before scaling."
    )
else:
    print(
        "No high-risk customers found. Review model predictions "
        "and risk thresholds."
    )

print(
    "4. Validate actual retention uplift, campaign cost, "
    "and contribution margin before claiming ROI."
)


# 9. Report locations
print("\n5. REPORTS SAVED")
print(f"Segment report: {SEGMENT_REPORT}")
print(f"ROI scenarios: {ROI_REPORT}")

print("\nBUSINESS IMPACT ANALYSIS COMPLETED SUCCESSFULLY")
