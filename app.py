
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Project paths
ROOT = Path(__file__).resolve().parent

CUSTOMERS_FILE = ROOT / "data" / "customers.csv"
RISK_FILE = ROOT / "reports" / "risk_analysis" / "customer_churn_risk.csv"
SEGMENT_FILE = ROOT / "reports" / "business_impact" / "retention_segments.csv"

# 2. Page settings
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom Customer Churn Dashboard")
st.caption("Customer Analytics | Churn Prediction | Business Impact")

# 3. Check required files
missing_files = [
    str(path.relative_to(ROOT))
    for path in [CUSTOMERS_FILE, RISK_FILE]
    if not path.exists()
]

if missing_files:
    st.error("Required files are missing:")
    for file in missing_files:
        st.write(f"- {file}")
    st.stop()


# 4. Load data
@st.cache_data(ttl="15m", max_entries=4)
def load_data():
    customers = pd.read_csv(CUSTOMERS_FILE)
    risks = pd.read_csv(RISK_FILE)

    if SEGMENT_FILE.exists():
        segments = pd.read_csv(SEGMENT_FILE)
    else:
        segments = pd.DataFrame()

    return customers, risks, segments


try:
    customers, risks, segments = load_data()
except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.stop()

required_columns = {
    "customer_id",
    "monthly_charges",
    "contract_type",
    "churn",
    "churn_probability",
    "risk_category",
}

missing_columns = required_columns - set(risks.columns)

if missing_columns:
    st.error(f"Risk report is missing columns: {sorted(missing_columns)}")
    st.stop()

risks["churn"] = pd.to_numeric(risks["churn"], errors="coerce")
risks["monthly_charges"] = pd.to_numeric(
    risks["monthly_charges"], errors="coerce"
)
risks["churn_probability"] = pd.to_numeric(
    risks["churn_probability"], errors="coerce"
)

risks = risks.dropna(
    subset=["customer_id", "churn", "monthly_charges",
            "churn_probability", "risk_category"]
).drop_duplicates(subset=["customer_id"])

risks["risk_category"] = risks["risk_category"].astype(str)
risks["contract_type"] = risks["contract_type"].fillna("Unknown").astype(str)

if risks.empty:
    st.error("No valid customer predictions were found.")
    st.stop()

# 5. Sidebar filters
st.sidebar.header("Customer Filters")

risk_options = sorted(risks["risk_category"].unique().tolist())
selected_risks = st.sidebar.multiselect(
    "Risk category",
    risk_options,
    default=risk_options,
)

contract_options = sorted(risks["contract_type"].unique().tolist())
selected_contracts = st.sidebar.multiselect(
    "Contract type",
    contract_options,
    default=contract_options,
)

customer_search = st.sidebar.text_input("Search customer ID")

filtered = risks[
    risks["risk_category"].isin(selected_risks)
    & risks["contract_type"].isin(selected_contracts)
].copy()

if customer_search.strip():
    filtered = filtered[
        filtered["customer_id"].astype(str).str.contains(
            customer_search.strip(), case=False, na=False
        )
    ]

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

# 6. KPI metrics
total_customers = len(filtered)
observed_churn = filtered["churn"].mean()
high_risk = (filtered["risk_category"] == "High").sum()
monthly_revenue = filtered["monthly_charges"].sum()
revenue_at_risk = (
    filtered["monthly_charges"] * filtered["churn_probability"]
).sum()
average_probability = filtered["churn_probability"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Observed Churn Rate", f"{observed_churn:.1%}")
col3.metric("High-Risk Customers", f"{high_risk:,}")

col4, col5, col6 = st.columns(3)
col4.metric("Monthly Revenue", f"₹{monthly_revenue:,.2f}")
col5.metric("Expected Monthly Revenue at Risk", f"₹{revenue_at_risk:,.2f}")
col6.metric("Average Predicted Churn", f"{average_probability:.1%}")

st.caption(
    "Metrics update according to the filters. Revenue at risk is an estimate, "
    "not a guarantee of actual revenue loss."
)

st.divider()

# 7. Risk distribution chart
st.subheader("Customer Risk Analysis")

left, right = st.columns(2)

risk_counts = (
    filtered["risk_category"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .rename_axis("Risk Category")
    .reset_index(name="Customers")
)

with left:
    fig_risk = px.bar(
        risk_counts,
        x="Risk Category",
        y="Customers",
        color="Risk Category",
        title="Risk Category Distribution",
        text="Customers",
    )
    fig_risk.update_layout(showlegend=False)
    st.plotly_chart(fig_risk, width="stretch")

with right:
    fig_probability = px.histogram(
        filtered,
        x="churn_probability",
        nbins=20,
        title="Churn Probability Distribution",
        labels={"churn_probability": "Predicted Churn Probability"},
    )
    fig_probability.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig_probability, width="stretch")

# 8. Contract analysis
st.subheader("Churn by Contract Type")

contract_summary = (
    filtered.groupby("contract_type")
    .agg(
        customers=("customer_id", "count"),
        observed_churn_rate=("churn", "mean"),
        average_predicted_probability=("churn_probability", "mean"),
    )
    .reset_index()
)

fig_contract = px.bar(
    contract_summary,
    x="contract_type",
    y="observed_churn_rate",
    title="Observed Churn Rate by Contract",
    text=contract_summary["observed_churn_rate"].map(
        lambda value: f"{value:.1%}"
    ),
    labels={
        "contract_type": "Contract Type",
        "observed_churn_rate": "Observed Churn Rate",
    },
)
fig_contract.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig_contract, width="stretch")

st.dataframe(
    contract_summary.style.format({
        "observed_churn_rate": "{:.1%}",
        "average_predicted_probability": "{:.1%}",
    }),
    hide_index=True,
)

# 9. Customer explorer
st.subheader("Customer Explorer")

display_columns = [
    column
    for column in [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "support_tickets_3m",
        "late_payments_6m",
        "churn",
        "churn_probability",
        "risk_category",
    ]
    if column in filtered.columns
]

customer_table = filtered[display_columns].sort_values(
    "churn_probability", ascending=False
)

st.dataframe(
    customer_table.style.format({
        "monthly_charges": "₹{:.2f}",
        "churn_probability": "{:.1%}",
    }),
    hide_index=True,
)

st.download_button(
    "Download Filtered Customers CSV",
    data=customer_table.to_csv(index=False).encode("utf-8"),
    file_name="filtered_churn_customers.csv",
    mime="text/csv",
)

# 10. Business impact summary
st.subheader("Business Impact Summary")

if (
    not segments.empty
    and {"risk_category", "customers"}.issubset(segments.columns)
):
    st.dataframe(segments, hide_index=True)
else:
    st.info(
        "Business impact report not found. Run Phase 8 to generate the report."
    )

st.divider()
st.caption(
    "Telecom Churn Prediction Project | Synthetic dataset demonstration"
)
