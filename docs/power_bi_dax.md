# Power BI and DAX guide

## Model

Import `customers.csv` and `reports/risk_analysis/customer_churn_risk.csv`. Relate the tables on `customer_id` with a one-to-one relationship. Use a single filter direction from Customers to Risk Scores. For PostgreSQL, use the `telecom_churn.customers`, `telecom_churn.churn_scores`, and `telecom_churn.v_risk_summary` objects instead.

## Measures

```DAX
Customers = DISTINCTCOUNT(Customers[customer_id])

Churned Customers = CALCULATE([Customers], Customers[churn] = 1)

Observed Churn Rate = DIVIDE([Churned Customers], [Customers])

Monthly Revenue = SUM(Customers[monthly_charges])

Expected Monthly Revenue at Risk =
SUMX(
    'Customer Churn Risk',
    'Customer Churn Risk'[monthly_charges] * 'Customer Churn Risk'[churn_probability]
)

High-risk Customers =
CALCULATE([Customers], 'Customer Churn Risk'[risk_category] = "High")

Average Predicted Churn = AVERAGE('Customer Churn Risk'[churn_probability])
```

## Suggested report pages

1. **Executive overview:** KPI cards for customers, observed churn, monthly revenue, expected revenue at risk, and high-risk customers.
2. **Risk and retention:** risk-category distribution, high-risk customer table sorted by expected revenue at risk, plus contract and payment-method slicers.
3. **Driver analysis:** observed churn and average predicted churn by contract type, internet service, and tenure band.

Format churn and probability measures as percentages, and label expected revenue at risk as an estimate rather than realized loss. The project data is synthetic, so refresh and validate measures before business use.
