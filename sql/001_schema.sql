-- PostgreSQL schema for the Telecom Churn Prediction project.
CREATE SCHEMA IF NOT EXISTS telecom_churn;

CREATE TABLE IF NOT EXISTS telecom_churn.customers (
    customer_id TEXT PRIMARY KEY,
    tenure_months INTEGER NOT NULL CHECK (tenure_months BETWEEN 1 AND 72),
    monthly_charges NUMERIC(10, 2) NOT NULL CHECK (monthly_charges > 0),
    total_charges NUMERIC(12, 2) NOT NULL CHECK (total_charges >= 0),
    contract_type TEXT NOT NULL,
    internet_service TEXT NOT NULL,
    payment_method TEXT NOT NULL,
    support_tickets_3m INTEGER NOT NULL CHECK (support_tickets_3m >= 0),
    monthly_data_gb NUMERIC(10, 2) NOT NULL CHECK (monthly_data_gb >= 0),
    late_payments_6m INTEGER NOT NULL CHECK (late_payments_6m >= 0),
    senior_citizen SMALLINT NOT NULL CHECK (senior_citizen IN (0, 1)),
    churn SMALLINT NOT NULL CHECK (churn IN (0, 1)),
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telecom_churn.churn_scores (
    customer_id TEXT PRIMARY KEY REFERENCES telecom_churn.customers(customer_id),
    churn_probability NUMERIC(6, 5) NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    risk_category TEXT NOT NULL CHECK (risk_category IN ('Low', 'Medium', 'High')),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_contract_type
    ON telecom_churn.customers (contract_type);
CREATE INDEX IF NOT EXISTS idx_churn_scores_risk_category
    ON telecom_churn.churn_scores (risk_category);

CREATE OR REPLACE VIEW telecom_churn.v_risk_summary AS
SELECT
    score.risk_category,
    COUNT(*) AS customers,
    AVG(customer.monthly_charges) AS average_monthly_charges,
    AVG(customer.churn) AS observed_churn_rate,
    AVG(score.churn_probability) AS average_predicted_churn_probability,
    SUM(customer.monthly_charges * score.churn_probability) AS expected_monthly_revenue_at_risk
FROM telecom_churn.customers AS customer
JOIN telecom_churn.churn_scores AS score USING (customer_id)
GROUP BY score.risk_category;
