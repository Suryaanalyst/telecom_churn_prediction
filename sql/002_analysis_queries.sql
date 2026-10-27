-- Reusable PostgreSQL analysis queries. These also map directly to Power BI visuals.

-- 1. Churn and predicted risk by contract.
SELECT
    customer.contract_type,
    COUNT(*) AS customers,
    ROUND(AVG(customer.churn)::numeric, 4) AS observed_churn_rate,
    ROUND(AVG(score.churn_probability)::numeric, 4) AS predicted_churn_probability
FROM telecom_churn.customers AS customer
JOIN telecom_churn.churn_scores AS score USING (customer_id)
GROUP BY customer.contract_type
ORDER BY observed_churn_rate DESC;

-- 2. Rank the highest-value customers to contact first.
WITH prioritized_customers AS (
    SELECT
        customer.customer_id,
        customer.contract_type,
        customer.monthly_charges,
        score.churn_probability,
        customer.monthly_charges * score.churn_probability AS expected_monthly_revenue_at_risk
    FROM telecom_churn.customers AS customer
    JOIN telecom_churn.churn_scores AS score USING (customer_id)
    WHERE score.risk_category = 'High'
)
SELECT *,
    RANK() OVER (ORDER BY expected_monthly_revenue_at_risk DESC) AS priority_rank
FROM prioritized_customers
ORDER BY priority_rank;

-- 3. Dashboard-ready segment summary.
SELECT *
FROM telecom_churn.v_risk_summary
ORDER BY CASE risk_category WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;
