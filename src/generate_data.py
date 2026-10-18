
from pathlib import Path

import numpy as np
import pandas as pd


# Make the generated data reproducible
rng = np.random.default_rng(42)

# Generate 1,000 customer records
n_customers = 1000

tenure = rng.integers(1, 73, n_customers)
monthly_charges = np.round(rng.uniform(25, 120, n_customers), 2)

contract_type = rng.choice(
    ["Month-to-month", "One year", "Two year"],
    n_customers,
    p=[0.55, 0.25, 0.20],
)

internet_service = rng.choice(
    ["Fiber optic", "DSL", "No internet"],
    n_customers,
    p=[0.50, 0.35, 0.15],
)

payment_method = rng.choice(
    ["Electronic check", "Credit card", "Bank transfer"],
    n_customers,
    p=[0.45, 0.30, 0.25],
)

support_tickets = rng.poisson(1.5, n_customers)
monthly_data_gb = np.round(
    rng.gamma(shape=2.5, scale=20, size=n_customers), 2
)
late_payments = rng.poisson(0.8, n_customers)
senior_citizen = rng.choice([0, 1], n_customers, p=[0.84, 0.16])

# Approximate total customer charges
total_charges = np.round(
    monthly_charges * tenure * rng.uniform(0.90, 1.05, n_customers),
    2,
)

# Create a churn probability using business-related risk factors
risk_score = (
    -2.0
    + 1.0 * (contract_type == "Month-to-month")
    + 0.7 * (internet_service == "Fiber optic")
    + 0.35 * (payment_method == "Electronic check")
    + 0.22 * support_tickets
    + 0.35 * late_payments
    + 0.30 * senior_citizen
    - 0.025 * tenure
    + 0.012 * (monthly_charges - 70)
)

churn_probability = 1 / (1 + np.exp(-risk_score))
churn = rng.binomial(1, churn_probability)

# Build the dataset
customers = pd.DataFrame(
    {
        "customer_id": [
            f"CUST{i:05d}" for i in range(1, n_customers + 1)
        ],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "support_tickets_3m": support_tickets,
        "monthly_data_gb": monthly_data_gb,
        "late_payments_6m": late_payments,
        "senior_citizen": senior_citizen,
        "churn": churn,
    }
)

# Save the dataset in the project's data folder
project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

output_file = data_dir / "customers.csv"
customers.to_csv(output_file, index=False)

# Display a summary
print("Telecom customer dataset generated successfully!")
print(f"Dataset location: {output_file}")
print(f"Total customers: {len(customers)}")
print(f"Total columns: {len(customers.columns)}")
print(f"Customers who churned: {customers['churn'].sum()}")
print(f"Churn rate: {customers['churn'].mean():.2%}")
print("\nFirst 5 customer records:")
print(customers.head().to_string(index=False))

