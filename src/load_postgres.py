"""Load project inputs and model scores into PostgreSQL without hard-coded secrets."""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parent.parent
CUSTOMERS_FILE = ROOT / "data" / "customers.csv"
SCORES_FILE = ROOT / "reports" / "risk_analysis" / "customer_churn_risk.csv"
SCHEMA_FILE = ROOT / "sql" / "001_schema.sql"


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL, e.g. postgresql+psycopg://user:password@host:5432/database")
    customers = pd.read_csv(CUSTOMERS_FILE)
    scores = pd.read_csv(SCORES_FILE)[["customer_id", "churn_probability", "risk_category"]]
    engine = create_engine(database_url)
    with engine.begin() as connection:
        for statement in SCHEMA_FILE.read_text(encoding="utf-8").split(";"):
            if statement.strip():
                connection.execute(text(statement))
        # The synthetic source is authoritative; replace rows for repeatable local loads.
        connection.execute(text("DELETE FROM telecom_churn.churn_scores"))
        connection.execute(text("DELETE FROM telecom_churn.customers"))
        customers.to_sql("customers", connection, schema="telecom_churn", if_exists="append", index=False, method="multi")
        scores.to_sql("churn_scores", connection, schema="telecom_churn", if_exists="append", index=False, method="multi")
    print(f"Loaded {len(customers)} customers and {len(scores)} scores into PostgreSQL.")


if __name__ == "__main__":
    main()
