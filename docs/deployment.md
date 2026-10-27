# Deployment guide

This project has not been deployed by this repository. The steps below are a reproducible preparation guide, not a deployment claim.

## Local verification

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dashboard.txt
python src\data_quality.py
python src\feature_engineering.py
python src\train_tuned_model.py
python src\customer_risk_analysis.py
python src\business_impact.py
python src\shap_explainability.py
pytest -q
streamlit run app.py
```

## PostgreSQL (optional)

1. Create a PostgreSQL database and set `DATABASE_URL` only in your shell or deployment secret manager.
2. Run the model scoring workflow first so `reports/risk_analysis/customer_churn_risk.csv` exists.
3. Execute `python src/load_postgres.py`. Do not commit a URL, password, or Streamlit `secrets.toml`.
4. Connect Power BI to the `telecom_churn` schema and use the DAX documentation in `power_bi_dax.md`.

## Streamlit hosting checklist

Set Python to a supported version, install both requirements files, and use `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT` if the platform supplies `PORT`. Supply database credentials only through the host's secret configuration. Confirm dashboard files are generated during the build/release step or are supplied from a durable data store. Add CI to run `pytest -q` on every change before promoting a build.
