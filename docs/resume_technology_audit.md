# Resume technology audit

The résumé lists Python, Pandas, NumPy, SQL, PostgreSQL, Power BI, DAX, Scikit-learn, cross-validation, hyperparameter tuning, SHAP, Plotly, Streamlit, and Pytest for this project.

| Resume technology | Project evidence after this update |
| --- | --- |
| Python, Pandas, NumPy, EDA | Existing generation, quality, EDA, and feature-engineering scripts |
| Scikit-learn classification | Existing baseline/comparison scripts plus `src/train_tuned_model.py` |
| Cross-validation and tuning | Stratified 5-fold `GridSearchCV`; results written to `reports/metrics/cross_validation_results.csv` |
| SHAP | `src/shap_explainability.py`; writes global feature-attribution CSV and chart |
| SQL and PostgreSQL | `sql/001_schema.sql`, `sql/002_analysis_queries.sql`, and `src/load_postgres.py` |
| Power BI and DAX | `docs/power_bi_dax.md` includes model relationship, measures, and report layout |
| Streamlit and Plotly | `app.py` dashboard with cached loading, Plotly charts, filters, and export |
| Pytest | `tests/` covers validation, metric calculations, and headless dashboard rendering |

The implementation now provides evidence for every listed technology. It does not establish a deployed PostgreSQL database, Power BI publication, or production model quality: those require a configured target environment and representative operational data.
