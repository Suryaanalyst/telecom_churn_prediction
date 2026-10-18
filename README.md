
# Telecom Customer Churn Prediction

## Project Overview

This project predicts customer churn using machine learning and
analyzes customer retention opportunities through business metrics.

The project includes data generation, data quality checks, exploratory
data analysis, feature engineering, model training, risk analysis,
business impact analysis, and an interactive Streamlit dashboard.

## Objectives

- Identify customers who are likely to churn.
- Compare multiple machine learning models.
- Classify customers into Low, Medium, and High risk categories.
- Estimate potential revenue at risk.
- Explore customer insights through an interactive dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Joblib

## Project Structure

```text
telecom-churn-prediction/
├── data/
├── models/
├── reports/
│   ├── figures/
│   ├── metrics/
│   ├── risk_analysis/
│   └── business_impact/
├── src/
├── app.py
├── requirements.txt
└── requirements-dashboard.txt
```

## Project Workflow

1. Generate customer data.
2. Validate data quality.
3. Perform exploratory data analysis.
4. Engineer features.
5. Train a baseline model.
6. Compare machine learning models.
7. Generate customer churn risk predictions.
8. Estimate business impact and retention scenarios.
9. Explore results using a Streamlit dashboard.
10. Document and prepare the project for sharing.

## Key Project Results

- Customers analyzed: 1,000
- High-risk customers identified: 129
- Medium-risk customers identified: 353
- Low-risk customers identified: 518

The above figures are from the current synthetic project dataset.

## Run the Project

Install the dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dashboard.txt
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Important Notes

This project uses synthetic customer data for educational and portfolio
purposes. Revenue-at-risk figures and retention ROI scenarios are
estimates based on assumptions, not guaranteed business outcomes.

Model performance should be evaluated using appropriate validation
data before real-world deployment.

## Author

Add your name and professional links here.
