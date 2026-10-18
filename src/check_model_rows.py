
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent

customers = pd.read_csv(ROOT / "data" / "customers.csv")
model_data = pd.read_csv(ROOT / "data" / "model_ready.csv")
model = joblib.load(ROOT / "models" / "best_model_phase6.joblib")

print("Customers rows:", len(customers))
print("Model-ready rows:", len(model_data))
print("Model-ready columns:", model_data.columns.tolist())
print("Model type:", type(model).__name__)

if "churn" not in model_data.columns:
    print("ERROR: churn column is missing!")
else:
    X = model_data.drop(columns=["churn"])
    print("Prediction input rows:", len(X))
    print("Prediction input columns:", len(X.columns))

    try:
        probabilities = model.predict_proba(X)
        print("Prediction output shape:", probabilities.shape)
        print("SUCCESS: Model predicted all rows.")
    except Exception as error:
        print("PREDICTION ERROR:", repr(error))
