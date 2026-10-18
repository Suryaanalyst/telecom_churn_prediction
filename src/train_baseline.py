
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "model_ready.csv"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "logistic_regression_baseline.joblib"
METRICS_FILE = METRICS_DIR / "baseline_metrics.json"
CONFUSION_FILE = METRICS_DIR / "baseline_confusion_matrix.csv"


# 2. Load the prepared dataset
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["churn"])
y = df["churn"]

print("=" * 55)
print("TELECOM CHURN - BASELINE MODEL")
print("=" * 55)

print(f"\nTotal records: {len(df)}")
print(f"Input features: {X.shape[1]}")
print(f"Churn rate: {y.mean():.2%}")


# 3. Split data into training and testing sets
# Stratification preserves the approximate churn ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\n1. TRAIN-TEST SPLIT")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# 4. Create a baseline model pipeline
# Scaling is fitted on training data only to avoid data leakage.
model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ]
)

# 5. Train the model
model.fit(X_train, y_train)

print("\n2. MODEL TRAINING")
print("Model: Logistic Regression")
print("Training completed successfully.")


# 6. Generate predictions on unseen test data
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# 7. Evaluate the model
metrics = {
    "model": "Logistic Regression",
    "test_records": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, y_probability)),
}

print("\n3. MODEL EVALUATION")
print("-" * 35)

for metric_name, value in metrics.items():
    if isinstance(value, float):
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    else:
        print(f"{metric_name.replace('_', ' ').title()}: {value}")


# 8. Print the classification report
print("\n4. CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Retained", "Churned"],
        zero_division=0,
    )
)


# 9. Save the confusion matrix
matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual_Retained", "Actual_Churned"],
    columns=["Predicted_Retained", "Predicted_Churned"],
)

matrix_df.to_csv(CONFUSION_FILE)

print("5. CONFUSION MATRIX")
print(matrix_df.to_string())


# 10. Save model and evaluation results
joblib.dump(model, MODEL_FILE)

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n6. OUTPUT FILES")
print(f"Saved model: {MODEL_FILE}")
print(f"Saved metrics: {METRICS_FILE}")
print(f"Saved confusion matrix: {CONFUSION_FILE}")

print("\nBASELINE MODEL COMPLETED SUCCESSFULLY")

