
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "model_ready.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_FILE = REPORT_DIR / "model_comparison.csv"
BEST_MODEL_FILE = MODEL_DIR / "best_model_phase6.joblib"
BEST_MODEL_INFO = REPORT_DIR / "best_model_phase6.txt"


# 2. Load the model-ready dataset
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["churn"])
y = df["churn"]

# Use the same split for every model to ensure a fair comparison.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("=" * 60)
print("TELECOM CHURN - MODEL COMPARISON")
print("=" * 60)

print(f"\nTotal customers: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# 3. Define the candidate models
models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    ),
}


# 4. Train and evaluate each model
results = []
trained_models = {}

print("\n1. TRAINING AND EVALUATION")

for name, model in models.items():
    print(f"\nTraining: {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    scores = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    results.append(scores)
    trained_models[name] = model

    print(f"  Accuracy:  {scores['accuracy']:.4f}")
    print(f"  Precision: {scores['precision']:.4f}")
    print(f"  Recall:    {scores['recall']:.4f}")
    print(f"  F1-score:  {scores['f1_score']:.4f}")
    print(f"  ROC-AUC:   {scores['roc_auc']:.4f}")


# 5. Create the comparison report
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="f1_score",
    ascending=False,
).reset_index(drop=True)

comparison.to_csv(COMPARISON_FILE, index=False)

print("\n2. MODEL PERFORMANCE COMPARISON")
print(comparison.round(4).to_string(index=False))


# 6. Select the best model based on F1-score
best_model_name = comparison.iloc[0]["model"]
best_model = trained_models[best_model_name]
best_f1 = comparison.iloc[0]["f1_score"]

joblib.dump(best_model, BEST_MODEL_FILE)

with open(BEST_MODEL_INFO, "w", encoding="utf-8") as file:
    file.write(f"Best model by F1-score: {best_model_name}\n")
    file.write(f"Test F1-score: {best_f1:.4f}\n")
    file.write(
        "Selection is based on the held-out test set; "
        "use validation data for final model selection.\n"
    )

print("\n3. BEST MODEL")
print(f"Selected model: {best_model_name}")
print(f"F1-score: {best_f1:.4f}")

print("\n4. OUTPUT FILES")
print(f"Comparison report: {COMPARISON_FILE}")
print(f"Saved model: {BEST_MODEL_FILE}")
print(f"Selection details: {BEST_MODEL_INFO}")

print("\nMODEL COMPARISON COMPLETED SUCCESSFULLY")
