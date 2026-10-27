"""Train a reproducible, tuned churn model with CV and a final holdout check."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "model_ready.csv"
MODEL_FILE = ROOT / "models" / "best_model_tuned.joblib"
METRICS_FILE = ROOT / "reports" / "metrics" / "tuned_model_evaluation.json"
CV_FILE = ROOT / "reports" / "metrics" / "cross_validation_results.csv"


def build_search() -> GridSearchCV:
    """Return a small, transparent search suitable for this portfolio dataset."""
    estimator = RandomForestClassifier(
        class_weight="balanced", random_state=42, n_jobs=1
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [6, None],
            "min_samples_leaf": [2, 5],
        },
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        # Avoid worker-process startup overhead on Windows for this small dataset.
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | int]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "test_records": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    X, y = df.drop(columns="churn"), df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    search = build_search()
    search.fit(X_train, y_train)
    metrics = evaluate_model(search.best_estimator_, X_test, y_test)
    metrics["best_cv_f1"] = float(search.best_score_)
    metrics["best_parameters"] = search.best_params_

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    cv_results.to_csv(CV_FILE, index=False)
    with METRICS_FILE.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    # Refit only after the unbiased holdout evaluation, for downstream scoring.
    final_model = RandomForestClassifier(
        **search.best_params_, class_weight="balanced", random_state=42, n_jobs=1
    ).fit(X, y)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    print("Tuned model evaluation (held-out test set):")
    print(json.dumps(metrics, indent=2))
    print(f"Saved final scoring model: {MODEL_FILE}")
    print(f"Saved CV results: {CV_FILE}")


if __name__ == "__main__":
    main()
