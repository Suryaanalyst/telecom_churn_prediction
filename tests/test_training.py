import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.train_tuned_model import evaluate_model


def test_evaluate_model_returns_bounded_metrics():
    X = pd.DataFrame({"a": [0, 1, 0, 1, 0, 1], "b": [1, 1, 0, 0, 1, 0]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    result = evaluate_model(model, X, y)
    assert result["test_records"] == 6
    assert all(0 <= result[key] <= 1 for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"))
