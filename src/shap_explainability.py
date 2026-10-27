"""Create SHAP feature-attribution artifacts for the tuned scoring model."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "best_model_tuned.joblib"
DATA_FILE = ROOT / "data" / "model_ready.csv"
REPORT_DIR = ROOT / "reports" / "shap"


def main() -> None:
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from error

    model = joblib.load(MODEL_FILE)
    X = pd.read_csv(DATA_FILE).drop(columns="churn")
    sample = X.sample(n=min(250, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    # Binary-class SHAP API output differs across SHAP releases.
    positive_values = values[:, :, 1] if getattr(values, "ndim", 0) == 3 else values
    mean_abs = pd.DataFrame({
        "feature": sample.columns,
        "mean_absolute_shap": abs(positive_values).mean(axis=0),
    }).sort_values("mean_absolute_shap", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mean_abs.to_csv(REPORT_DIR / "global_feature_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(positive_values, sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "global_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP artifacts to {REPORT_DIR}")


if __name__ == "__main__":
    main()
