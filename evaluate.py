"""
Model Evaluation & Comparison Module.

Loads all saved metric JSON files and produces a summary comparison table.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = os.path.join("results", "metrics")
PLOT_DIR = os.path.join("results", "plots")


def compare_models() -> pd.DataFrame:
    """Read all metric files and print a comparison table."""
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    rows = []

    # Ordinal regression
    path = os.path.join(METRICS_DIR, "ordinal_regression_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "Model": "Ordinal Regression",
            "Task": "Classification",
            "Accuracy": m.get("accuracy"),
            "F1": m.get("f1"),
            "RMSE": None,
            "R²": None,
            "AUROC": None,
        })

    # Time series
    path = os.path.join(METRICS_DIR, "timeseries_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        for name, vals in m.items():
            rows.append({
                "Model": name,
                "Task": "Time-Series Forecast",
                "Accuracy": None,
                "F1": None,
                "RMSE": vals.get("rmse"),
                "R²": None,
                "AUROC": None,
            })

    # Spatial
    path = os.path.join(METRICS_DIR, "spatial_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "Model": "Spatial Lag Model",
            "Task": "Spatial Classification",
            "Accuracy": m.get("spatial_lag_accuracy"),
            "F1": None,
            "RMSE": None,
            "R²": m.get("spatial_lag_r2"),
            "AUROC": None,
        })

    # LSTM
    path = os.path.join(METRICS_DIR, "lstm_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "Model": "LSTM",
            "Task": "Time-Series (DL)",
            "Accuracy": None,
            "F1": None,
            "RMSE": m.get("rmse"),
            "R²": m.get("r2"),
            "AUROC": None,
        })

    # ConvLSTM
    path = os.path.join(METRICS_DIR, "convlstm_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "Model": "ConvLSTM",
            "Task": "Spatio-Temporal (DL)",
            "Accuracy": m.get("accuracy"),
            "F1": m.get("f1"),
            "RMSE": None,
            "R²": None,
            "AUROC": m.get("auroc"),
        })

    if not rows:
        print("  No metric files found. Run models first.")
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)
    print("\n" + comparison.to_string(index=False))

    # Save comparison
    os.makedirs(METRICS_DIR, exist_ok=True)
    comparison.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"), index=False)
    print(f"\n  Saved comparison to {METRICS_DIR}/model_comparison.csv")

    # Bar chart of accuracies where available
    acc_df = comparison.dropna(subset=["Accuracy"])
    if not acc_df.empty:
        os.makedirs(PLOT_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(acc_df["Model"], acc_df["Accuracy"], color="steelblue")
        ax.set_xlabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_xlim(0, 1)
        for i, v in enumerate(acc_df["Accuracy"]):
            ax.text(v + 0.01, i, f"{v:.2%}", va="center")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "model_accuracy_comparison.png"), dpi=150)
        plt.close(fig)
        print("  Saved accuracy comparison plot.")

    return comparison


if __name__ == "__main__":
    compare_models()
