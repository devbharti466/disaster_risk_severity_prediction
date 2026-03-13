"""
Ordinal Risk Classification Model.

Uses linear regression with threshold-based ordinal classification:
  predicted value < 0.6  → Mild   (0)
  0.6 ≤ predicted < 1.4  → Moderate (1)
  predicted ≥ 1.4        → High   (2)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42
RESULTS_DIR = os.path.join("results", "metrics")
PLOT_DIR = os.path.join("results", "plots")


def _encode_to_ordinal(predicted: np.ndarray) -> np.ndarray:
    """Convert continuous predictions to ordinal classes via thresholds."""
    classes = np.zeros_like(predicted, dtype=int)
    classes[(predicted >= 0.6) & (predicted < 1.4)] = 1
    classes[predicted >= 1.4] = 2
    return classes


def train_ordinal_model(df: pd.DataFrame) -> dict:
    """
    Train the ordinal regression model and return evaluation metrics.
    """
    print("=" * 60)
    print("ORDINAL RISK CLASSIFICATION")
    print("=" * 60)

    df = df.copy()

    # Ensure target is encoded
    if "Risk_Encoded" not in df.columns:
        le = LabelEncoder()
        df["Risk_Encoded"] = le.fit_transform(df["Risk_Level"])

    # ── Feature selection ──
    feature_cols = []

    # One-hot encode Disaster Type
    if "Disaster Type" in df.columns:
        dtype_dummies = pd.get_dummies(df["Disaster Type"], prefix="dtype", sparse=True)
        feature_cols += list(dtype_dummies.columns)
    else:
        dtype_dummies = pd.DataFrame()

    # One-hot encode Season
    if "Season" in df.columns:
        season_dummies = pd.get_dummies(df["Season"], prefix="season", sparse=True)
        feature_cols += list(season_dummies.columns)
    else:
        season_dummies = pd.DataFrame()

    # Location – sparse one-hot (may be large)
    if "Location" in df.columns:
        loc_dummies = pd.get_dummies(df["Location"], prefix="loc", sparse=True)
        feature_cols += list(loc_dummies.columns)
    else:
        loc_dummies = pd.DataFrame()

    # Numeric features
    numeric_feats = ["Duration", "CPI", "Latitude", "Longitude"]
    numeric_feats = [f for f in numeric_feats if f in df.columns]
    feature_cols += numeric_feats

    # Build feature matrix
    X = pd.concat(
        [dtype_dummies, season_dummies, loc_dummies, df[numeric_feats]],
        axis=1,
    )
    y = df["Risk_Encoded"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict + threshold
    y_pred_cont = model.predict(X_test)
    y_pred = _encode_to_ordinal(y_pred_cont)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("\n" + classification_report(
        y_test, y_pred, target_names=["Mild", "Moderate", "High"], zero_division=0,
    ))

    # Confusion matrix plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Mild", "Moderate", "High"],
                yticklabels=["Mild", "Moderate", "High"], ax=ax)
    ax.set_title("Ordinal Classification – Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ordinal_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved confusion matrix plot.")

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open(os.path.join(RESULTS_DIR, "ordinal_regression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    train_ordinal_model(df)
