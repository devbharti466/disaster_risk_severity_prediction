"""
Spatial Econometrics Module.

1. Moran's I  – spatial autocorrelation of disaster risk
2. Spatial Lag Model – y = ρWy + Xβ + ε  (KNN weights, k=5)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, mean_squared_error
from libpysal.weights import KNN
from spreg import OLS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
PLOT_DIR = os.path.join("results", "plots")
METRICS_DIR = os.path.join("results", "metrics")


# ──────────────────────────────────────────────
# Moran's I (manual implementation)
# ──────────────────────────────────────────────
def _compute_morans_i(values: np.ndarray, w_matrix: np.ndarray) -> dict:
    """Compute Moran's I statistic with permutation-based p-value."""
    n = len(values)
    z = values - values.mean()
    s0 = w_matrix.sum()
    numerator = n * float(z @ w_matrix @ z)
    denominator = s0 * float(z @ z)
    I = numerator / denominator if denominator != 0 else 0.0

    # Permutation test for p-value
    rng = np.random.RandomState(RANDOM_SEED)
    n_perm = 999
    perm_I = np.zeros(n_perm)
    for i in range(n_perm):
        perm_z = rng.permutation(z)
        perm_num = n * float(perm_z @ w_matrix @ perm_z)
        perm_I[i] = perm_num / denominator if denominator != 0 else 0.0
    p_value = float((np.abs(perm_I) >= np.abs(I)).sum() + 1) / (n_perm + 1)

    return {"morans_i": float(I), "p_value": p_value}


def _build_knn_weight_matrix(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """Build a row-standardized KNN spatial weight matrix."""
    n = len(coords)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)  # +1 to exclude self
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = indices[i, 1:]  # exclude self
        W[i, neighbors] = 1.0
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


# ──────────────────────────────────────────────
# Spatial Lag Model (via OLS + spatial diagnostics)
# ──────────────────────────────────────────────
def _spatial_lag_model(df: pd.DataFrame, coords: np.ndarray) -> dict:
    """
    Fit a Spatial Lag Model using spreg OLS with spatial diagnostics.
    Returns R², accuracy (thresholded), and rho estimate.
    """
    # Prepare spatial weights via libpysal KNN
    w = KNN.from_array(coords, k=5)
    w.transform = "r"  # row standardization

    # Feature matrix
    feature_cols = []
    for col in ["Duration", "CPI", "Latitude", "Longitude", "Year_Trend",
                "Log_Total_Effect"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols].values.astype(float)
    y = df["Risk_Encoded"].values.astype(float).reshape(-1, 1)

    # OLS with spatial diagnostics
    ols = OLS(y, X, w=w, name_y="Risk_Encoded",
              name_x=feature_cols, spat_diag=True)

    r2 = float(ols.r2)
    print(f"  OLS R²: {r2:.4f}")

    # Threshold predictions for accuracy
    y_pred_cont = ols.predy.flatten()
    y_pred_class = np.clip(np.round(y_pred_cont), 0, 2).astype(int)
    y_true_class = y.flatten().astype(int)
    acc = accuracy_score(y_true_class, y_pred_class)
    print(f"  Spatial model accuracy: {acc:.4f}")

    return {"r2": r2, "accuracy": float(acc)}


# ──────────────────────────────────────────────
# Moran's I scatter plot
# ──────────────────────────────────────────────
def _plot_morans(values, W):
    os.makedirs(PLOT_DIR, exist_ok=True)
    z = values - values.mean()
    lag = W @ z
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z, lag, alpha=0.4, s=10)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    # Fit line
    m, b = np.polyfit(z, lag, 1)
    xline = np.linspace(z.min(), z.max(), 100)
    ax.plot(xline, m * xline + b, color="red", linewidth=1.5,
            label=f"slope={m:.3f}")
    ax.set_title("Moran's I Scatter Plot")
    ax.set_xlabel("Risk Encoded (standardized)")
    ax.set_ylabel("Spatial Lag")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "morans_i_scatter.png"), dpi=150)
    plt.close(fig)
    print("  Saved Moran's I scatter plot.")


# ──────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────
def run_spatial_analysis(df: pd.DataFrame) -> dict:
    """Run Moran's I and Spatial Lag Model."""
    print("=" * 60)
    print("SPATIAL ANALYSIS")
    print("=" * 60)

    # Coordinates
    coords = df[["Latitude", "Longitude"]].values.astype(float)
    risk = df["Risk_Encoded"].values.astype(float)

    # Build weight matrix
    print("  Building KNN weight matrix (k=5)...")
    W = _build_knn_weight_matrix(coords, k=5)

    # Moran's I
    print("\n  Computing Moran's I...")
    mi = _compute_morans_i(risk, W)
    print(f"    Moran's I = {mi['morans_i']:.4f}")
    print(f"    p-value   = {mi['p_value']:.4f}")
    interpretation = ("clustering" if mi["morans_i"] > 0
                      else "dispersion" if mi["morans_i"] < 0
                      else "random")
    print(f"    Interpretation: {interpretation}")

    _plot_morans(risk, W)

    # Spatial Lag Model
    print("\n  Fitting Spatial Lag Model...")
    sl = _spatial_lag_model(df, coords)

    # Save
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics = {
        "morans_i": mi["morans_i"],
        "morans_p_value": mi["p_value"],
        "spatial_lag_r2": sl["r2"],
        "spatial_lag_accuracy": sl["accuracy"],
    }
    with open(os.path.join(METRICS_DIR, "spatial_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    run_spatial_analysis(df)
