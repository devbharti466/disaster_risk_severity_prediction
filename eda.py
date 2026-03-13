"""
Exploratory Data Analysis module.

Generates key visualizations:
1. Disaster type distribution (bar chart)
2. India disaster heatmap (lat/lon scatter)
3. Monthly disaster trend (line plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = os.path.join("results", "plots")


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_disaster_type_distribution(df: pd.DataFrame):
    """Bar chart of disaster type counts."""
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = df["Disaster Type"].value_counts()
    counts.plot(kind="bar", ax=ax, color=sns.color_palette("viridis", len(counts)))
    ax.set_title("Disaster Type Distribution", fontsize=14)
    ax.set_xlabel("Disaster Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "disaster_type_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_india_heatmap(df: pd.DataFrame):
    """Scatter plot of disaster locations on India map coordinates."""
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(10, 12))
    scatter = ax.scatter(
        df["Longitude"], df["Latitude"],
        c=df["Risk_Encoded"], cmap="RdYlGn_r",
        alpha=0.6, s=15, edgecolors="k", linewidth=0.2,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Risk Level (0=Mild, 1=Moderate, 2=High)")
    ax.set_title("India Disaster Heatmap", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "india_disaster_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_monthly_trend(df: pd.DataFrame):
    """Line plot of monthly disaster counts over time."""
    _ensure_plot_dir()
    if "Start Year" not in df.columns or "Start Month" not in df.columns:
        print("  Skipping monthly trend – missing date columns")
        return
    monthly = (
        df.groupby(["Start Year", "Start Month"])
        .size()
        .reset_index(name="count")
    )
    monthly["date"] = pd.to_datetime(
        monthly["Start Year"].astype(str) + "-"
        + monthly["Start Month"].astype(str).str.zfill(2) + "-01"
    )
    monthly = monthly.sort_values("date")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["date"], monthly["count"], linewidth=0.8)
    ax.set_title("Monthly Disaster Count Trend (2000–2025)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Disaster Count")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "monthly_disaster_trend.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_risk_level_distribution(df: pd.DataFrame):
    """Pie chart of risk level balance."""
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(7, 7))
    df["Risk_Level"].value_counts().plot(
        kind="pie", ax=ax, autopct="%1.1f%%",
        colors=["#2ecc71", "#f39c12", "#e74c3c"],
    )
    ax.set_title("Risk Level Distribution", fontsize=14)
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "risk_level_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def run_eda(df: pd.DataFrame):
    """Execute all EDA visualizations."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    plot_disaster_type_distribution(df)
    plot_india_heatmap(df)
    plot_monthly_trend(df)
    plot_risk_level_distribution(df)
    print("EDA complete.\n")


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    run_eda(df)
