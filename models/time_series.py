"""
Time-Series Forecasting Module.

Models:
  1. ARIMA        – baseline temporal forecasting
  2. SARIMA       – seasonal ARIMA (s=12 months for monsoon cycle)
  3. Prophet      – Meta Prophet with trend + seasonality

Target: monthly disaster counts.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
PLOT_DIR = os.path.join("results", "plots")
METRICS_DIR = os.path.join("results", "metrics")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _build_monthly_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate to monthly disaster counts, filling missing months with 0."""
    monthly = (
        df.groupby(["Start Year", "Start Month"])
        .size()
        .reset_index(name="count")
    )
    monthly["date"] = pd.to_datetime(
        monthly["Start Year"].astype(str) + "-"
        + monthly["Start Month"].astype(str).str.zfill(2) + "-01"
    )
    monthly = monthly.set_index("date").sort_index()["count"]

    # Fill missing months
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx, fill_value=0)
    monthly.index.name = "date"
    return monthly


def _adf_test(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series, autolag="AIC")
    info = {
        "ADF Statistic": float(result[0]),
        "p-value": float(result[1]),
        "Stationary": result[1] < 0.05,
    }
    print(f"  ADF Statistic: {info['ADF Statistic']:.4f}")
    print(f"  p-value:       {info['p-value']:.6f}")
    print(f"  Stationary:    {info['Stationary']}")
    return info


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ──────────────────────────────────────────────
# ARIMA
# ──────────────────────────────────────────────
def _fit_arima(train: pd.Series, test: pd.Series) -> dict:
    print("\n  [ARIMA]")
    model = ARIMA(train, order=(2, 1, 2))
    fit = model.fit()
    pred = fit.forecast(steps=len(test))
    pred = np.maximum(pred.values, 0)
    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mae = float(mean_absolute_error(test, pred))
    mape = _mape(test.values, pred)
    print(f"    RMSE: {rmse:.2f}  MAE: {mae:.2f}  MAPE: {mape:.2f}%")
    return {"model": "ARIMA", "rmse": rmse, "mae": mae, "mape": mape,
            "pred": pred, "test_index": test.index}


# ──────────────────────────────────────────────
# SARIMA
# ──────────────────────────────────────────────
def _fit_sarima(train: pd.Series, test: pd.Series) -> dict:
    print("\n  [SARIMA]")
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=len(test))
    pred = np.maximum(pred.values, 0)
    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mae = float(mean_absolute_error(test, pred))
    mape = _mape(test.values, pred)
    print(f"    RMSE: {rmse:.2f}  MAE: {mae:.2f}  MAPE: {mape:.2f}%")
    return {"model": "SARIMA", "rmse": rmse, "mae": mae, "mape": mape,
            "pred": pred, "test_index": test.index}


# ──────────────────────────────────────────────
# Prophet
# ──────────────────────────────────────────────
def _fit_prophet(train: pd.Series, test: pd.Series) -> dict:
    print("\n  [Prophet]")
    prophet_train = pd.DataFrame({
        "ds": train.index,
        "y": train.values,
    })
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(prophet_train)
    future = pd.DataFrame({"ds": test.index})
    forecast = m.predict(future)
    pred = np.maximum(forecast["yhat"].values, 0)
    rmse = float(np.sqrt(mean_squared_error(test, pred)))
    mae = float(mean_absolute_error(test, pred))
    mape = _mape(test.values, pred)
    print(f"    RMSE: {rmse:.2f}  MAE: {mae:.2f}  MAPE: {mape:.2f}%")
    return {"model": "Prophet", "rmse": rmse, "mae": mae, "mape": mape,
            "pred": pred, "test_index": test.index}


# ──────────────────────────────────────────────
# Forecast plot
# ──────────────────────────────────────────────
def _plot_forecast(train, test, results_list):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train.values, label="Train", linewidth=0.8)
    ax.plot(test.index, test.values, label="Actual (test)", linewidth=1.2, color="black")
    for res in results_list:
        ax.plot(res["test_index"], res["pred"],
                label=f'{res["model"]} (RMSE={res["rmse"]:.2f})', linewidth=1)
    ax.set_title("Time-Series Forecast – Monthly Disaster Counts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Disaster Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "timeseries_forecast.png"), dpi=150)
    plt.close(fig)
    print("  Saved forecast plot.")


# ──────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────
def run_time_series(df: pd.DataFrame) -> dict:
    """Run all time-series models and return metrics."""
    print("=" * 60)
    print("TIME-SERIES FORECASTING")
    print("=" * 60)

    series = _build_monthly_series(df)
    print(f"  Monthly series length: {len(series)}")

    # Stationarity check
    _adf_test(series)

    # Chronological train/test split (80/20)
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    print(f"  Train months: {len(train)}, Test months: {len(test)}")

    results = []
    results.append(_fit_arima(train, test))
    results.append(_fit_sarima(train, test))
    results.append(_fit_prophet(train, test))

    _plot_forecast(train, test, results)

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics = {r["model"]: {"rmse": r["rmse"], "mae": r["mae"], "mape": r["mape"]}
               for r in results}
    with open(os.path.join(METRICS_DIR, "timeseries_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    run_time_series(df)
