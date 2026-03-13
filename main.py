"""
Main Pipeline – Disaster Risk Severity Prediction for India.

Orchestrates the entire workflow:
  1. Data Preprocessing (cleaning, imputation, feature engineering)
  2. Exploratory Data Analysis
  3. Ordinal Risk Classification
  4. Time-Series Forecasting (ARIMA, SARIMA, Prophet)
  5. Spatial Analysis (Moran's I, Spatial Lag)
  6. LSTM Temporal Model
  7. ConvLSTM Spatio-Temporal Model
  8. Model Comparison & Final Insights
"""

import sys
import os

# Ensure repo root is on path so model imports work
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import preprocess
from eda import run_eda
from models.ordinal_regression import train_ordinal_model
from models.time_series import run_time_series
from models.spatial_model import run_spatial_analysis
from models.lstm_model import run_lstm
from models.convlstm_model import run_convlstm
from evaluate import compare_models


def main():
    print("╔" + "═" * 58 + "╗")
    print("║  DISASTER RISK SEVERITY PREDICTION – INDIA (2000–2025) ║")
    print("╚" + "═" * 58 + "╝\n")

    # 1. Preprocessing
    df = preprocess()
    print()

    # 2. EDA
    run_eda(df)

    # 3. Ordinal classification
    train_ordinal_model(df)
    print()

    # 4. Time-series forecasting
    run_time_series(df)
    print()

    # 5. Spatial analysis
    run_spatial_analysis(df)
    print()

    # 6. LSTM
    run_lstm(df)
    print()

    # 7. ConvLSTM
    run_convlstm(df)
    print()

    # 8. Comparison
    compare_models()

    print("\n✅ Pipeline complete. Results saved in results/")


if __name__ == "__main__":
    main()
