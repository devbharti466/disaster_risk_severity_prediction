# Disaster Risk Severity Prediction – India (2000–2025)

A comprehensive machine learning pipeline for predicting disaster risk severity and analyzing disaster patterns across India. The project uses multiple modeling approaches including traditional statistics, deep learning, and spatial analysis.

## Project Structure

```
disaster_risk_severity_prediction/
├── main.py                    # Main orchestration pipeline
├── data_preprocessing.py      # Data loading, cleaning, imputation, feature engineering
├── eda.py                     # Exploratory data analysis visualizations
├── evaluate.py                # Model comparison & results aggregation
├── requirements.txt           # Python dependencies
├── datasets/                  # Raw data
├── models/
│   ├── ordinal_regression.py  # Ordinal risk classification model
│   ├── time_series.py         # ARIMA, SARIMA, Prophet forecasting
│   ├── spatial_model.py       # Spatial analysis (Moran's I, Spatial Lag)
│   ├── lstm_model.py          # LSTM temporal prediction
│   └── convlstm_model.py     # ConvLSTM spatio-temporal prediction
└── results/
    ├── metrics/               # JSON metrics + model_comparison.csv
    └── plots/                 # Visualization PNGs
```

## Setup

### Requirements

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

This sequentially executes:

1. **Data Preprocessing** – cleaning, imputation, feature engineering
2. **Exploratory Data Analysis** – generates 4 visualization plots
3. **Ordinal Risk Classification** – linear regression with threshold-based ordinal classification
4. **Time-Series Forecasting** – ARIMA, SARIMA, and Prophet models
5. **Spatial Analysis** – Moran's I statistic and Spatial Lag Model
6. **LSTM Model** – temporal prediction of monthly disaster counts
7. **ConvLSTM Model** – spatio-temporal binary classification
8. **Model Comparison** – aggregates all metrics into a summary table

## Results

### Model Comparison

| Model              | Task                    | Accuracy | F1    | RMSE  | R²     | AUROC |
|--------------------|-------------------------|----------|-------|-------|--------|-------|
| Ordinal Regression | Classification          | 0.583    | 0.590 | –     | –      | –     |
| ARIMA              | Time-Series Forecast    | –        | –     | 7.64  | –      | –     |
| SARIMA             | Time-Series Forecast    | –        | –     | 9.85  | –      | –     |
| Prophet            | Time-Series Forecast    | –        | –     | 9.82  | –      | –     |
| Spatial Lag Model  | Spatial Classification  | 0.942    | –     | –     | 0.863  | –     |
| LSTM               | Time-Series (DL)        | –        | –     | 7.40  | -0.383 | –     |
| ConvLSTM           | Spatio-Temporal (DL)    | 0.444    | 0.419 | –     | –      | 0.658 |

### Generated Reports

After running the pipeline, the following reports are generated in the `results/` directory:

**Metrics** (`results/metrics/`):
- `ordinal_regression_metrics.json` – Accuracy, Precision, Recall, F1
- `timeseries_metrics.json` – RMSE, MAE, MAPE for ARIMA, SARIMA, Prophet
- `spatial_metrics.json` – Moran's I, p-value, R², accuracy
- `lstm_metrics.json` – RMSE, MAE, R²
- `convlstm_metrics.json` – Accuracy, Precision, Recall, F1, AUROC
- `model_comparison.csv` – Aggregated comparison of all models

**Plots** (`results/plots/`):
- `disaster_type_distribution.png` – Bar chart of disaster types
- `india_disaster_heatmap.png` – Scatter plot of disaster locations by risk level
- `monthly_disaster_trend.png` – Monthly disaster count trend (2000–2025)
- `risk_level_distribution.png` – Pie chart of risk level balance
- `ordinal_confusion_matrix.png` – Confusion matrix for ordinal classification
- `timeseries_forecast.png` – Forecast comparison (ARIMA, SARIMA, Prophet)
- `morans_i_scatter.png` – Moran's I spatial autocorrelation scatter
- `lstm_prediction.png` – LSTM actual vs predicted disaster counts
- `convlstm_confusion_matrix.png` – ConvLSTM binary classification confusion matrix
- `model_accuracy_comparison.png` – Bar chart comparing model accuracies

## Dataset

- **Source**: `datasets/improved_ordinal_regression_dataset_with_location.xlsx - Sheet1.csv`
- **Records**: 2,541 disaster events in India (2000–2025)
- **Features**: Disaster type, location (lat/lon), temporal info, impact metrics (deaths, affected, damage), economic indicators (CPI), pre-computed risk levels

## Models

### 1. Ordinal Risk Classification
Classifies disasters into 3 risk levels (Mild, Moderate, High) using linear regression with threshold-based ordinal mapping. Features include one-hot encoded disaster type, season, location, and numeric features (Duration, CPI, coordinates).

### 2. Time-Series Forecasting
Forecasts monthly disaster counts using three models:
- **ARIMA(2,1,2)** – baseline temporal model
- **SARIMA(1,1,1)(1,1,1,12)** – seasonal model capturing monsoon cycles
- **Prophet** – Meta's Prophet with yearly seasonality

### 3. Spatial Analysis
Analyzes spatial clustering of disaster risk:
- **Moran's I** – tests spatial autocorrelation with permutation-based p-value
- **Spatial Lag Model** – OLS regression with KNN spatial weights (k=5)

### 4. LSTM
Predicts monthly disaster counts using a single-layer LSTM (50 hidden units) with 12-month input sequences. Uses MinMaxScaler fitted only on training data to prevent data leakage.

### 5. ConvLSTM
Spatio-temporal binary classification using a 3-layer ConvLSTM network operating on a 3×3 spatial grid over India with 6 feature channels and 12-month sequences. Handles class imbalance via weighted loss.
