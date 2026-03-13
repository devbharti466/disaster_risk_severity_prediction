"""
LSTM Model for Monthly Disaster Count Prediction.

Architecture: Input(12) → LSTM(50) → Dense(1)
Uses MinMaxScaler, chronological split, MSE loss, Adam optimiser.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SEQ_LEN = 12
HIDDEN_SIZE = 50
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3

PLOT_DIR = os.path.join("results", "plots")
METRICS_DIR = os.path.join("results", "metrics")


class DisasterLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _build_monthly_series(df: pd.DataFrame) -> np.ndarray:
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
    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx, fill_value=0)
    return monthly.values.astype(float)


def _create_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def run_lstm(df: pd.DataFrame) -> dict:
    """Train LSTM and return metrics."""
    print("=" * 60)
    print("LSTM MODEL")
    print("=" * 60)

    values = _build_monthly_series(df)
    print(f"  Monthly series length: {len(values)}")

    # Scale only on training data to prevent data leakage
    train_end_idx = int((len(values) - SEQ_LEN) * 0.8) + SEQ_LEN
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end_idx].reshape(-1, 1))
    scaled = scaler.transform(values.reshape(-1, 1)).flatten()

    # Sequences
    X, y = _create_sequences(scaled, SEQ_LEN)
    X = X.reshape(-1, SEQ_LEN, 1)

    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # To torch tensors
    device = torch.device("cpu")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = DisasterLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print(f"  Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}  Loss: {epoch_loss/len(train_dl):.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy().flatten()

    # Inverse transform
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = np.maximum(y_pred, 0)

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))
    print(f"\n  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")

    # Plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(y_actual)), y_actual, label="Actual", linewidth=1.2)
    ax.plot(range(len(y_pred)), y_pred, label=f"LSTM (R²={r2:.2f})", linewidth=1)
    ax.set_title("LSTM – Monthly Disaster Count Prediction")
    ax.set_xlabel("Test Month Index")
    ax.set_ylabel("Disaster Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "lstm_prediction.png"), dpi=150)
    plt.close(fig)
    print("  Saved LSTM prediction plot.")

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    with open(os.path.join(METRICS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    run_lstm(df)
