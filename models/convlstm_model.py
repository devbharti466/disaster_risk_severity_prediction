"""
ConvLSTM Model for Spatio-Temporal Disaster Occurrence Prediction.

Input:  12 months × 3×3 spatial grid × 6 feature channels
Output: binary (disaster occurrence)

Architecture:
  ConvLSTM2D → BatchNorm → Dropout
  → ConvLSTM2D → BatchNorm → Dropout
  → ConvLSTM2D → Flatten → Dense → Sigmoid

Class imbalance handled via weighted loss.
Chronological split: 70/15/15.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

GRID_SIZE = 3
SEQ_LEN = 12
N_CHANNELS = 6
EPOCHS = 50
BATCH_SIZE = 8
LR = 5e-4
DROPOUT = 0.3

PLOT_DIR = os.path.join("results", "plots")
METRICS_DIR = os.path.join("results", "metrics")


# ──────────────────────────────────────────────
# ConvLSTM Cell
# ──────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMLayer(nn.Module):
    """Process a sequence through a ConvLSTM cell."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)  # (B, T, hidden, H, W)


class ConvLSTMNet(nn.Module):
    def __init__(self, in_channels=N_CHANNELS, grid=GRID_SIZE):
        super().__init__()
        self.layer1 = ConvLSTMLayer(in_channels, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.drop1 = nn.Dropout3d(DROPOUT)

        self.layer2 = ConvLSTMLayer(64, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm3d(64)
        self.drop2 = nn.Dropout3d(DROPOUT)

        self.layer3 = ConvLSTMLayer(64, 64, kernel_size=3)
        self.fc = nn.Linear(64 * grid * grid, 1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        out = self.layer1(x)                    # (B, T, 64, H, W)
        out = out.permute(0, 2, 1, 3, 4)        # (B, 64, T, H, W) for BN3d
        out = self.drop1(self.bn1(out))
        out = out.permute(0, 2, 1, 3, 4)        # back to (B, T, 64, H, W)

        out = self.layer2(out)
        out = out.permute(0, 2, 1, 3, 4)
        out = self.drop2(self.bn2(out))
        out = out.permute(0, 2, 1, 3, 4)

        out = self.layer3(out)
        last = out[:, -1]                        # (B, 64, H, W)
        flat = last.reshape(last.size(0), -1)
        return torch.sigmoid(self.fc(flat)).squeeze(1)


# ──────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────
def _prepare_convlstm_data(df: pd.DataFrame):
    """
    Build spatio-temporal grids.
    1. Discretise lat/lon into a 3×3 grid over India.
    2. For each month, create a 3×3×6 feature tensor.
    3. Build 12-month sequences → binary target (disaster occurred in next month).
    """
    df = df.copy()

    # Grid bins
    lat_bins = np.linspace(df["Latitude"].min(), df["Latitude"].max(), GRID_SIZE + 1)
    lon_bins = np.linspace(df["Longitude"].min(), df["Longitude"].max(), GRID_SIZE + 1)
    df["lat_bin"] = np.clip(np.digitize(df["Latitude"], lat_bins) - 1, 0, GRID_SIZE - 1)
    df["lon_bin"] = np.clip(np.digitize(df["Longitude"], lon_bins) - 1, 0, GRID_SIZE - 1)

    # Monthly aggregation per grid cell
    months = sorted(df.groupby(["Start Year", "Start Month"]).groups.keys())
    grids = []
    for yr, mo in months:
        sub = df[(df["Start Year"] == yr) & (df["Start Month"] == mo)]
        grid = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE))
        for _, row in sub.iterrows():
            r, c = int(row["lat_bin"]), int(row["lon_bin"])
            grid[0, r, c] += 1                                   # event count
            grid[1, r, c] += row.get("Total Deaths", 0)          # deaths
            grid[2, r, c] += row.get("Total Affected", 0)        # affected
            grid[3, r, c] += row.get("Total Damage ('000 US$)", 0)  # damage
            grid[4, r, c] += row.get("CPI", 0)                   # CPI
            grid[5, r, c] += row.get("Historic_Encoded", 0)      # historic flag
        grids.append(grid)

    # Fill in months with no recorded disasters (zero grids)
    all_months = pd.date_range(
        f"{months[0][0]}-{months[0][1]:02d}-01",
        f"{months[-1][0]}-{months[-1][1]:02d}-01",
        freq="MS",
    )
    grids_raw = grids  # save before filling
    month_grid = {}
    for idx, (yr, mo) in enumerate(months):
        month_grid[(yr, mo)] = grids_raw[idx] if idx < len(grids_raw) else np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE))

    grids = []
    for ts in all_months:
        key = (ts.year, ts.month)
        if key in month_grid:
            grids.append(month_grid[key])
        else:
            grids.append(np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE)))

    grids = np.array(grids)  # (T, C, H, W) – includes zero-disaster months

    # Normalize per channel
    for ch in range(N_CHANNELS):
        ch_data = grids[:, ch]
        mx = ch_data.max()
        if mx > 0:
            grids[:, ch] = ch_data / mx

    # Target: whether total event count in next month exceeds the median
    # This creates a balanced binary classification problem
    monthly_counts = grids[:, 0].reshape(len(grids), -1).sum(axis=1)
    threshold = np.median(monthly_counts[monthly_counts > 0]) if (monthly_counts > 0).any() else 0

    X, y = [], []
    for i in range(len(grids) - SEQ_LEN):
        X.append(grids[i : i + SEQ_LEN])
        next_count = grids[i + SEQ_LEN, 0].sum()
        y.append(1 if next_count > threshold else 0)

    X = np.array(X, dtype=np.float32)  # (N, SEQ, C, H, W)
    y = np.array(y, dtype=np.float32)
    return X, y


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def run_convlstm(df: pd.DataFrame) -> dict:
    """Train ConvLSTM and return metrics."""
    print("=" * 60)
    print("CONVLSTM MODEL")
    print("=" * 60)

    X, y = _prepare_convlstm_data(df)
    print(f"  Sequences: {X.shape[0]}, Grid: {GRID_SIZE}×{GRID_SIZE}, "
          f"Channels: {N_CHANNELS}, SeqLen: {SEQ_LEN}")
    print(f"  Class balance: 0={int((y==0).sum())} 1={int((y==1).sum())}")

    # Chronological split 70/15/15
    n = len(X)
    tr = int(n * 0.70)
    va = int(n * 0.85)
    X_train, y_train = X[:tr], y[:tr]
    X_val, y_val = X[tr:va], y[tr:va]
    X_test, y_test = X[va:], y[va:]

    # Class weights for imbalance
    pos_weight_val = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    pos_weight = torch.tensor([pos_weight_val])

    device = torch.device("cpu")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvLSTMNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Override forward to return logits for loss
    class _Wrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            # replace sigmoid in last step
            self.net.fc_raw = nn.Linear(64 * GRID_SIZE * GRID_SIZE, 1)
            self.net.fc_raw.weight = self.net.fc.weight
            self.net.fc_raw.bias = self.net.fc.bias

        def forward(self, x):
            out = self.net.layer1(x)
            out = out.permute(0, 2, 1, 3, 4)
            out = self.net.drop1(self.net.bn1(out))
            out = out.permute(0, 2, 1, 3, 4)
            out = self.net.layer2(out)
            out = out.permute(0, 2, 1, 3, 4)
            out = self.net.drop2(self.net.bn2(out))
            out = out.permute(0, 2, 1, 3, 4)
            out = self.net.layer3(out)
            last = out[:, -1]
            flat = last.reshape(last.size(0), -1)
            return self.net.fc_raw(flat).squeeze(1)  # logits

    wrapper = _Wrapper(model).to(device)

    print(f"  Training ConvLSTM for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        wrapper.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = wrapper(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        wrapper.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = wrapper(xb)
                val_loss += criterion(logits, yb).item()

        avg_train = epoch_loss / max(len(train_dl), 1)
        avg_val = val_loss / max(len(val_dl), 1)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}  "
                  f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val

    # Test evaluation
    wrapper.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            logits = wrapper(xb)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(yb.numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, probs)
    except ValueError:
        auroc = 0.0

    print(f"\n  Test Accuracy:  {acc:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"  AUROC:          {auroc:.4f}")

    # Confusion matrix
    os.makedirs(PLOT_DIR, exist_ok=True)
    cm = confusion_matrix(all_labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disaster", "Disaster"],
                yticklabels=["No Disaster", "Disaster"], ax=ax)
    ax.set_title("ConvLSTM – Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "convlstm_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("  Saved ConvLSTM confusion matrix.")

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
    }
    with open(os.path.join(METRICS_DIR, "convlstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    from data_preprocessing import preprocess
    df = preprocess()
    run_convlstm(df)
