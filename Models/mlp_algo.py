import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Hyper‑parameters ────────────────────────────────────────────────────────
BIOCLIM_VARS   = [f"bio_{i}" for i in range(1, 20)]  # 19 features

TRAIN_CSV = Path("finalized_splits_kenya/train_filtered.csv")
VAL_CSV   = Path("finalized_splits_kenya/valid_filtered.csv")   # <── NEW
TEST_CSV  = Path("finalized_splits_kenya/test_filtered.csv")

RANDOM_STATE  = 5341
HIDDEN_UNITS  = 64
MAX_EPOCHS    = 50
BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def load_split(path: Path):
    df = pd.read_csv(path)
    X = df[BIOCLIM_VARS].astype(np.float32).to_numpy()
    df["target"] = df["target"].apply(ast.literal_eval)
    y = np.stack(df["target"].to_numpy()).astype(np.float32)  # (N, S)
    hotspot_ids = df["hotspot_id"].to_numpy()
    return X, y, hotspot_ids

X_train, y_train, _   = load_split(TRAIN_CSV)
X_val,   y_val,  _    = load_split(VAL_CSV)
X_test,  y_test, hid  = load_split(TEST_CSV)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

def make_loader(X, y, batch, shuffle):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()         
        )
    def forward(self, x):
        return self.net(x)

output_dim = y_train.shape[1]
model = MLP(in_dim=X_train.shape[1], hidden=HIDDEN_UNITS, out_dim=output_dim).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

for epoch in range(1, MAX_EPOCHS + 1):
    # ---- train ----
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss  = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds  = model(xb)
            loss   = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"Train BCE: {train_loss:.4f}  •  Val BCE: {val_loss:.4f}")

model.eval()
with torch.no_grad():
    preds_test = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
rmse = np.sqrt(np.mean((y_test - preds_test) ** 2))

for i, hotspot in enumerate(hid):
    np.save(f"/Users/Desktop/Testing_Env/evaluate_results/predictions_kenya/mlp/1/{hotspot}.npy", preds_test[i])
    print(f"Hotspot {hotspot}: max‑pred={preds_test[i].max():.3f}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    },
    "mlp_encounter_rate.pt",
)
