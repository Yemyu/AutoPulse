#!/usr/bin/env python3
"""
Global LSTM + series embedding forecasting
One global LSTM trained across ALL series (pooled), with a learned embedding
per series. 12-month window -> predict next month. Recursive 3-month forecast
on the representative subset (no leakage: future window values come from prior
predictions). Sales normalized per series (log1p + standardize on train part).

Run:
  python scripts/08_model_lstm.py
"""
import os
import warnings
import random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
HORIZON = 3
N_SUBSET = 30
WIN = 12
SEED = 42
EPOCHS = 40
BATCH = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

TRAIN_STYLE = "#4C78A8"
TEST_STYLE = "#F58518"
FC_STYLE = "#72B7B2"


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}


import _subset  # shared stratified ~150-series subset (replaces top-30 representative_subset)


def msin(m):
    return np.sin(2 * np.pi * m / 12.0)


def mcos(m):
    return np.cos(2 * np.pi * m / 12.0)


class LSTMModel(nn.Module):
    def __init__(self, n_series, emb=10, hidden=40):
        super().__init__()
        self.emb = nn.Embedding(n_series, emb)
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + emb + 2, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, xseq, sidx, xmeta):
        e = self.emb(sidx)
        out, _ = self.lstm(xseq)
        h = out[:, -1, :]
        z = torch.cat([h, e, xmeta], dim=1)
        return self.head(z).squeeze(-1)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[LSTM] device: {device}")

    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    feat = pd.read_csv(FEAT)
    subset = _subset.load_subset()
    print(f"[LSTM] representative subset: {len(subset)} series")
    if not subset:
        return

    names = sorted(sales["series_name"].unique())
    name2idx = {n: i for i, n in enumerate(names)}
    print(f"[LSTM] total series pooled: {len(names)}")

    # per-series normalized series
    norm = {}
    for name in names:
        s = (sales[sales["series_name"] == name].sort_values("date")
             .set_index("date")["monthly_sales"].asfreq("MS", fill_value=0))
        vals = np.log1p(s.values.astype(float))
        tr = vals[:-HORIZON] if (name in subset and len(vals) > HORIZON) else vals
        mu, sd = float(tr.mean()), float(tr.std()) + 1e-6
        norm[name] = (mu, sd, (vals - mu) / sd, s.index, s.values.astype(float))

    # training samples
    Xseq, Xmeta, yv, sidxv = [], [], [], []
    for name in names:
        mu, sd, vn, idx, _ = norm[name]
        months = idx.month.values
        T = len(vn)
        for i in range(WIN, T):
            if name in subset and i >= T - HORIZON:
                continue
            seq = np.stack([vn[i - WIN:i], msin(months[i - WIN:i]), mcos(months[i - WIN:i])], axis=1)
            Xseq.append(seq)
            Xmeta.append([msin(months[i]), mcos(months[i])])
            yv.append(vn[i])
            sidxv.append(name2idx[name])
    Xseq = torch.tensor(np.array(Xseq), dtype=torch.float32, device=device)
    Xmeta = torch.tensor(np.array(Xmeta), dtype=torch.float32, device=device)
    yv = torch.tensor(np.array(yv), dtype=torch.float32, device=device)
    sidxv = torch.tensor(np.array(sidxv), dtype=torch.long, device=device)
    print(f"[LSTM] train samples: {len(yv)}")

    model = LSTMModel(len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    lossf = nn.MSELoss()
    N = len(yv)
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        model.train()
        for b in range(0, N, BATCH):
            ix = perm[b:b + BATCH]
            pred = model(Xseq[ix], sidxv[ix], Xmeta[ix])
            loss = lossf(pred, yv[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ix)
        if ep % 10 == 0:
            print(f"[LSTM] epoch {ep:02d}  loss {tot / N:.4f}")

    # recursive forecast on subset
    model.eval()
    rows = []
    examples = []
    preds_rows = []
    with torch.no_grad():
        for name in subset:
            mu, sd, vn, idx, raw = norm[name]
            months = idx.month.values
            T = len(vn)
            if T <= WIN + HORIZON:
                rows.append({"series_name": name, "status": "too_short"})
                continue
            hist = list(vn[:T - HORIZON])
            preds = []
            for step in range(HORIZON):
                i = T - HORIZON + step
                win = np.array(hist[-WIN:], dtype=float)
                wm = months[i - WIN:i]
                seq = torch.tensor(np.stack([win, msin(wm), mcos(wm)], axis=1),
                                   dtype=torch.float32, device=device).unsqueeze(0)
                meta = torch.tensor([[msin(months[i]), mcos(months[i])]],
                                    dtype=torch.float32, device=device)
                si = torch.tensor([name2idx[name]], dtype=torch.long, device=device)
                pn = float(model(seq, si, meta)[0].cpu())
                hist.append(pn)
                log_val = pn * sd + mu
                preds.append(max(float(np.expm1(log_val)), 0.0))
            actuals = raw[T - HORIZON:]
            met = metrics(actuals, preds)
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            for j in range(HORIZON):
                preds_rows.append({"series_name": name,
                                    "date": pd.Timestamp(idx[T - HORIZON + j]).strftime("%Y-%m-%d"),
                                    "actual": float(actuals[j]), "pred": float(preds[j])})
            if len(examples) < 9:
                examples.append((name, pd.Series(raw[:T - HORIZON], index=idx[:T - HORIZON]),
                                 pd.Series(actuals, index=idx[T - HORIZON:]),
                                 pd.Series(preds, index=idx[T - HORIZON:])))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "lstm_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "lstm_preds.csv"), index=False)
    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"[LSTM] forecast ok: {len(ok)}/{len(subset)}")
    if len(ok):
        print(f"[LSTM] mean WMAPE={ok['WMAPE'].mean():.1f}%  "
              f"MAPE={ok['MAPE'].mean():.1f}%  RMSE={ok['RMSE'].mean():.1f}  MAE={ok['MAE'].mean():.1f}")

    n_ex = min(9, len(examples))
    if n_ex:
        cols, rows_n = 3, (n_ex + 2) // 3
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 2.6), constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        for i, (name, tr, te, fc) in enumerate(examples):
            ax = axes[i]
            ax.plot(tr.index, tr.values, color=TRAIN_STYLE, lw=1.2, label="train")
            ax.plot(te.index, te.values, color=TEST_STYLE, lw=1.6, marker="o", ms=3, label="actual")
            ax.plot(fc.index, fc.values, color=FC_STYLE, lw=1.6, marker="s", ms=3, label="forecast")
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")
        for j in range(n_ex, len(axes)):
            axes[j].axis("off")
        fig.suptitle("Global LSTM + series embedding — 3-month forecast (representative subset)", fontsize=11)
        fig.savefig(os.path.join(FIG, "lstm_forecast.png"), dpi=130)
        print("[LSTM] figure saved -> figures/lstm_forecast.png")
    print("[LSTM] done.")


if __name__ == "__main__":
    main()
