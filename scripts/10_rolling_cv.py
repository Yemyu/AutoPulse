#!/usr/bin/env python3
"""
Stage 3 — improvement ②: Rolling-origin / multi-horizon validation
==================================================================
Evaluate ARIMA / XGBoost / LSTM at horizons = 3/6/9/12 months on the SAME
30-series representative subset used by the baseline.

Key trick to keep it fast & leakage-free: train each model ONCE while
withholding the last MAX_EXCLUDE (=12) months of every series, then for each
horizon h forecast exactly the LAST h months (lead-in uses real history,
recursion only inside the h forecast months). This is a proper rolling-origin
style check: "how does error grow as we predict further ahead?"

Outputs:
  data/processed/stage3/cv_results.csv        (model, horizon, mean_wmape, std_wmape, n)
  figures/cv_wmape_by_horizon.png

Run:
  python scripts/10_rolling_cv.py
"""
import os
# 避免 torch 的 OpenMP 与 XGBoost 的 OpenMP 线程池冲突导致段错误 (SIGSEGV)
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
import random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HORIZONS = [3, 6, 9, 12]
N_SUBSET = 30
MAX_EXCLUDE = 12          # withhold this many months for training
LSTM_EPOCHS = 30          # slightly fewer than baseline 40; enough for horizon comparison
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------- shared helpers ----------
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


def auto_order(train):
    best_aic, best = np.inf, (1, 1, 1)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(train, order=(p, d, q)).fit()
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic, best = fit.aic, (p, d, q)
                except Exception:
                    continue
    return best


# ---------- ARIMA (per-series, cheap) ----------
def arima_cv(subset, sales):
    out = {}
    for h in HORIZONS:
        w = []
        for name in subset:
            s = (sales[sales["series_name"].astype(str) == name]
                 .sort_values("date").set_index("date")["monthly_sales"]
                 .asfreq("MS", fill_value=0))
            if len(s) <= h + 6:
                continue
            train, test = s.iloc[:-h], s.iloc[-h:]
            try:
                order = auto_order(train)
                fc = ARIMA(train, order=order).fit().forecast(h).clip(lower=0)
                w.append(metrics(test.values, fc.values)["WMAPE"])
            except Exception:
                continue
        out[h] = (float(np.mean(w)), float(np.std(w)), len(w)) if w else (np.nan, np.nan, 0)
    return out


# ---------- XGBoost (train once, forecast each h) ----------
STATIC_NUM = ["official_price_wan", "avg_rating", "positive_ratio", "negative_ratio", "review_count"]
STATIC_CAT = ["energy_type", "vehicle_class", "brand"]
LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
FEAT_COLS = LAG_COLS + ["month_sin", "month_cos", "year"] + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]


def build_row(date, history, static_row):
    h = np.array(history, dtype=float)
    return {
        "lag_1": h[-1] if len(h) >= 1 else 0.0,
        "lag_2": h[-2] if len(h) >= 2 else 0.0,
        "lag_3": h[-3] if len(h) >= 3 else 0.0,
        "roll_mean_3": float(np.mean(h[-3:])) if len(h) >= 1 else 0.0,
        "roll_mean_6": float(np.mean(h[-6:])) if len(h) >= 1 else 0.0,
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12),
        "year": date.year,
        "official_price_wan": static_row["official_price_wan"],
        "avg_rating": static_row["avg_rating"],
        "positive_ratio": static_row["positive_ratio"],
        "negative_ratio": static_row["negative_ratio"],
        "review_count": static_row["review_count"],
        "energy_type_enc": static_row["energy_type_enc"],
        "vehicle_class_enc": static_row["vehicle_class_enc"],
        "brand_enc": static_row["brand_enc"],
    }


def xgb_cv(subset, sales, feat):
    from xgboost import XGBRegressor
    sales = sales.copy()
    sales["series_name"] = sales["series_name"].astype(str)
    feat = feat.copy()
    feat["series_name"] = feat["series_name"].astype(str)

    fsub = feat[["series_name"] + STATIC_NUM + STATIC_CAT].copy()
    for c in STATIC_CAT:
        fsub[c] = fsub[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(fsub[c].unique())}
        fsub[c + "_enc"] = fsub[c].map(mp)
    for c in STATIC_NUM:
        fsub[c] = pd.to_numeric(fsub[c], errors="coerce")
        fsub[c] = fsub[c].fillna(fsub[c].median())
    static = fsub.set_index("series_name")[[c for c in fsub.columns if c.endswith("_enc") or c in STATIC_NUM]]

    common = set(sales["series_name"]) & set(feat["series_name"])
    sm = sales[sales["series_name"].isin(common)].sort_values(["series_name", "date"]).copy()
    g = sm.groupby("series_name")["monthly_sales"]
    sm["lag_1"] = g.shift(1)
    sm["lag_2"] = g.shift(2)
    sm["lag_3"] = g.shift(3)
    sm["roll_mean_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    sm["roll_mean_6"] = g.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    sm["month_of_year"] = sm["date"].dt.month
    sm["year"] = sm["date"].dt.year
    sm["month_sin"] = np.sin(2 * np.pi * sm["month_of_year"] / 12)
    sm["month_cos"] = np.cos(2 * np.pi * sm["month_of_year"] / 12)
    sm = sm.merge(static, left_on="series_name", right_index=True, how="left")
    sm["rev_rank"] = sm.groupby("series_name").cumcount(ascending=False)
    train = sm[(sm["rev_rank"] >= MAX_EXCLUDE) & sm[FEAT_COLS].notna().all(axis=1)].copy()
    Xtr, ytr = train[FEAT_COLS], np.log1p(train["monthly_sales"].values)
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
    model.fit(Xtr, ytr)
    print(f"[CV-XGB] trained on {len(train)} rows (excluded last {MAX_EXCLUDE}m)")

    out = {}
    for h in HORIZONS:
        w = []
        for name in subset:
            sd = sm[sm["series_name"] == name].sort_values("date")
            if len(sd) <= h + 6:
                continue
            sr = static.loc[name]
            test_part = sd.iloc[-h:]
            history = sd.iloc[:-h]["monthly_sales"].astype(float).tolist()
            preds = []
            for _, r in test_part.iterrows():
                row = build_row(r["date"], history, sr)
                p = max(float(np.expm1(model.predict(pd.DataFrame([row], columns=FEAT_COLS))[0])), 0.0)
                preds.append(p)
                history.append(p)
            w.append(metrics(test_part["monthly_sales"].values, preds)["WMAPE"])
        out[h] = (float(np.mean(w)), float(np.std(w)), len(w)) if w else (np.nan, np.nan, 0)
    return out


# ---------- LSTM (global, train once) ----------
def msin(m): return np.sin(2 * np.pi * m / 12.0)
def mcos(m): return np.cos(2 * np.pi * m / 12.0)


class LSTMModel(nn.Module):
    def __init__(self, n_series, emb=10, hidden=40):
        super().__init__()
        self.emb = nn.Embedding(n_series, emb)
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + emb + 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xseq, sidx, xmeta):
        e = self.emb(sidx)
        out, _ = self.lstm(xseq)
        h = out[:, -1, :]
        z = torch.cat([h, e, xmeta], dim=1)
        return self.head(z).squeeze(-1)


WIN = 12


def lstm_cv(subset, sales):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[CV-LSTM] device: {device}")
    sales = sales.copy()
    sales["series_name"] = sales["series_name"].astype(str)
    names = sorted(sales["series_name"].unique())
    name2idx = {n: i for i, n in enumerate(names)}

    norm = {}
    for name in names:
        s = (sales[sales["series_name"] == name].sort_values("date")
             .set_index("date")["monthly_sales"].asfreq("MS", fill_value=0))
        vals = np.log1p(s.values.astype(float))
        tr = vals[:-MAX_EXCLUDE] if (name in subset and len(vals) > MAX_EXCLUDE) else vals
        mu, sd = float(tr.mean()), float(tr.std()) + 1e-6
        norm[name] = (mu, sd, (vals - mu) / sd, s.index, s.values.astype(float))

    Xseq, Xmeta, yv, sidxv = [], [], [], []
    for name in names:
        mu, sd, vn, idx, _ = norm[name]
        months = idx.month.values
        T = len(vn)
        for i in range(WIN, T):
            if name in subset and i >= T - MAX_EXCLUDE:
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
    print(f"[CV-LSTM] train samples: {len(yv)}")

    model = LSTMModel(len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    lossf = nn.MSELoss()
    N = len(yv)
    for ep in range(LSTM_EPOCHS):
        perm = torch.randperm(N, device=device)
        tot = 0.0
        model.train()
        for b in range(0, N, 256):
            ix = perm[b:b + 256]
            pred = model(Xseq[ix], sidxv[ix], Xmeta[ix])
            loss = lossf(pred, yv[ix])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(ix)
        if ep % 10 == 0:
            print(f"[CV-LSTM] epoch {ep:02d} loss {tot / N:.4f}")
    model.eval()

    out = {}
    with torch.no_grad():
        for h in HORIZONS:
            w = []
            for name in subset:
                mu, sd, vn, idx, raw = norm[name]
                months = idx.month.values
                T = len(vn)
                if T <= WIN + h:
                    continue
                hist = list(vn[:T - h])
                preds = []
                for step in range(h):
                    i = T - h + step
                    wm = months[i - WIN:i]
                    seq = torch.tensor(np.stack([np.array(hist[-WIN:]), msin(wm), mcos(wm)], axis=1),
                                       dtype=torch.float32, device=device).unsqueeze(0)
                    meta = torch.tensor([[msin(months[i]), mcos(months[i])]], dtype=torch.float32, device=device)
                    si = torch.tensor([name2idx[name]], dtype=torch.long, device=device)
                    pn = float(model(seq, si, meta)[0].cpu())
                    hist.append(pn)
                    preds.append(max(float(np.expm1(pn * sd + mu)), 0.0))
                w.append(metrics(raw[T - h:], preds)["WMAPE"])
            out[h] = (float(np.mean(w)), float(np.std(w)), len(w)) if w else (np.nan, np.nan, 0)
    return out


def main():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    feat = pd.read_csv(FEAT)
    subset = _subset.load_subset()
    print(f"[CV] representative subset: {len(subset)} series; horizons={HORIZONS}")

    print("[CV] running ARIMA ...")
    ar = arima_cv(subset, sales)
    print("[CV] running XGBoost ...")
    xg = xgb_cv(subset, sales, feat)
    print("[CV] running LSTM ...")
    ls = lstm_cv(subset, sales)

    rows = []
    for h in HORIZONS:
        for model, d in [("ARIMA", ar), ("XGBoost", xg), ("LSTM", ls)]:
            m, s, n = d[h]
            rows.append({"model": model, "horizon": h, "mean_wmape": m, "std_wmape": s, "n_series": n})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "cv_results.csv"), index=False)
    print("\n===== Rolling-origin CV (WMAPE % by horizon) =====")
    print(res.to_string(index=False))

    # figure: WMAPE vs horizon
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    colors = {"ARIMA": "#F58518", "XGBoost": "#54A24B", "LSTM": "#72B7B2"}
    for model in ["ARIMA", "XGBoost", "LSTM"]:
        sub = res[res["model"] == model]
        ax.plot(sub["horizon"], sub["mean_wmape"], marker="o", lw=2, color=colors[model], label=model)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("WMAPE (%) — lower is better")
    ax.set_title("Rolling-origin CV: error grows with horizon (30 series)")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "cv_wmape_by_horizon.png"), dpi=130)
    print("[CV] figure saved -> figures/cv_wmape_by_horizon.png")
    print("[CV] done.")


if __name__ == "__main__":
    main()
