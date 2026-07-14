#!/usr/bin/env python3
"""
Stage 3 — improvement ⑤: Prediction intervals & coverage
=========================================================
Give every model a 90% prediction interval (nominal) on the SAME 3-month
holdout of the 30-series subset, then measure:
  PICP  (Prediction Interval Coverage Probability) = fraction of actuals
         falling inside [lower, upper]; target ≈ 0.90
  MPIW  (Mean Prediction Interval Width) = average interval width
         (reported in raw units and as % of mean actual)

Models & how the interval is obtained:
  ARIMA   : analytic conf_int(alpha=0.10) from get_forecast
  Prophet : yhat_lower / yhat_upper with interval_width=0.9
  XGBoost : 3 quantile regressors (alpha 0.05 / 0.50 / 0.95)
  LSTM    : point forecast ± 1.645 * sigma, where sigma is estimated from
            in-sample (training) one-step residuals

Outputs:
  data/processed/stage3/interval_results.csv
  figures/intervals_coverage.png  +  intervals_example.png

Run:
  python scripts/12_intervals.py
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
from xgboost import XGBRegressor
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HORIZON = 3
N_SUBSET = 30
NOMINAL = 0.90
Z = 1.6448536269514722
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

STATIC_NUM = ["official_price_wan", "avg_rating", "positive_ratio", "negative_ratio", "review_count"]
STATIC_CAT = ["energy_type", "vehicle_class", "brand"]
LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
FEAT_COLS = LAG_COLS + ["month_sin", "month_cos", "year"] + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]


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


def build_row(date, history, sr):
    h = np.array(history, dtype=float)
    return {
        "lag_1": h[-1] if len(h) >= 1 else 0.0, "lag_2": h[-2] if len(h) >= 2 else 0.0,
        "lag_3": h[-3] if len(h) >= 3 else 0.0,
        "roll_mean_3": float(np.mean(h[-3:])) if len(h) >= 1 else 0.0,
        "roll_mean_6": float(np.mean(h[-6:])) if len(h) >= 1 else 0.0,
        "month_sin": np.sin(2 * np.pi * date.month / 12), "month_cos": np.cos(2 * np.pi * date.month / 12),
        "year": date.year,
        "official_price_wan": sr["official_price_wan"], "avg_rating": sr["avg_rating"],
        "positive_ratio": sr["positive_ratio"], "negative_ratio": sr["negative_ratio"],
        "review_count": sr["review_count"],
        "energy_type_enc": sr["energy_type_enc"], "vehicle_class_enc": sr["vehicle_class_enc"],
        "brand_enc": sr["brand_enc"],
    }


def picp_mpiw(actual, lower, upper):
    actual = np.asarray(actual, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    inside = np.mean((actual >= lower) & (actual <= upper))
    width = np.mean(upper - lower)
    width_pct = width / actual.mean() * 100 if actual.mean() != 0 else np.nan
    return inside, width, width_pct


# ---------- ARIMA ----------
def arima_pi(subset, sales):
    al, au, ap, aa = [], [], [], []
    for name in subset:
        s = (sales[sales["series_name"].astype(str) == name].sort_values("date")
             .set_index("date")["monthly_sales"].asfreq("MS", fill_value=0))
        if len(s) <= HORIZON + 6:
            continue
        train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:]
        try:
            order = auto_order(train)
            res = ARIMA(train, order=order).fit()
            fc = res.get_forecast(HORIZON)
            mean = fc.predicted_mean.clip(lower=0).values
            ci = fc.conf_int(alpha=1 - NOMINAL)
            lo = ci.iloc[:, 0].clip(lower=0).values
            hi = ci.iloc[:, 1].clip(lower=0).values
            aa.extend(test.values); ap.extend(mean); al.extend(lo); au.extend(hi)
        except Exception:
            continue
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


# ---------- Prophet ----------
def prophet_pi(subset, sales):
    from prophet import Prophet
    aa, ap, al, au = [], [], [], []
    for name in subset:
        s = (sales[sales["series_name"].astype(str) == name].sort_values("date")
             .set_index("date")["monthly_sales"].asfreq("MS", fill_value=0))
        if len(s) <= HORIZON + 6:
            continue
        train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:]
        try:
            df = pd.DataFrame({"ds": train.index, "y": train.values.astype(float)})
            m = Prophet(interval_width=NOMINAL, yearly_seasonality=True,
                        weekly_seasonality=False, daily_seasonality=False)
            m.fit(df)
            future = m.make_future_dataframe(periods=HORIZON, freq="MS")
            fc = m.predict(future)
            last = fc.iloc[-HORIZON:]
            aa.extend(test.values)
            ap.extend(last["yhat"].clip(lower=0).values)
            al.extend(last["yhat_lower"].clip(lower=0).values)
            au.extend(last["yhat_upper"].clip(lower=0).values)
        except Exception:
            continue
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


# ---------- XGBoost (quantile) ----------
def xgb_pi(subset, sales, feat):
    sales = sales.copy(); sales["series_name"] = sales["series_name"].astype(str)
    feat = feat.copy(); feat["series_name"] = feat["series_name"].astype(str)
    fsub = feat[["series_name"] + STATIC_NUM + STATIC_CAT].copy()
    for c in STATIC_CAT:
        fsub[c] = fsub[c].astype(str).fillna("NA")
        fsub[c + "_enc"] = fsub[c].map({v: i for i, v in enumerate(fsub[c].unique())})
    for c in STATIC_NUM:
        fsub[c] = pd.to_numeric(fsub[c], errors="coerce").fillna(pd.to_numeric(fsub[c], errors="coerce").median())
    static = fsub.set_index("series_name")[[c for c in fsub.columns if c.endswith("_enc") or c in STATIC_NUM]]
    common = set(sales["series_name"]) & set(feat["series_name"])
    sm = sales[sales["series_name"].isin(common)].sort_values(["series_name", "date"]).copy()
    g = sm.groupby("series_name")["monthly_sales"]
    for c in LAG_COLS:
        sm[c] = g.shift(int(c[-1])) if c.startswith("lag") else \
            g.shift(1).rolling(int(c[-1])).mean().reset_index(level=0, drop=True)
    sm["month_of_year"] = sm["date"].dt.month; sm["year"] = sm["date"].dt.year
    sm["month_sin"] = np.sin(2 * np.pi * sm["month_of_year"] / 12)
    sm["month_cos"] = np.cos(2 * np.pi * sm["month_of_year"] / 12)
    sm = sm.merge(static, left_on="series_name", right_index=True, how="left")
    train = sm[sm[FEAT_COLS].notna().all(axis=1)].copy()
    Xtr, ytr = train[FEAT_COLS], np.log1p(train["monthly_sales"].values)

    models = {}
    for a in (0.05, 0.50, 0.95):
        m = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, n_jobs=1,
                         objective="reg:quantileerror", quantile_alpha=a)
        m.fit(Xtr, ytr)
        models[a] = m

    aa, ap, al, au = [], [], [], []
    for name in subset:
        sd = sm[sm["series_name"] == name].sort_values("date")
        if len(sd) <= HORIZON + 6:
            continue
        sr = static.loc[name]
        test_part = sd.iloc[-HORIZON:]
        hist = sd.iloc[:-HORIZON]["monthly_sales"].astype(float).tolist()
        lo = []; pt = []; hi = []
        for _, r in test_part.iterrows():
            row = build_row(r["date"], hist, sr)
            X = pd.DataFrame([row], columns=FEAT_COLS)
            p05 = max(float(np.expm1(models[0.05].predict(X)[0])), 0.0)
            p50 = max(float(np.expm1(models[0.50].predict(X)[0])), 0.0)
            p95 = max(float(np.expm1(models[0.95].predict(X)[0])), 0.0)
            lo.append(p05); pt.append(p50); hi.append(p95)
            hist.append(p50)
        aa.extend(test_part["monthly_sales"].values); ap.extend(pt); al.extend(lo); au.extend(hi)
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


# ---------- LSTM (residual sigma) ----------
def msin(m): return np.sin(2 * np.pi * m / 12.0)
def mcos(m): return np.cos(2 * np.pi * m / 12.0)


class LSTMModel(nn.Module):
    def __init__(self, n_series, emb=10, hidden=40):
        super().__init__()
        self.emb = nn.Embedding(n_series, emb)
        self.lstm = nn.LSTM(3, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + emb + 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xseq, sidx, xmeta):
        e = self.emb(sidx); out, _ = self.lstm(xseq); h = out[:, -1, :]
        return self.head(torch.cat([h, e, xmeta], dim=1)).squeeze(-1)


WIN = 12


def lstm_pi(subset, sales):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sales = sales.copy(); sales["series_name"] = sales["series_name"].astype(str)
    names = sorted(sales["series_name"].unique()); name2idx = {n: i for i, n in enumerate(names)}
    VAL_WINDOW = 12
    norm = {}
    for name in names:
        s = (sales[sales["series_name"] == name].sort_values("date").set_index("date")["monthly_sales"]
             .asfreq("MS", fill_value=0))
        vals = np.log1p(s.values.astype(float))
        tr = vals[:-HORIZON] if (name in subset and len(vals) > HORIZON) else vals
        mu, sd = float(tr.mean()), float(tr.std()) + 1e-6
        norm[name] = (mu, sd, (vals - mu) / sd, s.index, s.values.astype(float))
    Xseq, Xmeta, yv, sidxv = [], [], [], []
    for name in names:
        mu, sd, vn, idx, _ = norm[name]; months = idx.month.values; T = len(vn)
        for i in range(WIN, T):
            if name in subset and i >= T - HORIZON:
                continue
            seq = np.stack([vn[i - WIN:i], msin(months[i - WIN:i]), mcos(months[i - WIN:i])], axis=1)
            Xseq.append(seq); Xmeta.append([msin(months[i]), mcos(months[i])])
            yv.append(vn[i]); sidxv.append(name2idx[name])
    Xseq = torch.tensor(np.array(Xseq), dtype=torch.float32, device=device)
    Xmeta = torch.tensor(np.array(Xmeta), dtype=torch.float32, device=device)
    yv = torch.tensor(np.array(yv), dtype=torch.float32, device=device)
    sidxv = torch.tensor(np.array(sidxv), dtype=torch.long, device=device)
    model = LSTMModel(len(names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005); lossf = nn.MSELoss(); N = len(yv)
    for ep in range(40):
        perm = torch.randperm(N, device=device); tot = 0.0; model.train()
        for b in range(0, N, 256):
            ix = perm[b:b + 256]; pred = model(Xseq[ix], sidxv[ix], Xmeta[ix])
            loss = lossf(pred, yv[ix]); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ix)
    model.eval()

    # estimate sigma from raw-space residuals on the validation window (12m before holdout)
    val_raw_errors = []
    with torch.no_grad():
        for name in subset:
            mu, sd, vn, idx, raw = norm[name]; months = idx.month.values; T = len(vn)
            if T <= WIN + HORIZON + VAL_WINDOW:
                continue
            for step in range(VAL_WINDOW):
                i = T - HORIZON - VAL_WINDOW + step
                wm = months[i - WIN:i]
                seq = torch.tensor(np.stack([vn[i - WIN:i], msin(wm), mcos(wm)], axis=1),
                                   dtype=torch.float32, device=device).unsqueeze(0)
                meta = torch.tensor([[msin(months[i]), mcos(months[i])]], dtype=torch.float32, device=device)
                si = torch.tensor([name2idx[name]], dtype=torch.long, device=device)
                pn = float(model(seq, si, meta)[0].cpu())
                pred_raw = max(float(np.expm1(mu + sd * pn)), 0.0)
                val_raw_errors.append(pred_raw - raw[i])
    sigma_raw = float(np.std(val_raw_errors)) if val_raw_errors else 0.05
    print(f"[PI-LSTM] raw-space validation residual std = {sigma_raw:.2f}")

    aa, ap, al, au = [], [], [], []
    with torch.no_grad():
        for name in subset:
            mu, sd, vn, idx, raw = norm[name]; months = idx.month.values; T = len(vn)
            if T <= WIN + HORIZON:
                continue
            hist = list(vn[:T - HORIZON]); preds = []
            for step in range(HORIZON):
                i = T - HORIZON + step
                wm = months[i - WIN:i]
                seq = torch.tensor(np.stack([np.array(hist[-WIN:]), msin(wm), mcos(wm)], axis=1),
                                   dtype=torch.float32, device=device).unsqueeze(0)
                meta = torch.tensor([[msin(months[i]), mcos(months[i])]], dtype=torch.float32, device=device)
                si = torch.tensor([name2idx[name]], dtype=torch.long, device=device)
                pn = float(model(seq, si, meta)[0].cpu())
                hist.append(pn); preds.append(max(float(np.expm1(pn * sd + mu)), 0.0))
            aa.extend(raw[T - HORIZON:].tolist())
            ap.extend(preds)
            al.extend([max(p - Z * sigma_raw, 0.0) for p in preds])
            au.extend([p + Z * sigma_raw for p in preds])
    return np.array(aa), np.array(ap), np.array(al), np.array(au)


def summarize(name, aa, ap, al, au):
    aa, ap, al, au = map(np.asarray, (aa, ap, al, au))
    picp, width, width_pct = picp_mpiw(aa, al, au)
    wmape = np.sum(np.abs(aa - ap)) / np.sum(np.abs(aa)) * 100 if aa.sum() != 0 else np.nan
    return {"model": name, "PICP": round(picp, 3), "MPIW": round(width, 1),
            "MPIW_pct": round(width_pct, 1), "WMAPE": round(wmape, 1), "n_points": len(aa)}


def main():
    sales = pd.read_csv(SALES); sales["date"] = pd.to_datetime(sales["date"])
    feat = pd.read_csv(FEAT); subset = _subset.load_subset()
    print(f"[PI] subset={len(subset)}; building 90% intervals (nominal={NOMINAL})")

    print("[PI] ARIMA ..."); a = arima_pi(subset, sales)
    print("[PI] Prophet ..."); p = prophet_pi(subset, sales)
    print("[PI] XGBoost (quantile) ..."); x = xgb_pi(subset, sales, feat)
    print("[PI] LSTM (residual sigma) ..."); l = lstm_pi(subset, sales)

    rows = [summarize("ARIMA", *a), summarize("Prophet", *p),
            summarize("XGBoost", *x), summarize("LSTM", *l)]
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "interval_results.csv"), index=False)
    print("\n===== Prediction-interval coverage (90% nominal, 3m holdout) =====")
    print(res.to_string(index=False))

    # figure 1: PICP bars with target line
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#72B7B2"]
    ax.bar(res["model"], res["PICP"], color=colors)
    ax.axhline(NOMINAL, color="red", ls="--", lw=1.2, label=f"target {NOMINAL:.0%}")
    for i, v in enumerate(res["PICP"]):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("PICP (coverage)"); ax.set_title("Prediction-interval coverage (higher≈target)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIG, "intervals_coverage.png"), dpi=130)

    # figure 2: example series with XGBoost interval band
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    name = subset[0]
    s = (sales[sales["series_name"].astype(str) == name].sort_values("date")
         .set_index("date")["monthly_sales"].asfreq("MS", fill_value=0))
    tr, te = s.iloc[:-HORIZON], s.iloc[-HORIZON:]
    ax.plot(tr.index, tr.values, color="#4C78A8", lw=1.3, label="train")
    ax.plot(te.index, te.values, color="#F58518", lw=1.8, marker="o", ms=4, label="actual")
    # pull xgboost point+interval for this series from saved arrays
    idx0 = list(subset).index(name) * HORIZON
    xs = x[1][idx0:idx0 + HORIZON]; xl = x[2][idx0:idx0 + HORIZON]; xu = x[3][idx0:idx0 + HORIZON]
    ax.plot(te.index, xs, color="#54A24B", lw=1.6, marker="s", ms=3, label="XGBoost point")
    ax.fill_between(te.index, xl, xu, color="#54A24B", alpha=0.2, label="XGBoost 90% PI")
    ax.set_title(f"Prediction interval example: {name}"); ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    fig.savefig(os.path.join(FIG, "intervals_example.png"), dpi=130)
    print("[PI] figures saved -> figures/intervals_coverage.png, figures/intervals_example.png")
    print("[PI] done.")


if __name__ == "__main__":
    main()
