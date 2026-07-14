#!/usr/bin/env python3
"""
Stage 3 - Step 8: ARIMA baseline forecasting
============================================
Per-series ARIMA on a representative subset (top-30 by total sales, among the
series that appear in BOTH sales_filtered_24m & analysis_input, matched by
series_name because the two tables use different platform series_id spaces).

Horizon = 3 months. Auto-selects (p,d,q) by AIC over a small grid, forecasts
the last 3 months, computes MAPE / RMSE / MAE / WMAPE, saves results + plots.

Run:
  python scripts/05_model_arima.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup
from statsmodels.tsa.arima.model import ARIMA

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HORIZON = 3
N_SUBSET = 30
TRAIN_STYLE = "#4C78A8"
TEST_STYLE = "#F58518"
FC_STYLE = "#54A24B"


# ---------- metrics ----------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape}


# ---------- representative subset (deterministic, reused by all stage-3 scripts) ----------
import _subset  # shared stratified ~150-series subset (replaces top-30 representative_subset)


# ---------- ARIMA order search ----------
def auto_order(train):
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(train, order=(p, d, q)).fit()
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order


def main():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    feat = pd.read_csv(FEAT)
    subset = _subset.load_subset()
    print(f"[ARIMA] representative subset: {len(subset)} series")
    if not subset:
        print("[ARIMA] empty subset, abort.")
        return

    rows = []
    examples = []
    preds_rows = []
    for name in subset:
        s = (sales[sales["series_name"].astype(str) == name]
             .sort_values("date").set_index("date")["monthly_sales"]
             .asfreq("MS", fill_value=0))
        if len(s) <= HORIZON + 6:
            rows.append({"series_name": name, "status": "too_short"})
            continue
        train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:]
        try:
            order = auto_order(train)
            fit = ARIMA(train, order=order).fit()
            fc = fit.forecast(HORIZON).clip(lower=0)
            m = metrics(test.values, fc.values)
            m.update({"series_name": name, "order": str(order), "status": "ok"})
            rows.append(m)
            for j, d in enumerate(test.index):
                preds_rows.append({"series_name": name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                    "actual": float(test.values[j]), "pred": float(fc.values[j])})
            if len(examples) < 9:
                examples.append((name, train, test, fc))
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "arima_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "arima_preds.csv"), index=False)

    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"[ARIMA] fitted ok: {len(ok)}/{len(subset)}")
    if len(ok):
        print(f"[ARIMA] mean WMAPE={ok['WMAPE'].mean():.1f}%  "
              f"MAPE={ok['MAPE'].mean():.1f}%  "
              f"RMSE={ok['RMSE'].mean():.1f}  MAE={ok['MAE'].mean():.1f}")

    n_ex = min(9, len(examples))
    if n_ex:
        cols = 3
        rows_n = (n_ex + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 2.6), constrained_layout=True)
        axes = np.array(axes).reshape(-1)
        for i, (name, train, test, fc) in enumerate(examples):
            ax = axes[i]
            ax.plot(train.index, train.values, color=TRAIN_STYLE, lw=1.2, label="train")
            ax.plot(test.index, test.values, color=TEST_STYLE, lw=1.6, marker="o", ms=3, label="actual")
            ax.plot(fc.index, fc.values, color=FC_STYLE, lw=1.6, marker="s", ms=3, label="forecast")
            ax.set_title(name, fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7, loc="upper left")
        for j in range(n_ex, len(axes)):
            axes[j].axis("off")
        fig.suptitle("ARIMA baseline — 3-month forecast (representative subset)", fontsize=11)
        fig.savefig(os.path.join(FIG, "arima_forecast.png"), dpi=130)
        print("[ARIMA] figure saved -> figures/arima_forecast.png")
    print("[ARIMA] done.")


if __name__ == "__main__":
    main()
