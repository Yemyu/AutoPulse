#!/usr/bin/env python3
"""
Prophet forecasting on representative subset
Per-series Prophet (Facebook) on the same top-30 subset as ARIMA. Horizon=3.
Prophet auto-handles yearly seasonality (car sales low/high season).

Run:
  python scripts/06_model_prophet.py
"""
import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup

# ensure cmdstan backend (one-time compile)
try:
    import cmdstanpy
    try:
        cmdstanpy.cmdstan_path()
    except Exception:
        print("[Prophet] installing cmdstan backend (one-time)...")
        cmdstanpy.install_cmdstan()
except Exception as e:
    print(f"[Prophet] cmdstan setup note: {e}")

from prophet import Prophet

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
HORIZON = 3
N_SUBSET = 30
TRAIN_STYLE = "#4C78A8"
TEST_STYLE = "#F58518"
FC_STYLE = "#9D7EBF"


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


def main():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    feat = pd.read_csv(FEAT)
    subset = _subset.load_subset()
    print(f"[Prophet] representative subset: {len(subset)} series")
    if not subset:
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
        df = pd.DataFrame({"ds": train.index, "y": train.values})
        try:
            m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                        yearly_seasonality=True, seasonality_mode="additive")
            m.fit(df)
            future = m.make_future_dataframe(periods=HORIZON, freq="MS")
            fc = m.predict(future).iloc[-HORIZON:]["yhat"].clip(lower=0).values
            for j, d in enumerate(test.index):
                preds_rows.append({"series_name": name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                    "actual": float(test.values[j]), "pred": float(fc[j])})
            met = metrics(test.values, fc)
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            if len(examples) < 9:
                examples.append((name, train, test, pd.Series(fc, index=test.index)))
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "prophet_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "prophet_preds.csv"), index=False)
    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"[Prophet] fitted ok: {len(ok)}/{len(subset)}")
    if len(ok):
        print(f"[Prophet] mean WMAPE={ok['WMAPE'].mean():.1f}%  "
              f"MAPE={ok['MAPE'].mean():.1f}%  RMSE={ok['RMSE'].mean():.1f}  MAE={ok['MAE'].mean():.1f}")

    n_ex = min(9, len(examples))
    if n_ex:
        cols, rows_n = 3, (n_ex + 2) // 3
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
        fig.suptitle("Prophet — 3-month forecast (representative subset)", fontsize=11)
        fig.savefig(os.path.join(FIG, "prophet_forecast.png"), dpi=130)
        print("[Prophet] figure saved -> figures/prophet_forecast.png")
    print("[Prophet] done.")


if __name__ == "__main__":
    main()
