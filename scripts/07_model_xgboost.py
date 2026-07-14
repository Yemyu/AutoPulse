#!/usr/bin/env python3
"""
Stage 3 - Step 10 (b): XGBoost monthly sales forecasting
========================================================
Monthly-level model. Joins static config+sentiment features (from
analysis_input, by series_name) onto monthly sales, adds calendar +
lag/rolling features. Recursive 3-month forecast on the representative
subset (no leakage: future lags come from predictions, not actuals).

Trained on ALL common series' history; evaluated on the top-30 subset's
last 3 months — same set ARIMA/Prophet use, for apples-to-apples comparison.

Run:
  python scripts/07_model_xgboost.py
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
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES = os.path.join(BASE, "data", "processed", "sales_filtered_24m.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
FIG = os.path.join(BASE, "figures")
PROC = os.path.join(BASE, "data", "processed", "stage3")
os.makedirs(PROC, exist_ok=True)
HORIZON = 3
N_SUBSET = 30

STATIC_NUM = ["official_price_wan", "avg_rating", "positive_ratio", "negative_ratio", "review_count"]
STATIC_CAT = ["energy_type", "vehicle_class", "brand"]
FEAT_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6",
             "month_sin", "month_cos", "year"] + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]


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


def build_row(date, history, static_row):
    h = np.array(history, dtype=float)
    lag1 = h[-1] if len(h) >= 1 else 0.0
    lag2 = h[-2] if len(h) >= 2 else 0.0
    lag3 = h[-3] if len(h) >= 3 else 0.0
    rm3 = float(np.mean(h[-3:])) if len(h) >= 1 else 0.0
    rm6 = float(np.mean(h[-6:])) if len(h) >= 1 else 0.0
    moy = date.month
    return {
        "lag_1": lag1, "lag_2": lag2, "lag_3": lag3,
        "roll_mean_3": rm3, "roll_mean_6": rm6,
        "month_sin": np.sin(2 * np.pi * moy / 12),
        "month_cos": np.cos(2 * np.pi * moy / 12),
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


def main():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    feat = pd.read_csv(FEAT)
    feat["series_name"] = feat["series_name"].astype(str)
    subset = _subset.load_subset()
    print(f"[XGBoost] representative subset: {len(subset)} series")
    if not subset:
        return

    # static features with category encoding
    fsub = feat[["series_name"] + STATIC_NUM + STATIC_CAT].copy()
    for c in STATIC_CAT:
        fsub[c] = fsub[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(fsub[c].unique())}
        fsub[c + "_enc"] = fsub[c].map(mp)
    for c in STATIC_NUM:
        fsub[c] = pd.to_numeric(fsub[c], errors="coerce")
    for c in STATIC_NUM:
        fsub[c] = fsub[c].fillna(fsub[c].median())
    static = fsub.set_index("series_name")[[c for c in fsub.columns if c.endswith("_enc") or c in STATIC_NUM]]

    # monthly table (common series)
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

    # train pool: all rows with valid lags (no leakage — lags are from actual past, all in train)
    train = sm[sm[FEAT_COLS].notna().all(axis=1)].copy()
    Xtr = train[FEAT_COLS]
    ytr = np.log1p(train["monthly_sales"].values)
    print(f"[XGBoost] train rows: {len(train)}")

    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         objective="reg:squarederror", n_jobs=-1)
    model.fit(Xtr, ytr)

    # recursive forecast per subset series
    rows = []
    examples = []
    preds_rows = []
    for name in subset:
        sd = sm[sm["series_name"] == name].sort_values("date")
        if len(sd) <= HORIZON + 6:
            rows.append({"series_name": name, "status": "too_short"})
            continue
        try:
            sr = static.loc[name]
            train_part = sd.iloc[:-HORIZON]
            history = train_part["monthly_sales"].astype(float).tolist()
            test_part = sd.iloc[-HORIZON:]
            preds = []
            for _, r in test_part.iterrows():
                row = build_row(r["date"], history, sr)
                X = pd.DataFrame([row], columns=FEAT_COLS)
                p = float(np.expm1(model.predict(X)[0]))
                p = max(p, 0.0)
                preds.append(p)
                history.append(p)  # feed prediction back as future lag
            for j, d in enumerate(test_part["date"].values):
                preds_rows.append({"series_name": name, "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                                    "actual": float(test_part["monthly_sales"].values[j]),
                                    "pred": float(preds[j])})
            met = metrics(test_part["monthly_sales"].values, preds)
            met.update({"series_name": name, "status": "ok"})
            rows.append(met)
            if len(examples) < 9:
                examples.append((name, train_part.set_index("date")["monthly_sales"],
                                 test_part.set_index("date")["monthly_sales"],
                                 pd.Series(preds, index=test_part["date"])))
        except Exception as e:
            rows.append({"series_name": name, "status": f"error: {type(e).__name__}"})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(PROC, "xgboost_results.csv"), index=False)
    if preds_rows:
        pd.DataFrame(preds_rows).to_csv(os.path.join(PROC, "xgboost_preds.csv"), index=False)
    ok = res[res["status"] == "ok"] if "status" in res.columns else pd.DataFrame()
    print(f"[XGBoost] forecast ok: {len(ok)}/{len(subset)}")
    if len(ok):
        print(f"[XGBoost] mean WMAPE={ok['WMAPE'].mean():.1f}%  "
              f"MAPE={ok['MAPE'].mean():.1f}%  RMSE={ok['RMSE'].mean():.1f}  MAE={ok['MAE'].mean():.1f}")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=FEAT_COLS).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[1].barh(imp.index, imp.values, color="#4C78A8")
    axes[1].set_title("XGBoost feature importance", fontsize=10)
    axes[1].tick_params(labelsize=8)
    n_ex = min(9, len(examples))
    if n_ex:
        cols, rows_n = 3, (n_ex + 2) // 3
        ax = axes[0]
        # show 1 representative series large
        name, tr, te, fc = examples[0]
        ax.plot(tr.index, tr.values, color="#4C78A8", lw=1.4, label="train")
        ax.plot(te.index, te.values, color="#F58518", lw=1.8, marker="o", ms=4, label="actual")
        ax.plot(fc.index, fc.values, color="#E45756", lw=1.8, marker="s", ms=4, label="forecast")
        ax.set_title(f"XGBoost example: {name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
    else:
        axes[0].axis("off")
    fig.suptitle("XGBoost — 3-month recursive forecast + feature importance", fontsize=11)
    fig.savefig(os.path.join(FIG, "xgboost_forecast.png"), dpi=130)
    print("[XGBoost] figure saved -> figures/xgboost_forecast.png")
    print("[XGBoost] done.")


if __name__ == "__main__":
    main()
