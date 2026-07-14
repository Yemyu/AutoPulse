#!/usr/bin/env python3
"""
Stage 3 — improvement ④: XGBoost ablation (lag features vs pure features)
=========================================================================
Run XGBoost twice on the same 30-series subset, same 3-month holdout:
  (A) FULL      : lag_1/2/3 + roll_mean_3/6 + calendar + static   (baseline)
  (B) NO-LAG    : calendar + static only  (no recent-sales info)

The gap quantifies how much the autoregressive (recent-actuals) signal
contributes versus the static config/sentiment features alone. Recursive
forecast, no future leakage.

Outputs:
  data/processed/stage3/xgb_ablation.csv
  figures/xgb_ablation.png

Run:
  python scripts/11_xgb_ablation.py
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
os.makedirs(FIG, exist_ok=True)

HORIZON = 3
N_SUBSET = 30
STATIC_NUM = ["official_price_wan", "avg_rating", "positive_ratio", "negative_ratio", "review_count"]
STATIC_CAT = ["energy_type", "vehicle_class", "brand"]
LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
CAL = ["month_sin", "month_cos", "year"]
FULL_COLS = LAG_COLS + CAL + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]
NOLAG_COLS = CAL + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "WMAPE": (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
                  if np.sum(np.abs(y_true)) > 0 else np.nan),
    }


import _subset  # shared stratified ~150-series subset (replaces top-30 representative_subset)


def build_row(date, history, static_row, use_lag=True):
    h = np.array(history, dtype=float)
    row = {
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
    if use_lag:
        row.update({
            "lag_1": h[-1] if len(h) >= 1 else 0.0,
            "lag_2": h[-2] if len(h) >= 2 else 0.0,
            "lag_3": h[-3] if len(h) >= 3 else 0.0,
            "roll_mean_3": float(np.mean(h[-3:])) if len(h) >= 1 else 0.0,
            "roll_mean_6": float(np.mean(h[-6:])) if len(h) >= 1 else 0.0,
        })
    return row


def run(version, use_lag, subset, sm, static):
    cols = FULL_COLS if use_lag else NOLAG_COLS
    # FAIR ablation: both versions train on the SAME rows (those where lag
    # features exist), so the only difference is the feature set, not sample size.
    mask = sm[LAG_COLS].notna().all(axis=1)
    train = sm[mask].copy()
    Xtr, ytr = train[cols], np.log1p(train["monthly_sales"].values)
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    print(f"[Ablation:{version}] trained on {len(train)} rows")

    rows, examples = [], []
    for name in subset:
        sd = sm[sm["series_name"] == name].sort_values("date")
        if len(sd) <= HORIZON + 6:
            continue
        sr = static.loc[name]
        test_part = sd.iloc[-HORIZON:]
        history = sd.iloc[:-HORIZON]["monthly_sales"].astype(float).tolist()
        preds = []
        for _, r in test_part.iterrows():
            row = build_row(r["date"], history, sr, use_lag=use_lag)
            p = max(float(np.expm1(model.predict(pd.DataFrame([row], columns=cols))[0])), 0.0)
            preds.append(p)
            history.append(p)
        met = metrics(test_part["monthly_sales"].values, preds)
        met.update({"series_name": name, "version": version})
        rows.append(met)
        if len(examples) < 6:
            examples.append((name, sd.iloc[:-HORIZON].set_index("date")["monthly_sales"],
                             test_part.set_index("date")["monthly_sales"],
                             pd.Series(preds, index=test_part["date"])))
    return pd.DataFrame(rows), examples


def main():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    feat = pd.read_csv(FEAT)
    feat["series_name"] = feat["series_name"].astype(str)
    subset = _subset.load_subset()

    fsub = feat[["series_name"] + STATIC_NUM + STATIC_CAT].copy()
    for c in STATIC_CAT:
        fsub[c] = fsub[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(fsub[c].unique())}
        fsub[c + "_enc"] = fsub[c].map(mp)
    for c in STATIC_NUM:
        fsub[c] = pd.to_numeric(fsub[c], errors="coerce").fillna(pd.to_numeric(fsub[c], errors="coerce").median())
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

    full_df, ex_full = run("FULL", True, subset, sm, static)
    nolag_df, ex_nolag = run("NO-LAG", False, subset, sm, static)

    out = pd.concat([full_df, nolag_df], ignore_index=True)
    out.to_csv(os.path.join(PROC, "xgb_ablation.csv"), index=False)
    f = full_df["WMAPE"].mean(); n = nolag_df["WMAPE"].mean()
    print(f"\n[Ablation] FULL WMAPE={f:.1f}%   NO-LAG WMAPE={n:.1f}%   "
          f"lag contribution={((n - f) / n * 100):.1f}% of NO-LAG error removed")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(["FULL\n(with lag)", "NO-LAG\n(pure feature)"], [f, n], color=["#54A24B", "#F58518"])
    axes[0].set_ylabel("WMAPE (%)"); axes[0].set_title("XGBoost ablation: lag features")
    for i, v in enumerate([f, n]):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    name, tr, te, fc = ex_full[0]
    axes[1].plot(tr.index, tr.values, color="#4C78A8", lw=1.4, label="train")
    axes[1].plot(te.index, te.values, color="#F58518", lw=1.8, marker="o", ms=4, label="actual")
    axes[1].plot(fc.index, fc.values, color="#E45756", lw=1.8, marker="s", ms=4, label="forecast")
    axes[1].set_title(f"FULL example: {name}"); axes[1].legend(fontsize=8); axes[1].tick_params(labelsize=8)
    fig.suptitle("XGBoost ablation — contribution of recent-sales (lag) features", fontsize=12)
    fig.savefig(os.path.join(FIG, "xgb_ablation.png"), dpi=130)
    print("[Ablation] figure saved -> figures/xgb_ablation.png")
    print("[Ablation] done.")


if __name__ == "__main__":
    main()
