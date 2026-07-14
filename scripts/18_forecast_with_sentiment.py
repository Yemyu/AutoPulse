#!/usr/bin/env python3
"""
Stage 5 - Step 1: sentiment -> sales forecasting fusion
==========================================================
把阶段四的车系级月度情感序列作为外生特征，融入阶段三的销量预测模型，
对比"无情感"和"有情感"两版本的预测精度。

模型：
  - XGBoost（global，主）：阶段三同款特征 + 情感滞后特征
  - Prophet（per-series，辅）：加综合情感滞后作为 regressor

输出：
  data/processed/stage5/forecast_comparison.csv
  data/processed/stage5/feature_importance.csv
  data/processed/stage5/per_series_metrics.csv
  figures/stage5_forecast_comparison.png
  figures/stage5_sentiment_feature_importance.png

Run:
  python scripts/18_forecast_with_sentiment.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 避免 torch/XGBoost OpenMP 冲突导致段错误
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
SENTIMENT = os.path.join(BASE, "data", "processed", "stage4", "sentiment_monthly_by_series.csv")
SUBSET_CSV = os.path.join(BASE, "data", "processed", "subset_150.csv")
PROC = os.path.join(BASE, "data", "processed", "stage5")
FIG = os.path.join(BASE, "figures")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

HORIZON = 3
MAX_EXCLUDE = 12
ASPECTS = ["appearance", "interior", "space", "power", "control",
           "comfort", "fuel_consumption", "configuration", "intelligence", "value"]
TOP3 = ["comfort", "value", "intelligence"]
STATIC_NUM = ["official_price_wan", "avg_rating", "positive_ratio", "negative_ratio", "review_count"]
STATIC_CAT = ["energy_type", "vehicle_class", "brand"]
LAG_COLS = ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6"]
CAL = ["month_sin", "month_cos", "year"]


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else np.nan
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else np.nan
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) if np.var(y_true) > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WMAPE": wmape, "R2": r2}


def load_data():
    sales = pd.read_csv(SALES)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["series_name"] = sales["series_name"].astype(str)
    sales["series_id"] = sales["series_id"].astype(str)
    sales["period"] = sales["date"].dt.strftime("%Y-%m")

    feat = pd.read_csv(FEAT)
    feat["series_name"] = feat["series_name"].astype(str)
    feat["series_id"] = feat["series_id"].astype(int).astype(str)

    sent = pd.read_csv(SENTIMENT)
    sent["series_id"] = sent["series_id"].astype(int).astype(str)
    sent["period"] = sent["period"].astype(str).str[:7]
    sent["overall"] = sent[ASPECTS].mean(axis=1)

    sent = sent.sort_values(["series_id", "period"]).reset_index(drop=True)
    for a in ASPECTS + ["overall"]:
        for lag in [1, 2, 3]:
            sent[f"{a}_lag{lag}"] = sent.groupby("series_id")[a].shift(lag)
    sent["overall_change"] = sent["overall_lag1"] - sent["overall_lag2"]

    sm = sales.merge(sent, on=["series_id", "period"], how="left")
    sentiment_cols = [f"{a}_lag{lag}" for a in ASPECTS + ["overall"] for lag in [1, 2, 3]] + ["overall_change"]
    for c in sentiment_cols:
        sm[c] = sm[c].fillna(0)

    # build static features from analysis_input; drop any overlapping columns from sales first
    for c in STATIC_CAT + STATIC_NUM:
        if c in sm.columns:
            sm = sm.drop(columns=[c])
    static = feat.drop_duplicates("series_id").set_index("series_id")[["series_name"] + STATIC_NUM + STATIC_CAT]
    for c in STATIC_CAT:
        static[c] = static[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(sorted(static[c].unique()))}
        static[c + "_enc"] = static[c].map(mp)
    for c in STATIC_NUM:
        static[c] = pd.to_numeric(static[c], errors="coerce")
        static[c] = static[c].fillna(static[c].median())
    sm = sm.merge(static.reset_index()[["series_id"] + STATIC_NUM + STATIC_CAT], on="series_id", how="left")

    g = sm.groupby("series_id")["monthly_sales"]
    sm["lag_1"] = g.shift(1)
    sm["lag_2"] = g.shift(2)
    sm["lag_3"] = g.shift(3)
    sm["roll_mean_3"] = g.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    sm["roll_mean_6"] = g.shift(1).rolling(6).mean().reset_index(level=0, drop=True)
    sm["month_of_year"] = sm["date"].dt.month
    sm["year"] = sm["date"].dt.year
    sm["month_sin"] = np.sin(2 * np.pi * sm["month_of_year"] / 12)
    sm["month_cos"] = np.cos(2 * np.pi * sm["month_of_year"] / 12)
    sm["rev_rank"] = sm.groupby("series_id").cumcount(ascending=False)

    for c in STATIC_CAT:
        sm[c] = sm[c].astype(str).fillna("NA")
        mp = {v: i for i, v in enumerate(sorted(sm[c].unique()))}
        sm[c + "_enc"] = sm[c].map(mp)
    for c in STATIC_NUM:
        sm[c] = pd.to_numeric(sm[c], errors="coerce")
        sm[c] = sm[c].fillna(sm[c].median())

    sm = sm.sort_values(["series_id", "date"]).reset_index(drop=True)
    return sm, static


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


def xgb_forecast(sm, subset, feature_cols, version, static):
    base_cols = LAG_COLS + CAL + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]
    all_cols = feature_cols
    train = sm[(sm["rev_rank"] >= MAX_EXCLUDE) & sm[all_cols].notna().all(axis=1)].copy()
    if len(train) < 50:
        raise ValueError(f"Too few training rows: {len(train)}")
    Xtr, ytr = train[all_cols], np.log1p(train["monthly_sales"].values)
    model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
    model.fit(Xtr, ytr)
    print(f"[XGB-{version}] trained on {len(train)} rows, {len(all_cols)} features")

    rows, preds_rows = [], []
    for name in subset:
        sd = sm[sm["series_name"] == name].sort_values("date")
        if len(sd) <= HORIZON + 6:
            continue
        sr = static.loc[sd["series_id"].iloc[0]]
        test_part = sd.iloc[-HORIZON:]
        history = sd.iloc[:-HORIZON]["monthly_sales"].astype(float).tolist()
        preds = []
        for _, r in test_part.iterrows():
            row = build_row(r["date"], history, sr)
            # fill sentiment columns for this row from the merged dataframe
            for c in all_cols:
                if c not in row:
                    row[c] = float(r[c]) if c in r.index and pd.notna(r[c]) else 0.0
            p = max(float(np.expm1(model.predict(pd.DataFrame([row], columns=all_cols))[0])), 0.0)
            preds.append(p)
            preds_rows.append({"series_name": name, "date": r["date"],
                               "actual": float(r["monthly_sales"]), "pred": p, "version": version})
            history.append(p)
        met = metrics(test_part["monthly_sales"].values, preds)
        met.update({"series_name": name, "version": version, "model": "XGBoost"})
        rows.append(met)
    return pd.DataFrame(rows), pd.DataFrame(preds_rows), model


def prophet_forecast(sm, subset, use_sentiment, static):
    try:
        from prophet import Prophet
    except Exception as e:
        print(f"[Prophet] skip: {e}")
        return pd.DataFrame()

    rows, preds_rows = [], []
    version = f"Prophet{'+sent' if use_sentiment else '-baseline'}"
    for name in subset:
        sd = sm[sm["series_name"] == name].sort_values("date")
        if len(sd) <= HORIZON + 6:
            continue
        train = sd.iloc[:-HORIZON]
        test = sd.iloc[-HORIZON:]
        df = pd.DataFrame({
            "ds": train["date"].values,
            "y": train["monthly_sales"].values.astype(float),
        })
        if use_sentiment:
            df["overall_lag1"] = train["overall_lag1"].values

        m = Prophet(weekly_seasonality=False, daily_seasonality=False,
                    yearly_seasonality=True, seasonality_mode="additive")
        if use_sentiment:
            m.add_regressor("overall_lag1")
        try:
            m.fit(df)
        except Exception:
            continue

        future = m.make_future_dataframe(periods=HORIZON, freq="MS")
        if use_sentiment:
            # future lag1 sentiment: carry forward the last known value (naive but leakage-free)
            last_val = float(train["overall_lag1"].iloc[-1])
            future["overall_lag1"] = last_val
        fc = m.predict(future).iloc[-HORIZON:]["yhat"].clip(lower=0).values
        for j, d in enumerate(test.index):
            preds_rows.append({"series_name": name, "date": pd.Timestamp(d),
                               "actual": float(test["monthly_sales"].values[j]), "pred": float(fc[j]), "version": version})
        met = metrics(test["monthly_sales"].values, fc)
        met.update({"series_name": name, "version": version, "model": "Prophet"})
        rows.append(met)
    return pd.DataFrame(rows), pd.DataFrame(preds_rows)


def aggregate(df, version, model, preds):
    ok = df[df["version"] == version]
    if ok.empty:
        return None
    pv = preds[preds["version"] == version]
    a = pv["actual"].values.astype(float)
    p = pv["pred"].values.astype(float)
    wmape_vol = np.sum(np.abs(a - p)) / np.sum(np.abs(a)) * 100 if np.sum(np.abs(a)) > 0 else np.nan
    return {
        "model": model,
        "version": version,
        "WMAPE_vol": wmape_vol,
        "WMAPE_mean": ok["WMAPE"].mean(),
        "WMAPE_med": ok["WMAPE"].median(),
        "MAPE_mean": ok["MAPE"].mean(),
        "MAE_mean": ok["MAE"].mean(),
        "RMSE_mean": ok["RMSE"].mean(),
        "R2_mean": ok["R2"].mean(),
        "n_series": len(ok),
    }


def plot_comparison(comp, fig_path):
    comp = comp.sort_values("WMAPE_vol")
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = ["#54A24B" if i == 0 else "#4C78A8" for i in range(len(comp))]
    ax.barh(comp["version"], comp["WMAPE_vol"], color=colors)
    ax.set_xlabel("volume-weighted WMAPE (%)  lower = better")
    ax.set_title("Stage 5 — sentiment fusion forecast comparison (150-series stratified subset)")
    for i, v in enumerate(comp["WMAPE_vol"].values):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=9)
    fig.savefig(fig_path, dpi=130)
    print(f"[Plot] {fig_path}")


def plot_feature_importance(model, feature_cols, fig_path):
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(imp["feature"], imp["importance"], color="#4C78A8")
    ax.set_xlabel("gain importance")
    ax.set_title("Stage 5 — XGBoost feature importance (with sentiment)")
    fig.savefig(fig_path, dpi=130)
    print(f"[Plot] {fig_path}")


def main():
    sm, static = load_data()
    subset = pd.read_csv(SUBSET_CSV)["series_name"].astype(str).tolist()
    subset = [s for s in subset if s in sm["series_name"].values]
    print(f"[Stage 5] evaluating on {len(subset)} series from stratified subset")

    base_cols = LAG_COLS + CAL + STATIC_NUM + [c + "_enc" for c in STATIC_CAT]
    top3_sent_cols = [f"{a}_lag{lag}" for a in TOP3 for lag in [1, 2, 3]] + ["overall_lag1", "overall_change"]
    full_sent_cols = [f"{a}_lag{lag}" for a in ASPECTS for lag in [1, 2, 3]] + \
                     [f"overall_lag{lag}" for lag in [1, 2, 3]] + ["overall_change"]

    xgb_base, pred_base, _ = xgb_forecast(sm, subset, base_cols, "XGBoost-baseline", static)
    xgb_top3, pred_top3, model_top3 = xgb_forecast(sm, subset, base_cols + top3_sent_cols, "XGBoost+Top3sent", static)
    xgb_full, pred_full, model_full = xgb_forecast(sm, subset, base_cols + full_sent_cols, "XGBoost+Fullsent", static)

    prophet_base, pred_prophet_base = prophet_forecast(sm, subset, use_sentiment=False, static=static)
    prophet_sent, pred_prophet_sent = prophet_forecast(sm, subset, use_sentiment=True, static=static)

    per_series = pd.concat([xgb_base, xgb_top3, xgb_full, prophet_base, prophet_sent], ignore_index=True)
    per_series.to_csv(os.path.join(PROC, "per_series_metrics.csv"), index=False)
    all_preds = pd.concat([pred_base, pred_top3, pred_full, pred_prophet_base, pred_prophet_sent], ignore_index=True)
    all_preds.to_csv(os.path.join(PROC, "forecast_preds.csv"), index=False)

    rows = []
    for df, version, model, preds in [
        (xgb_base, "XGBoost-baseline", "XGBoost", pred_base),
        (xgb_top3, "XGBoost+Top3sent", "XGBoost", pred_top3),
        (xgb_full, "XGBoost+Fullsent", "XGBoost", pred_full),
        (prophet_base, "Prophet-baseline", "Prophet", pred_prophet_base),
        (prophet_sent, "Prophet+sent", "Prophet", pred_prophet_sent),
    ]:
        r = aggregate(df, version, model, preds)
        if r:
            rows.append(r)
    comp = pd.DataFrame(rows).sort_values("WMAPE_vol")
    comp.to_csv(os.path.join(PROC, "forecast_comparison.csv"), index=False)
    print("\n===== Stage 5 forecast comparison =====")
    print(comp.round(3).to_string(index=False))

    # feature importance for the top3 model (best balance of interpretability and performance)
    fi = pd.DataFrame({"feature": base_cols + top3_sent_cols,
                       "importance": model_top3.feature_importances_})
    fi = fi.sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(PROC, "feature_importance.csv"), index=False)

    plot_comparison(comp, os.path.join(FIG, "stage5_forecast_comparison.png"))
    plot_feature_importance(model_top3, base_cols + top3_sent_cols,
                            os.path.join(FIG, "stage5_sentiment_feature_importance.png"))
    print("[Stage 5] done.")


if __name__ == "__main__":
    main()
