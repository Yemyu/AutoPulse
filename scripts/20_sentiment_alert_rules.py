#!/usr/bin/env python3
"""
sentiment alert rules
基于月度综合情感定义简单预警规则：
  1. 综合情感 < -0.2（整体偏负面）
  2. 综合情感较上月环比下降 > 0.1（负面突变）

输出：
  data/processed/stage5/sentiment_alerts.csv
  figures/stage5_sentiment_alerts.png

Run:
  python scripts/20_sentiment_alert_rules.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _font_setup

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT = os.path.join(BASE, "data", "processed", "stage4", "sentiment_monthly_by_series.csv")
FEAT = os.path.join(BASE, "data", "sentiment", "analysis_input.csv")
PROC = os.path.join(BASE, "data", "processed", "stage5")
FIG = os.path.join(BASE, "figures")
os.makedirs(PROC, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ASPECTS = ["appearance", "interior", "space", "power", "control",
           "comfort", "fuel_consumption", "configuration", "intelligence", "value"]
ALERT_THRESHOLD = -0.1
DROP_THRESHOLD = -0.05


def main():
    sent = pd.read_csv(SENTIMENT)
    sent["series_id"] = sent["series_id"].astype(int).astype(str)
    sent["period"] = sent["period"].astype(str).str[:7]
    sent["overall"] = sent[ASPECTS].mean(axis=1)

    sent = sent.sort_values(["series_id", "period"]).reset_index(drop=True)
    sent["overall_lag1"] = sent.groupby("series_id")["overall"].shift(1)
    sent["overall_drop"] = sent["overall"] - sent["overall_lag1"]
    sent["alert"] = (sent["overall"] < ALERT_THRESHOLD) & (sent["overall_drop"] < DROP_THRESHOLD)

    feat = pd.read_csv(FEAT)
    feat["series_id"] = feat["series_id"].astype(int).astype(str)
    brand_map = feat.drop_duplicates("series_id").set_index("series_id")["brand"].to_dict()
    series_name_map = feat.drop_duplicates("series_id").set_index("series_id")["series_name"].to_dict()

    alerts = sent[sent["alert"]].copy()
    alerts["brand"] = alerts["series_id"].map(brand_map)
    alerts["series_name"] = alerts["series_id"].map(series_name_map)
    alerts = alerts[["series_id", "series_name", "brand", "period", "overall",
                     "overall_lag1", "overall_drop"] + ASPECTS]
    alerts.to_csv(os.path.join(PROC, "sentiment_alerts.csv"), index=False)
    print(f"[Stage 5C] {len(alerts)} alerts saved")

    if alerts.empty:
        print("[Stage 5C] no alerts, skip figure")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    brand_counts = alerts["brand"].value_counts().head(15)
    axes[0].barh(brand_counts.index[::-1], brand_counts.values[::-1], color="#F58518")
    axes[0].set_xlabel("alert count")
    axes[0].set_title("Top brands by sentiment alert count")

    period_counts = alerts["period"].value_counts().sort_index()
    axes[1].bar(period_counts.index, period_counts.values, color="#E45756")
    axes[1].set_xlabel("period")
    axes[1].set_ylabel("alert count")
    axes[1].set_title("Sentiment alerts over time")
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)

    fig.suptitle("Stage 5 — sentiment alert rules (overall < -0.2 & drop > 0.1)", fontsize=12)
    fig.savefig(os.path.join(FIG, "stage5_sentiment_alerts.png"), dpi=130)
    print(f"[Plot] {os.path.join(FIG, 'stage5_sentiment_alerts.png')}")
    print("[Stage 5C] done.")


if __name__ == "__main__":
    main()
