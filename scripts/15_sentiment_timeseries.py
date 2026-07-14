#!/usr/bin/env python3
"""Stage 4 - 15: 情感时序聚合 + 对齐销量

把 ABSA 的 10 个维度情感按月 / 车系聚合，与月度销量对齐，
输出后续归因建模（XGBoost/SHAP）和时序因果（Granger/VAR）用的宽表。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import _font_setup  # noqa: 统一中文字体（保持各脚本一致）

ASPECTS = [
    "appearance", "interior", "space", "power", "control",
    "comfort", "fuel_consumption", "configuration", "intelligence", "value",
]
MIN_REVIEWS = 3  # 某车系某月评论 < 此数，情感分视为不稳定，置 NaN

RAW = BASE_DIR / "data" / "sentiment" / "absa" / "absa_results.csv"
SALES = BASE_DIR / "data" / "processed" / "sales_filtered_24m.csv"
OUTDIR = BASE_DIR / "data" / "processed" / "stage4"
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1) 读 ABSA 结果
    absa = pd.read_csv(RAW)
    # CSV 中 series_id 为浮点(如 12345.0)，统一转成整数串 "12345"，
    # 否则与销量表的整数串 join 不上
    absa["series_id"] = absa["series_id"].astype(float).astype(int).astype(str)
    absa = absa[absa["success"] == True]  # noqa: 仅保留成功样本
    absa["pt"] = pd.to_datetime(absa["publish_time"], errors="coerce")
    absa["period"] = absa["pt"].dt.to_period("M").astype(str)
    absa = absa[absa["period"].notna()]

    # 2) 按 车系 × 月份 聚合维度情感均值 + 评论数
    g = absa.groupby(["series_id", "period"])
    mean_df = g[ASPECTS].mean()
    cnt = g.size().rename("review_count")
    sent = mean_df.join(cnt).reset_index()

    # 评论不足则情感分置 NaN（避免少量评论带来的剧烈波动）
    sent.loc[sent["review_count"] < MIN_REVIEWS, ASPECTS] = np.nan

    # 3) 读销量并对齐月份
    sales = pd.read_csv(SALES)
    # sales 的 series_id 含非数字(如 sg22395)，保持原字符串格式；
    # 数字部分(如 "12345")能与 absa 转出的整数串匹配
    sales["series_id"] = sales["series_id"].astype(str)
    sales["period"] = pd.to_datetime(sales["date"]).dt.to_period("M").astype(str)

    # 4) 左连接：保留全部销量记录，附上情感分
    merged = pd.merge(sales, sent, on=["series_id", "period"], how="left")

    merged.to_csv(OUTDIR / "sentiment_sales_monthly.csv", index=False)
    sent.to_csv(OUTDIR / "sentiment_monthly_by_series.csv", index=False)

    n_with_sent = int(merged[ASPECTS[0]].notna().sum())
    print(f"[1] 情感×月份聚合: {len(sent)} 行 ({sent['series_id'].nunique()} 车系)")
    print(f"[2] 对齐销量宽表: {len(merged)} 行, 其中含情感分的: {n_with_sent}")
    print(f"    输出 -> {OUTDIR / 'sentiment_sales_monthly.csv'}")
    print(f"    输出 -> {OUTDIR / 'sentiment_monthly_by_series.csv'}")

    # 5) 品牌级聚合：时序更密，适合 Granger/VAR 因果分析
    brand = (
        merged.dropna(subset=ASPECTS)
        .groupby(["brand", "period"])[ASPECTS + ["monthly_sales"]]
        .mean()
        .reset_index()
    )
    brand.to_csv(OUTDIR / "sentiment_sales_monthly_brand.csv", index=False)
    print(f"[3] 品牌级宽表: {len(brand)} 行 ({brand['brand'].nunique()} 品牌) "
          f"-> {OUTDIR / 'sentiment_sales_monthly_brand.csv'}")


if __name__ == "__main__":
    main()
