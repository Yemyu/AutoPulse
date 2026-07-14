#!/usr/bin/env python3
"""Stage 4 - 17: 时序因果 (Granger 因果 + VAR)

回答"舆情是否领先于销量变化"——检验各维度情感是否 Granger 引起销量变动。
两层：
  (1) 全市场级：按月聚合，检验舆情→销量（lag 1/2/3）
  (2) 品牌级：逐品牌检验，统计各维度在多少品牌上显著
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
import _font_setup  # noqa

ASPECTS = [
    "appearance", "interior", "space", "power", "control",
    "comfort", "fuel_consumption", "configuration", "intelligence", "value",
]
STAGE4 = BASE_DIR / "data" / "processed" / "stage4"
FIG = BASE_DIR / "figures"
FIG.mkdir(exist_ok=True)


def granger_pvals(y, x, maxlag=3):
    """x 是否 Granger 引起 y？返回各 lag 的 p 值 (ssr_ftest)。"""
    df = pd.DataFrame({"y": y.values, "x": x.values}).dropna()
    if len(df) <= maxlag + 5:
        return None
    try:
        # statsmodels 0.14 返回键为 np.int64，p 值在 res[lag][0]['ssr_ftest'][1]
        res = grangercausalitytests(df[["y", "x"]], maxlag=maxlag, verbose=False)
        return {int(lag): float(val[0]["ssr_ftest"][1]) for lag, val in res.items()}
    except Exception:
        return None


def main():
    brand = pd.read_csv(STAGE4 / "sentiment_sales_monthly_brand.csv")
    brand["period"] = pd.to_datetime(brand["period"])
    brand = brand.sort_values(["brand", "period"])

# 全市场级：按月聚合
    mkt = brand.groupby("period").mean(numeric_only=True)
    mkt = mkt.diff().dropna()  # 一阶差分，去趋势/平稳化
    print(f"全市场时序: {len(mkt)} 个月 (差分后)")

    market_rows = []
    for a in ASPECTS:
        p = granger_pvals(mkt["monthly_sales"], mkt[a], maxlag=3)
        if p:
            market_rows.append({
                "aspect": a,
                "p_lag1": p.get(1), "p_lag2": p.get(2), "p_lag3": p.get(3),
                "min_p": min(p.values()),
                "sig_any": any(v < 0.05 for v in p.values()),
            })
    market_df = pd.DataFrame(market_rows).sort_values("min_p")
    market_df.to_csv(STAGE4 / "granger_market.csv", index=False)
    print("\n全市场 Granger (情感→销量, 差分序列):")
    for _, r in market_df.iterrows():
        flag = " *显著*" if r["sig_any"] else ""
        print(f"  {r['aspect']:20s} min_p={r['min_p']:.4f}{flag}")

# 品牌级：逐品牌
    sig_counts = {a: 0 for a in ASPECTS}
    tot_counts = {a: 0 for a in ASPECTS}
    for bname, g in brand.groupby("brand"):
        if len(g) < 15:
            continue
        g = g.set_index("period").select_dtypes(include=[np.number]).diff().dropna()
        for a in ASPECTS:
            p = granger_pvals(g["monthly_sales"], g[a], maxlag=2)
            if p:
                tot_counts[a] += 1
                if any(v < 0.05 for v in p.values()):
                    sig_counts[a] += 1
    brand_summary = pd.DataFrame({
        "aspect": ASPECTS,
        "n_brands_tested": [tot_counts[a] for a in ASPECTS],
        "n_sig": [sig_counts[a] for a in ASPECTS],
    })
    brand_summary["sig_rate"] = brand_summary["n_sig"] / brand_summary["n_brands_tested"].replace(0, np.nan)
    brand_summary = brand_summary.sort_values("sig_rate", ascending=False)
    brand_summary.to_csv(STAGE4 / "granger_brand_summary.csv", index=False)
    print("\n品牌级 Granger 显著比例 (n_brands_tested>=15):")
    for _, r in brand_summary.iterrows():
        rate = f"{r['sig_rate']*100:.0f}%" if pd.notna(r["sig_rate"]) else "NA"
        print(f"  {r['aspect']:20s} 显著 {int(r['n_sig'])}/{int(r['n_brands_tested'])} = {rate}")

# 可视化
    # 图1：全市场情感均值 vs 平均销量（差分前原始水平）
    raw = brand.groupby("period").mean(numeric_only=True)
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(raw.index, raw["monthly_sales"], color="tab:blue", label="平均月销量")
    ax1.set_ylabel("平均月销量", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(raw.index, raw["comfort"], color="tab:red", alpha=0.7, label="舒适性情感")
    ax2.plot(raw.index, raw["value"], color="tab:green", alpha=0.7, label="性价比情感")
    ax2.set_ylabel("情感分 (均值)")
    ax1.set_title("全市场：月度销量 vs 舆情情感 (2022-2026)")
    fig.tight_layout()
    fig.savefig(FIG / "stage4_market_timeseries.png", dpi=120)
    plt.close()

    # 图2：全市场 Granger 最小 p 值条形图
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["tab:red" if p < 0.05 else "tab:gray" for p in market_df["min_p"]]
    ax.barh(market_df["aspect"], -np.log10(market_df["min_p"] + 1e-12), color=colors)
    ax.axvline(-np.log10(0.05), color="black", ls="--", label="p=0.05 阈值")
    ax.set_xlabel("-log10(p)")
    ax.set_title("全市场 Granger: 舆情→销量的因果证据强度 (越长越显著)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage4_granger_market.png", dpi=120)
    plt.close()

    # 图3：品牌级显著比例
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bs = brand_summary.dropna(subset=["sig_rate"]).sort_values("sig_rate")
    ax.barh(bs["aspect"], bs["sig_rate"] * 100, color="tab:purple")
    ax.set_xlabel("显著 Granger 的品牌占比 (%)")
    ax.set_title("品牌级：舆情领先销量的显著比例")
    fig.tight_layout()
    fig.savefig(FIG / "stage4_granger_brand.png", dpi=120)
    plt.close()

    print(f"\n输出图: stage4_market_timeseries.png, stage4_granger_market.png, stage4_granger_brand.png")


if __name__ == "__main__":
    main()
