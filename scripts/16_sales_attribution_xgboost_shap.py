#!/usr/bin/env python3
"""Stage 4 - 16: 车系级销量归因 (XGBoost + SHAP)

用车系静态特征 + ABSA 维度情感，预测该车系的平均月销量，
并用 SHAP 量化每个情感维度对销量的贡献方向与大小。
核心对比：加入情感特征 vs 仅用静态特征，看舆情是否真的提升解释力。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import shap

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


def build_series_level():
    """聚合到车系级：情感均值 + 销量统计 + 静态特征。"""
    sent = pd.read_csv(STAGE4 / "sentiment_monthly_by_series.csv")
    sent = sent[sent["series_id"].notna()]
    # CSV round-trip 后 series_id 被推断为浮点(12345.0)，统一回整数串
    sent["series_id"] = sent["series_id"].astype(float).astype(int).astype(str)
    sent_g = sent.groupby("series_id")[ASPECTS].mean()
    sent_g["review_total"] = sent.groupby("series_id").size()

    sales = pd.read_csv(BASE_DIR / "data" / "processed" / "sales_filtered_24m.csv")
    sales["series_id"] = sales["series_id"].astype(str)
    sales_g = sales.groupby("series_id").agg(
        avg_monthly_sales=("monthly_sales", "mean"),
        sales_cv=("monthly_sales", lambda x: x.std() / x.mean() if x.mean() else np.nan),
        n_months=("monthly_sales", "count"),
    )
    static = (
        sales.groupby("series_id")
        .agg(brand=("brand", "first"), category=("category", "first"))
        .reset_index()
    )
    df = sales_g.join(sent_g, how="inner").reset_index()
    df = df.merge(static, on="series_id", how="left")
    return df


def fit_and_eval(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        n_jobs=1, random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = r2_score(yte, pred)
    mape = mean_absolute_percentage_error(yte, pred)
    print(f"[{label}] R2={r2:.3f}  MAPE={mape*100:.1f}%  (测试集 {len(yte)} 车系)")
    return model, (r2, mape)


def main():
    df = build_series_level().dropna(subset=ASPECTS).copy()
    print(f"车系级样本: {len(df)} 车系 (均有情感分)")

    # 目标：对数销量（右偏严重，log 后更稳）
    y = np.log1p(df["avg_monthly_sales"])

    # 特征 A：静态 + 情感
    X_full = pd.get_dummies(
        df[ASPECTS + ["brand", "category"]], columns=["brand", "category"], drop_first=True
    )
    # 特征 B：仅静态
    X_static = pd.get_dummies(
        df[["brand", "category"]], columns=["brand", "category"], drop_first=True
    )

    m_full, (r2_full, mape_full) = fit_and_eval(X_full, y, "with_sentiment")
    m_static, (r2_st, mape_st) = fit_and_eval(X_static, y, "without_sentiment")

    # SHAP（用含情感的模型）
    explainer = shap.TreeExplainer(m_full)
    sv = explainer.shap_values(X_full)
    # summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(sv, X_full, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "stage4_shap_summary.png", dpi=120)
    plt.close()
    # bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(sv, X_full, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIG / "stage4_shap_bar.png", dpi=120)
    plt.close()
    # 保存 SHAP 值
    pd.DataFrame(sv, columns=X_full.columns).to_csv(STAGE4 / "shap_values_series.csv", index=False)

    # 指标对比表
    pd.DataFrame(
        {
            "model": ["with_sentiment", "without_sentiment"],
            "R2": [r2_full, r2_st],
            "MAPE": [mape_full, mape_st],
        }
    ).to_csv(STAGE4 / "attribution_metrics.csv", index=False)

    # 提取情感维度的平均 |SHAP| 排序（仅看舆情贡献）
    aspect_shap = pd.Series(
        {a: np.abs(sv[:, i]).mean() for i, a in enumerate(X_full.columns) if a in ASPECTS},
        name="mean_abs_shap",
    ).sort_values(ascending=False).to_frame()
    aspect_shap.to_csv(STAGE4 / "aspect_shap_ranking.csv")

    print("\n情感维度 SHAP 贡献排名 (mean |SHAP|, 越大越影响销量):")
    for a, v in aspect_shap["mean_abs_shap"].items():
        print(f"  {a:20s}: {float(v):.4f}")
    print(f"\n情感特征带来的 R2 提升: {r2_full - r2_st:+.3f}")
    print(f"输出图: {FIG / 'stage4_shap_summary.png'}, {FIG / 'stage4_shap_bar.png'}")


if __name__ == "__main__":
    main()
