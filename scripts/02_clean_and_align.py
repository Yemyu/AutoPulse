#!/usr/bin/env python3
"""
舆情数据清洗与三表对齐
读取 sentiment_reviews.csv (明细) + sales.csv (销量) + vehicles.csv (配置),
产出分析就绪表 data/sentiment/analysis_input.csv (一行一车系):

  舆情指标 (来自 reviews 聚合)  +  销量标签 (来自 sales)  +  车型特征 (来自 vehicles)

供后续影响因子回归 / NLP 情感分析直接使用。

运行:
  python scripts/02_clean_and_align.py
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SENT_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment")

REVIEWS = os.path.join(SENT_DIR, "sentiment_reviews.csv")
SALES = os.path.join(RAW_DIR, "sales.csv")
VEH = os.path.join(RAW_DIR, "vehicles.csv")
OUT = os.path.join(SENT_DIR, "analysis_input.csv")


def build_summary(df):
    """车系级情感聚合 (内存中, 不写文件, 避免与采集进程冲突)"""
    summary = df.groupby(['series_id', 'series_name']).agg(
        review_count=('review_id', 'count'),
        avg_rating=('rating_overall', 'mean'),
        median_rating=('rating_overall', 'median'),
        avg_content_len=('content_len', 'mean'),
        positive_cnt=('rating_overall', lambda x: (x >= 4.5).sum()),
        neutral_cnt=('rating_overall', lambda x: ((x >= 3.5) & (x < 3.5)).sum()),
        negative_cnt=('rating_overall', lambda x: (x < 3.5).sum()),
    ).reset_index()
    total = summary['positive_cnt'] + summary['neutral_cnt'] + summary['negative_cnt']
    summary['positive_ratio'] = (summary['positive_cnt'] / total).round(3)
    summary['negative_ratio'] = (summary['negative_cnt'] / total).round(3)
    dim_cols = [c for c in df.columns if c.startswith('rating_') and c != 'rating_overall']
    dim_agg = df.groupby(['series_id', 'series_name'])[dim_cols].mean().reset_index()
    dim_agg.columns = [f'avg_{c}' if c not in ('series_id', 'series_name') else c
                       for c in dim_agg.columns]
    return summary.merge(dim_agg, on=['series_id', 'series_name'], how='left')


def main():
    # 1) 清洗评论明细
    rev = pd.read_csv(REVIEWS, encoding='utf-8-sig', low_memory=False)
    n0 = len(rev)
    rev = rev.drop_duplicates(subset='review_id').dropna(subset=['review_id'])
    n1 = len(rev)
    print(f"[清洗] 评论: {n0:,} -> 去重后 {n1:,} (移除 {n0 - n1} 条重复)")

    # 统一 series_id 为 str 便于跨表关联
    rev['series_id'] = rev['series_id'].astype(str)

    # 2) 车系级情感聚合
    summary = build_summary(rev)
    summary['series_id'] = summary['series_id'].astype(str)
    print(f"[聚合] 车系级情感指标: {len(summary)} 个车系")

    # 3) 销量按车系聚合 (回归标签 Y)
    sales = pd.read_csv(SALES, encoding='utf-8-sig', low_memory=False)
    sales['series_id'] = sales['series_id'].astype(str)
    sales_agg = sales.groupby('series_id').agg(
        total_sales=('monthly_sales', 'sum'),
        avg_monthly_sales=('monthly_sales', 'mean'),
        n_months=('monthly_sales', 'count'),
        brand=('brand', 'first'),
        category=('category', 'first'),
    ).reset_index()
    sales_agg['log_avg_monthly_sales'] = np.log1p(sales_agg['avg_monthly_sales'].clip(lower=0))
    print(f"[销量] 聚合: {len(sales_agg)} 个车系有销量记录")

    # 4) 车型配置 (每个车系取一条代表性记录)
    veh = pd.read_csv(VEH, encoding='utf-8-sig', low_memory=False)
    veh['series_id'] = veh['series_id'].astype(str)
    key_cols = ['official_price_wan', 'vehicle_class', 'energy_type',
                'manufacturer', 'brand_name']
    key_cols = [c for c in key_cols if c in veh.columns]
    veh1 = veh.sort_values('series_id').drop_duplicates(subset='series_id')[['series_id'] + key_cols]
    print(f"[配置] 车系级特征: {len(veh1)} 个车系")

    # 5) 三表对齐
    df = summary.merge(sales_agg, on='series_id', how='left')
    df = df.merge(veh1, on='series_id', how='left')

    matched_sales = int(df['total_sales'].notna().sum())
    matched_veh = int(df['official_price_wan'].notna().sum()) if 'official_price_wan' in df else 0
    print(f"[对齐] 舆情 {len(df)} 系 -> 命中销量 {matched_sales} | 命中配置 {matched_veh}")

    # 6) 输出
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f"[输出] {OUT} ({len(df)} 行 x {df.shape[1]} 列)")

    # 7) 对齐质量速览
    print("\n=== 对齐质量 ===")
    print(f"  有舆情+销量: {matched_sales} 系")
    print(f"  有舆情无销量: {len(df) - matched_sales} 系 (回归时需过滤)")
    if matched_sales:
        print(f"  销量中位: {df['avg_monthly_sales'].median():.0f} 辆/月")


if __name__ == '__main__':
    main()
