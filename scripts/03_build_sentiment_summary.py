#!/usr/bin/env python3
"""
舆情汇总 + 数据质量报告生成器
================================
读取 sentiment_reviews.csv, 重新生成 sentiment_summary.csv (车系级聚合),
并输出一份完整的数据覆盖与质量报告 (JSON + Markdown)。

用途:
  - 爬虫中途停止后, 补生成最终的汇总表
  - 量化数据覆盖范围 (品牌/车系/评论数) 与质量 (重复/缺失/极性)

运行:
  python scripts/03_build_sentiment_summary.py
"""
import os
import json
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SENT_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

REVIEWS = os.path.join(SENT_DIR, "sentiment_reviews.csv")
SUMMARY = os.path.join(SENT_DIR, "sentiment_summary.csv")
VEH = os.path.join(RAW_DIR, "vehicles.csv")
SALES = os.path.join(RAW_DIR, "sales.csv")
# 质量报告写入 docs/ (作为数据 README 的可复现补充), 不再散落在 sentiment/
REPORT_JSON = os.path.join(DOCS_DIR, "data_quality_report.json")
REPORT_MD = os.path.join(DOCS_DIR, "data_quality_report.md")


def load_reviews():
    df = pd.read_csv(REVIEWS, encoding='utf-8-sig', low_memory=False)
    return df


def build_summary(df):
    """车系级聚合 (复刻爬虫 generate_summary 逻辑)"""
    summary = df.groupby(['series_id', 'series_name']).agg(
        review_count=('review_id', 'count'),
        avg_rating=('rating_overall', 'mean'),
        median_rating=('rating_overall', 'median'),
        min_rating=('rating_overall', 'min'),
        max_rating=('rating_overall', 'max'),
        avg_content_len=('content_len', 'mean'),
        total_digg=('digg_count', 'sum'),
        total_comment=('comment_count', 'sum'),
        earliest_review=('publish_time', 'min'),
        latest_review=('publish_time', 'max'),
        positive_cnt=('rating_overall', lambda x: (x >= 4.5).sum()),
        neutral_cnt=('rating_overall', lambda x: ((x >= 3.5) & (x < 4.5)).sum()),
        negative_cnt=('rating_overall', lambda x: (x < 3.5).sum()),
    ).reset_index()

    total_check = summary['positive_cnt'] + summary['neutral_cnt'] + summary['negative_cnt']
    summary['positive_ratio'] = (summary['positive_cnt'] / total_check).round(3)
    summary['neutral_ratio'] = (summary['neutral_cnt'] / total_check).round(3)
    summary['negative_ratio'] = (summary['negative_cnt'] / total_check).round(3)

    dim_cols = [c for c in df.columns if c.startswith('rating_') and c != 'rating_overall']
    dim_agg = df.groupby(['series_id', 'series_name'])[dim_cols].mean().reset_index()
    dim_agg.columns = [f'avg_{c}' if c not in ('series_id', 'series_name') else c
                       for c in dim_agg.columns]
    summary = summary.merge(dim_agg, on=['series_id', 'series_name'], how='left')
    summary.to_csv(SUMMARY, index=False, encoding='utf-8-sig')
    return summary


def attach_brand(df):
    """用 series_id 关联销量表/主表, 补出品牌维度 (reviews 本身无 brand 列)"""
    brand_map = None
    if os.path.exists(SALES):
        s = pd.read_csv(SALES, encoding='utf-8-sig', low_memory=False)
        if 'brand' in s.columns and 'series_id' in s.columns:
            brand_map = s[['series_id', 'brand']].drop_duplicates()
            brand_map['series_id'] = brand_map['series_id'].astype(str)
    if brand_map is None and os.path.exists(VEH):
        v = pd.read_csv(VEH, encoding='utf-8-sig', low_memory=False)
        if 'brand_name' in v.columns and 'series_id' in v.columns:
            brand_map = v[['series_id', 'brand_name']].rename(columns={'brand_name': 'brand'})
            brand_map['series_id'] = brand_map['series_id'].astype(str)
    if brand_map is not None:
        df['series_id_str'] = df['series_id'].astype(str)
        df = df.merge(brand_map, left_on='series_id_str', right_on='series_id', how='left')
        df = df.drop(columns=['series_id_str', 'series_id_y'], errors='ignore')
        if 'series_id_x' in df.columns:
            df = df.rename(columns={'series_id_x': 'series_id'})
    return df


def main():
    df = load_reviews()
    print(f"[1/4] 读取评论: {len(df):,} 条, 覆盖车系 {df['series_id'].nunique()} 个")

    # 2) 汇总表
    summary = build_summary(df)
    print(f"[2/4] 生成汇总表: {len(summary)} 个车系 -> {os.path.basename(SUMMARY)}")

    # 3) 品牌维度
    dfb = attach_brand(df)
    has_brand = 'brand' in dfb.columns
    if has_brand:
        dfb['brand'] = dfb['brand'].fillna('未知')
        brand_stats = dfb.groupby('brand').agg(
            series=('series_id', 'nunique'),
            reviews=('review_id', 'count'),
        ).sort_values('reviews', ascending=False)
        brand_stats['avg_rating'] = dfb.groupby('brand')['rating_overall'].mean().round(2)
    else:
        brand_stats = None

    # 4) 质量指标
    dup_rev = int(df['review_id'].duplicated().sum())
    missing_rating = int(df['rating_overall'].isna().sum())
    missing_content = int((df['content'].astype(str).str.len() == 0).sum())
    empty_rating_series = int((summary['avg_rating'].isna()).sum())

    # 极性 (基于有评分的子集)
    rated = df[df['rating_overall'].notna()]
    pos = int((rated['rating_overall'] >= 4.5).sum())
    neu = int(((rated['rating_overall'] >= 3.5) & (rated['rating_overall'] < 4.5)).sum())
    neg = int((rated['rating_overall'] < 3.5).sum())
    rated_total = len(rated)

    report = {
        'total_reviews': int(len(df)),
        'total_series': int(df['series_id'].nunique()),
        'total_brands': int(dfb['brand'].nunique()) if has_brand else None,
        'date_range': {
            'earliest': str(summary['earliest_review'].min()),
            'latest': str(summary['latest_review'].max()),
        },
        'quality': {
            'duplicate_review_ids': dup_rev,
            'duplicate_ratio': round(dup_rev / len(df), 4),
            'missing_rating': missing_rating,
            'missing_rating_ratio': round(missing_rating / len(df), 4),
            'empty_content': missing_content,
            'series_with_no_rating': empty_rating_series,
        },
        'sentiment_polarity': {
            'rated_reviews': rated_total,
            'positive_ge_4_5': pos,
            'neutral_3_5_4_5': neu,
            'negative_lt_3_5': neg,
            'positive_ratio': round(pos / rated_total, 3) if rated_total else None,
            'negative_ratio': round(neg / rated_total, 3) if rated_total else None,
        },
    }
    if brand_stats is not None:
        report['by_brand'] = [
            {'brand': b, 'series': int(r.series), 'reviews': int(r.reviews),
             'avg_rating': float(r.avg_rating) if pd.notna(r.avg_rating) else None}
            for b, r in brand_stats.iterrows()
        ]

    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Markdown 报告
    lines = ["# 舆情数据质量与覆盖报告", "",
             f"- 总评论数: **{report['total_reviews']:,}**",
             f"- 覆盖车系: **{report['total_series']:,}**",
             f"- 覆盖品牌: **{report['total_brands']}**",
             f"- 评论时间范围: {report['date_range']['earliest']} ~ {report['date_range']['latest']}",
             "",
             "## 数据质量",
             f"- 重复 review_id: {report['quality']['duplicate_review_ids']} "
             f"({report['quality']['duplicate_ratio']*100:.2f}%)",
             f"- 缺失总评分: {report['quality']['missing_rating']} "
             f"({report['quality']['missing_rating_ratio']*100:.2f}%)",
             f"- 空内容: {report['quality']['empty_content']}",
             f"- 无评分车系: {report['quality']['series_with_no_rating']}",
             "",
             "## 情感极性 (有评分子集)",
             f"- 正向(≥4.5): {report['sentiment_polarity']['positive_ge_4_5']} "
             f"({report['sentiment_polarity']['positive_ratio']*100:.1f}%)",
             f"- 中性(3.5–4.5): {report['sentiment_polarity']['neutral_3_5_4_5']}",
             f"- 负向(<3.5): {report['sentiment_polarity']['negative_lt_3_5']} "
             f"({report['sentiment_polarity']['negative_ratio']*100:.1f}%)",
             ""]
    if brand_stats is not None:
        lines += ["## 分品牌覆盖", "",
                  "| 品牌 | 车系数 | 评论数 | 平均评分 |",
                  "|------|--------|--------|----------|"]
        for b, r in brand_stats.iterrows():
            ar = f"{r.avg_rating:.2f}" if pd.notna(r.avg_rating) else "—"
            lines.append(f"| {b} | {int(r.series)} | {int(r.reviews):,} | {ar} |")
        lines.append("")
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[3/4] 质量报告 -> {os.path.basename(REPORT_JSON)} / {os.path.basename(REPORT_MD)}")
    print(f"[4/4] 完成. 品牌数={report['total_brands']}, "
          f"正向率={report['sentiment_polarity']['positive_ratio']*100:.1f}%, "
          f"负向率={report['sentiment_polarity']['negative_ratio']*100:.1f}%")
    if brand_stats is not None:
        print("\n分品牌:")
        for b, r in brand_stats.iterrows():
            ar = f"{r.avg_rating:.2f}" if pd.notna(r.avg_rating) else '—'
            print(f"  {b:>6}: {int(r.series):>3}系 / {int(r.reviews):>6,}条 / 均分{ar}")


if __name__ == '__main__':
    main()
