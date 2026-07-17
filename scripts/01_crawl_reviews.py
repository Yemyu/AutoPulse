#!/usr/bin/env python3
"""
懂车帝口碑舆情采集
  基于懂车帝公开JSON API, 纯requests请求

数据用途:
  - 与 vehicles.csv 通过 series_id 关联 → 车辆参数(X特征)
  - 与 sales.csv 通过 series_id 关联 → 月度销量(Y目标)
  - 聚合为车系级情感指标 → 影响因子回归分析

运行方式:
  python 01_crawl_reviews.py              # 全量采集(默认大众)
  python 01_crawl_reviews.py --all        # 全品牌采集(所有有销量的车系)
  python 01_crawl_reviews.py --series 415 # 指定单个车系

输出:
  sentiment_reviews.csv — 评论明细表(40054条)
"""

import os
import sys
import time
import random
import json
import argparse
from datetime import datetime

import requests
import pandas as pd

#  配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # 项目根目录
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")          # 原始数据
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "sentiment") # 产出数据

API_URL = "https://www.dongchedi.com/motor/pc/car/series/get_review_list"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.dongchedi.com/',
}

DEFAULTS = {
    'max_per_series': 200,      # 每个车系最多采多少条(建议100-200做统计分析足够)
    'page_size': 15,            # API实际每页返回15条(不要改这个值)
    'delay': (1.5, 3.5),       # 请求间隔(秒), 随机
}


#  核心功能

def load_vehicle_series(brand_filter=None):
    """
    从车辆主表读取目标车系
    返回: DataFrame(series_id, series_name, brand_name) 只含int型ID的车系
    """
    veh_path = os.path.join(RAW_DIR, "vehicles.csv")
    df = pd.read_csv(veh_path, encoding='utf-8-sig', low_memory=False)

    # 去重取车系级别
    series_df = df[['series_id', 'series_name', 'brand_name']].drop_duplicates()

    if brand_filter:
        series_df = series_df[series_df['brand_name'] == brand_filter].copy()

    # 只保留 int 型 series_id(API需要)
    def is_int(x):
        try:
            int(str(x).strip())
            return True
        except:
            return False
    series_df = series_df[series_df['series_id'].apply(is_int)].copy()
    series_df['series_id'] = series_df['series_id'].astype(int)

    return series_df.reset_index(drop=True)


def load_sales_series():
    """从销量表读取有销量记录的车系(用于--all模式)"""
    sales_path = os.path.join(RAW_DIR, "sales.csv")
    df = pd.read_csv(sales_path, encoding='utf-8-sig', low_memory=False)
    return df[['series_id', 'series_name', 'brand']].drop_duplicates()


def fetch_koubei_page(series_id, page=1, size=20):
    """
    调用API获取一页口碑数据
    返回: (reviews列表, total_count, has_more) 或 (None, None, None) on error
    """
    params = {
        'series_id': int(series_id),
        'page': page,
        'size': size,
        'city_name': '',
        'sort_by': 'default',
    }
    try:
        resp = requests.get(API_URL, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    ⚠️ 请求失败: {e}")
        return None, None, None

    if not data.get('data'):
        return [], 0, False

    d = data['data']
    return (
        d.get('review_list', []),
        d.get('total_count', 0),
        d.get('has_more', False),
    )


def parse_review(rev, series_id, series_name):
    """将一条原始API评论解析为结构化字典"""
    user_info = rev.get('user_info') or {}
    buy_info = rev.get('buy_car_info') or {}
    score_info = rev.get('score_info') or {}

    # 时间戳→日期
    ts = rev.get('create_time')
    publish_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else None

    # 总评分(百分制→5分制)
    raw_score = score_info.get('score')
    rating = round(raw_score / 100.0, 1) if raw_score is not None else None

    # 各维度分(百分制→5分制)
    dim_keys = [
        'appearance_score', 'space_score', 'interiors_score',
        'power_score', 'control_score', 'comfort_score',
        'oil_consumption_score', 'configuration_score',
    ]
    dim_scores = {}
    for k in dim_keys:
        v = score_info.get(k)
        if v is not None:
            dim_scores[k] = round(v / 100.0, 1)

    return {
        # === 关联键 ===
        'series_id': int(series_id),
        'series_name': series_name,

        # === 评论标识 ===
        'review_id': rev.get('gid_str', ''),
        'platform': 'dongchedi',

        # === 用户信息 ===
        'user_nickname': user_info.get('name', ''),
        'user_id': str(user_info.get('user_id', '')),

        # === 时间 ===
        'publish_time': publish_date,

        # === 内容 ===
        'content': rev.get('content', ''),
        'content_len': len(rev.get('content', '')),

        # === 评分(核心字段, 用于NLP+回归) ===
        'rating_overall': rating,
        'rating_appearance': dim_scores.get('appearance_score'),
        'rating_space': dim_scores.get('space_score'),
        'rating_interiors': dim_scores.get('interiors_score'),
        'rating_power': dim_scores.get('power_score'),
        'rating_control': dim_scores.get('control_score'),
        'rating_comfort': dim_scores.get('comfort_score'),
        'rating_oil_consumption': dim_scores.get('oil_consumption_score'),
        'rating_config': dim_scores.get('configuration_score'),

        # === 互动数据 ===
        'digg_count': rev.get('digg_count_en'),
        'comment_count': rev.get('comment_count_en'),

        # === 购买信息 ===
        'car_model': buy_info.get('car_name', ''),
        'buy_location': buy_info.get('location', ''),
        'buy_price': buy_info.get('price', ''),
        'buy_time': buy_info.get('bought_time', ''),
        'fuel_type': buy_info.get('fuel_form', ''),
        'consumption': buy_info.get('consumption', ''),
    }


def crawl_one_series(series_id, series_name, max_reviews=200):
    """
    采集单个车系的全部口碑(分页)
    返回: list[dict] 评论列表
    """
    all_revs = []
    page = 1

    while len(all_revs) < max_reviews:
        review_list, total_count, has_more = fetch_koubei_page(
            series_id, page=page, size=DEFAULTS['page_size']
        )

        if review_list is None:
            break  # 网络错误, 终止该车系

        if not review_list:
            print(f"    无评论数据(total={total_count})")
            break

        for rev in review_list:
            if len(all_revs) >= max_reviews:
                break
            parsed = parse_review(rev, series_id, series_name)
            all_revs.append(parsed)

        print(f"    第{page}页: +{len(review_list)}条 "
              f"(累计{len(all_revs)}/{total_count}总)",
              flush=True)

        if not has_more:
            break

        page += 1
        time.sleep(random.uniform(*DEFAULTS['delay']))

    return all_revs


def append_csv(reviews, output_path):
    """增量追加写入CSV(防中断丢数据)"""
    if not reviews:
        return
    df = pd.DataFrame(reviews)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        # 写入带BOM的UTF-8, Excel友好
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    else:
        df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8-sig')


def get_completed_ids(output_path):
    """已完成的series_id集合(断点续传)"""
    if not os.path.exists(output_path):
        return set()
    try:
        df = pd.read_csv(output_path, encoding='utf-8-sig')
        return set(df['series_id'].astype(int).unique())
    except Exception:
        return set()


def generate_summary(output_path):
    """
    生成车系级汇总统计(用于与vehicles/sales关联分析)
    输出: sentiment_summary.csv
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return None

    df = pd.read_csv(output_path, encoding='utf-8-sig')
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

    # 计算正/中/负比例
    total_check = summary['positive_cnt'] + summary['neutral_cnt'] + summary['negative_cnt']
    summary['positive_ratio'] = (summary['positive_cnt'] / total_check).round(3)
    summary['neutral_ratio'] = (summary['neutral_cnt'] / total_check).round(3)
    summary['negative_ratio'] = (summary['negative_cnt'] / total_check).round(3)

    # 各维度均分
    dim_cols = [c for c in df.columns if c.startswith('rating_') and c != 'rating_overall']
    dim_agg = df.groupby(['series_id', 'series_name'])[dim_cols].mean().reset_index()
    dim_agg.columns = [f'avg_{c}' if c not in ('series_id', 'series_name') else c
                        for c in dim_agg.columns]

    summary = summary.merge(dim_agg, on=['series_id', 'series_name'], how='left')

    out_path = output_path.replace('reviews.csv', 'summary.csv')
    summary.to_csv(out_path, index=False, encoding='utf-8-sig')

    return summary, out_path


#  主流程

def main():
    parser = argparse.ArgumentParser(description='懂车帝口碑舆情采集器')
    parser.add_argument('--brand', type=str, default='大众',
                        help='目标品牌(默认大众)')
    parser.add_argument('--all', action='store_true',
                        help='全品牌模式(采集所有有销量的车系)')
    parser.add_argument('--series', type=int, nargs='+',
                        help='指定series_id列表')
    parser.add_argument('--max', type=int, default=DEFAULTS['max_per_series'],
                        help=f'每车系最大采集数(默认{DEFAULTS["max_per_series"]})')
    parser.add_argument('--brands', type=str, default=None,
                        help='逗号分隔品牌列表, 配合--all仅采集这些品牌(按销量表筛选)')
    args = parser.parse_args()

    start = time.time()
    output_file = os.path.join(OUTPUT_DIR, "sentiment_reviews.csv")

    print("=" * 60)
    print("  懂车帝口碑舆情采集器")
    print("=" * 60)

# 确定目标
    if args.series:
        # 指定车系
        target = pd.DataFrame({
            'series_id': args.series,
            'series_name': [f'指定_{sid}' for sid in args.series],
            'brand_name': ['手动']*len(args.series),
        })
    elif args.all:
        # 全品牌: 取销量表中所有有int型ID的车系
        sales_df = load_sales_series()
        if args.brands:
            blist = [b.strip() for b in args.brands.split(',')]
            sales_df = sales_df[sales_df['brand'].isin(blist)]
            print(f"  (按品牌筛选: {blist})")
        def is_int(x):
            try:
                int(str(x).strip()); return True
            except: return False
        sales_int = sales_df[sales_df['series_id'].apply(is_int)].copy()
        sales_int['series_id'] = sales_int['series_id'].astype(int)
        target = sales_int.rename(columns={'brand': 'brand_name'})[
            ['series_id', 'series_name', 'brand_name']
        ].drop_duplicates()
    else:
        # 默认: 按品牌从车辆主表取
        target = load_vehicle_series(brand_filter=args.brand)

    print(f"\n  目标车系: {len(target)} 个")

# 断点续传
    completed = get_completed_ids(output_file)
    if completed:
        pending = target[~target['series_id'].isin(completed)]
        print(f"  ✓ 已完成 {len(completed)} 个, 待采集 {len(pending)} 个")
        target = pending.reset_index(drop=True)

    if target.empty:
        print("\n  ✅ 所有目标已完成!")
        return

# 开始采集
    all_new = []
    success, empty, fail = 0, 0, 0

    print(f"\n{'─'*60}")
    print(f"  开始采集 ({len(target)}个车系, 每系上限{args.max}条)")
    print(f"{'─'*60}\n")

    for idx, row in target.iterrows():
        sid = int(row['series_id'])
        sname = row['series_name']

        print(f"[{idx+1}/{len(target)}] {sname} (ID:{sid})", flush=True)
        reviews = crawl_one_series(sid, sname, max_reviews=args.max)

        if reviews:
            append_csv(reviews, output_file)
            all_new.extend(reviews)
            success += 1
            print(f"  ✓ {len(reviews)}条")
        elif reviews == []:  # 空列表=无数据(非错误)
            empty += 1
            print(f"  ✗ 无数据")
        else:  # None = 错误
            fail += 1
            print(f"  ⚠️ 失败")

        completed.add(sid)
        if idx < len(target) - 1:
            time.sleep(random.uniform(1.0, 2.0))

# 汇总
    elapsed = time.time() - start
    final_total = len(all_new)

    print(f"\n{'='*60}")
    print(f"  ✅ 采集完成!")
    print(f"  新增: {final_total} 条 | 成功: {success} | 空: {empty} | 失败: {fail}")
    print(f"  文件: {output_file}")

    # 生成汇总表
    result = generate_summary(output_file)
    if result:
        summary_df, summary_path = result
        print(f"  汇总: {summary_path} ({len(summary_df)}个车系的聚合指标)")

    print(f"  耗时: {elapsed:.1f}秒")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
