<p align="center">
  <a href="./README.md">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="./README_EN.md">🌐 English</a>
</p>

# AutoPulse · 汽车口碑舆情分析

> 多品牌车型用户口碑采集 + 车型配置 / 销量联合分析，量化用户舆情对销量的影响。

## 项目简介

AutoPulse 是一套面向汽车行业的舆情监控与分析流水线，分三层：

1. **采集层** — 通过懂车帝公开口碑 API 采集多品牌车型的用户口碑（星级评分 + 文本评论），纯 `requests`，零浏览器、零登录。
2. **静态层** — 车型配置参数（懂车帝）、月度销量（太平洋汽车）。
3. **分析层** — 车系级情感指标聚合，结合配置 / 销量做影响因子分析。

三张主表通过 `series_id`（车系 ID）关联。

## 数据文件

| 文件 | 记录数 | 时间范围 | 说明 |
|------|--------|---------|------|
| `data/raw/vehicles.csv` | 1,139 车系（4,334 配置版本） | 2022–2026 款 | 车型配置（92 列特征） |
| `data/raw/sales.csv` | 33,845 条（1,122 车系） | 2022-01 ~ 2026-05 | 月度销量 |
| `data/sentiment/sentiment_reviews.csv` | 40,054 条（490 车系） | 2019-06 ~ 2026-07 | 用户口碑明细 |
| `data/sentiment/sentiment_summary.csv` | 490 车系 | — | 车系级情感聚合指标 |
| `data/sentiment/analysis_input.csv` | 490 行（对齐就绪） | — | 舆情+销量+配置，一行一车系 |

> **采集已完成**：覆盖销量表中全部 **502 个整数 ID 车系中的 490 个**（其余 12 个平台无口碑数据），横跨 **95 个品牌**。三表对齐后 **489 个车系同时具备舆情 + 销量 + 配置**，可直接进回归。

## 数据来源与采集

所有数据均来自**公开汽车平台**，由自研爬虫脚本采集，**无人工整理、可完全复现**。原始 / 中间数据已加入 `.gitignore`（体积大），克隆后用三步脚本即可重新生成。

| 数据 | 来源平台 | 采集脚本 | 说明 |
|------|---------|---------|------|
| `vehicles.csv` | 懂车帝 | 配置爬虫 | 车型静态参数（价格 / 尺寸 / 能源 / 动力 / 电池等） |
| `sales.csv` | 太平洋汽车 | 销量爬虫 | 月度销量（跨平台 ID 对齐 + 少量线性插值补缺） |
| `series_mapping.csv` | 懂车帝 × 太平洋 | ID 桥接 | 统一双方车系 ID，使三表可关联 |
| `sentiment_reviews.csv` | 懂车帝口碑 API | `01_crawl_reviews.py` | 用户口碑明细（评分 + 文本），逐页采集、断点续传 |
| `sentiment_summary.csv` | 由 reviews 聚合 | `03_build_sentiment_summary.py` | 车系级情感指标 |

**配置数据清洗（2026-07-09 完成）**：`vehicles.csv` 从原始 **248 列**经去冗余精简至 **92 列**；缺失值区分「条件缺失」（如纯电动无发动机参数、燃油车无电池参数）予以保留而非填零，避免引入虚假特征；核心字段（价格 / 尺寸 / 能源类型 / 品牌 / 级别）100% 完整。销量数据经跨平台 ID 对齐与少量线性插值补齐时间窗口缺口。

> 三张主表均通过 `series_id` 关联；舆情明细本身**不含品牌列**，需经 `series_id` 回连销量 / 配置表补出品牌（详见 `data/README.md`，含[英文版](data/README_EN.md)）。

## 快速开始

所有脚本在 conda 环境 `nlp-sentiment` 中运行。原始 / 中间数据已 gitignore，以下步骤可完整复现全部数据：

```bash
# 1. 采集全量舆情 (覆盖销量表全部整数ID车系, 断点续传)
python scripts/01_crawl_reviews.py --all --max 100

# 2. 生成车系级汇总 + 数据质量报告
python scripts/03_build_sentiment_summary.py

# 3. 清洗 + 三表对齐 -> analysis_input.csv
python scripts/02_clean_and_align.py
```

## 目录结构

```
AutoPulse/
├── data/
│   ├── README.md       # 数据说明 + 质量报告 (中英文切换)
│   ├── README_EN.md    # English
│   ├── raw/            # vehicles.csv (配置), sales.csv (销量) — 已 gitignore
│   └── sentiment/      # sentiment_reviews.csv, sentiment_summary.csv — 已 gitignore
├── scripts/
│   ├── 01_crawl_reviews.py            # 舆情采集器 v7 (断点续传)
│   ├── 02_clean_and_align.py          # 清洗 + 三表对齐 -> analysis_input.csv
│   ├── 03_build_sentiment_summary.py  # 汇总 + 质量报告生成
│   └── 04_explore_eda.py              # 探索性数据分析与可视化 (阶段二)
├── reference/          # 第三方参考 (Cars_Scraper)
├── requirements.txt
├── README.md          # 中文 (英文版见 README_EN.md)
└── README_EN.md       # English
```

## 后续路线

- [x] 全量舆情采集（502 个整数 ID 车系中的 490 个，横跨 95 品牌，其余 12 个平台无口碑）
- [x] 数据清洗与三表对齐（`02_clean_and_align.py` 产出 analysis_input.csv，489 系具备舆情+销量+配置）
- [ ] 中文 NLP 文本情感（jieba 分词 + SnowNLP）补充星级之外的细粒度情感
- [ ] 跨品牌影响因子回归（口碑指标 × 配置 × 销量）
- [ ] BI 可视化 / 舆情监控看板

---

*本项目为个人数据分析作品集，不包含任何客户敏感信息。*
