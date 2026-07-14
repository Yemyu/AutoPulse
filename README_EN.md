<p align="center">
  <a href="./README.md">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="./README_EN.md">🌐 English</a>
</p>

# AutoPulse · Automotive Review Sentiment Analytics

> Multi-brand user-review collection + combined vehicle-spec / sales analytics, quantifying how user sentiment impacts sales.

## Overview

AutoPulse is an automotive sentiment-monitoring and analytics pipeline with three layers:

1. **Collection layer** — scrapes multi-brand user reviews (star ratings + text) from Dongchedi's public review API, pure `requests`, no browser, no login.
2. **Static layer** — vehicle specifications (Dongchedi) and monthly sales (pcauto).
3. **Analytics layer** — series-level sentiment aggregation, combined with spec / sales for impact-factor analysis.

The three core tables are linked through `series_id` (car-series ID).

## Data Files

| File | Records | Period | Description |
|------|---------|--------|-------------|
| `data/raw/vehicles.csv` | 1,139 series (4,334 trim versions) | 2022–2026 | Vehicle specs (92 feature cols) |
| `data/raw/sales.csv` | 33,845 (1,122 series) | 2022-01 ~ 2026-05 | Monthly sales |
| `data/sentiment/sentiment_reviews.csv` | 40,054 (490 series) | 2019-06 ~ 2026-07 | User review details |
| `data/sentiment/sentiment_summary.csv` | 490 series | — | Series-level sentiment aggregation |
| `data/sentiment/analysis_input.csv` | 490 rows (aligned) | — | Sentiment + sales + spec, one row per series |

> **Collection complete**: covers **490 of the 502 integer-ID series** in the sales table (the other 12 have no review data on the platform), spanning **95 brands**. After aligning the three tables, **489 series have sentiment + sales + spec simultaneously**, ready for regression.

## Data Sources & Collection

All data come from **public automotive platforms**, collected by self-built crawler scripts — **no manual curation, fully reproducible**. Raw / intermediate data are in `.gitignore` (too large); after cloning, regenerate everything with the three-step scripts below.

| Data | Source platform | Script | Description |
|------|-----------------|--------|-------------|
| `vehicles.csv` | Dongchedi | spec crawler | Static vehicle params (price / dimensions / energy / power / battery, etc.) |
| `sales.csv` | pcauto | sales crawler | Monthly sales (cross-platform ID alignment + light linear interpolation) |
| `series_mapping.csv` | Dongchedi × pcauto | ID bridge | Unifies both platforms' series IDs so the three tables can join |
| `sentiment_reviews.csv` | Dongchedi review API | `01_crawl_reviews.py` | User review details (rating + text), paginated, resumable |
| `sentiment_summary.csv` | aggregated from reviews | `03_build_sentiment_summary.py` | Series-level sentiment metrics |

**Spec-data cleaning (done 2026-07-09):** `vehicles.csv` was reduced from **248 raw columns** to **92** after deduplication; missing values are distinguished as "conditional missing" (e.g. BEVs lack engine params, fuel cars lack battery params) and kept rather than zero-filled, avoiding fake features; core fields (price / dimensions / energy type / brand / class) are 100% complete. Sales data were gap-filled via cross-platform ID alignment and light linear interpolation.

> The three core tables are linked through `series_id`; the review details themselves **have no brand column**, so brand must be recovered by joining back to the sales / spec tables via `series_id` (see `data/README_EN.md`; Chinese version: `data/README.md`).

## Quick Start

All scripts run in the conda environment `nlp-sentiment`. Raw / intermediate data are gitignored; the steps below fully reproduce all data:

```bash
# 1. Collect full sentiment (covers all integer-ID series in the sales table, resumable)
python scripts/01_crawl_reviews.py --all --max 100

# 2. Build series-level summary + data quality report
python scripts/03_build_sentiment_summary.py

# 3. Clean + align three tables -> analysis_input.csv
python scripts/02_clean_and_align.py
```

## Directory Structure

```
AutoPulse/
├── data/
│   ├── README.md       # data description + quality report (bilingual switch)
│   ├── README_EN.md    # English
│   ├── raw/            # vehicles.csv (specs), sales.csv (sales) — gitignored
│   └── sentiment/      # sentiment_reviews.csv, sentiment_summary.csv — gitignored
├── scripts/
│   ├── 01_crawl_reviews.py            # review crawler v7 (resumable)
│   ├── 02_clean_and_align.py          # clean + align three tables -> analysis_input.csv
│   ├── 03_build_sentiment_summary.py  # summary + quality report
│   └── 04_explore_eda.py              # exploratory data analysis & visualization (Stage 2)
├── reference/          # third-party reference (Cars_Scraper)
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Full sentiment collection (490 of 502 integer-ID series, 95 brands; other 12 have no platform reviews)
- [x] Data cleaning & three-table alignment (`02_clean_and_align.py` → `analysis_input.csv`, 489 series with sentiment + sales + spec)
- [ ] Chinese NLP text sentiment (jieba tokenization + SnowNLP) to supplement fine-grained sentiment beyond star ratings
- [ ] Cross-brand impact-factor regression (sentiment metrics × spec × sales)
- [ ] BI visualization / sentiment monitoring dashboard

---

*This project is a personal data-analytics portfolio piece and contains no client-sensitive information.*
