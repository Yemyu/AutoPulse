<p align="center">
  <a href="./README.md">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="./README_EN.md">🌐 English</a>
</p>

# Data Documentation

This file is the single source of truth for all AutoPulse datasets — vehicle specifications, monthly sales, series-ID mapping, and user sentiment — used for modeling, analysis, and the dashboard. All data come from public automotive platforms and are used for learning, research, and demonstration only.

## Data Sources

| Data category | Source platform(s) | Collection method |
|---------------|--------------------|-------------------|
| Vehicle specs / monthly sales (raw) | Public auto platforms such as Autohome (汽车之家) and PCauto (太平洋汽车) | Comprehensively collected |
| User sentiment / reviews | Dongchedi (懂车帝) public reviews | Automated via `scripts/01_crawl_reviews.py` |

## Dataset Overview

The three core tables are linked through `series_id` (car-series ID):

```
vehicles ———— series_id ———— series_mapping ———— series_id ———— sales
(vehicle spec features)                                      (monthly sales)
```

For XGBoost: aggregate `sales` by series → left-join `vehicles` (take the first spec row) →
filter `has_sales_data=True` → a 1,122-row × 90+-column training set.

---

## 1. `vehicles.csv` — Vehicle Specification Data

### Overview

| Item | Value |
|------|-------|
| Records | 4,334 (1,139 series × multiple trim versions) |
| Brands | 153 |
| Feature columns | 92 (reduced from 248 raw columns after cleaning) |
| Model years | 2022–2026 |
| Source | Public auto platforms (Autohome, PCauto, etc.), comprehensively collected |

**Energy-type distribution (deduplicated by series):**

| Type | Series | Share |
|------|--------|-------|
| Fuel | 493 | 43.3% |
| BEV (Battery EV) | 344 | 30.2% |
| PHEV (Plug-in Hybrid) | 211 | 18.5% |
| Range Extender (EREV) | 76 | 6.7% |
| HEV (Hybrid) | 10 | 0.9% |
| PHEV + BEV | 4 | 0.4% |
| Other (Fuel Cell) | 1 | 0.1% |

**Top-5 body classes (deduplicated by series):** Compact SUV (246) > Mid-size SUV (170) > Compact Car (150) > Large SUV (95) > Small SUV (93)

### Missing Values

Total missing = 90,042, all **conditional missing** (legitimate, no imputation needed):

- BEV / EREV → no engine parameters (26 NaN columns)
- Fuel → no motor / battery parameters (13 NaN columns)
- Core fields (price, dimensions, energy type, brand, class) are **100% complete**

### Key Columns

#### Basic specs (4 cols, no missing)

| Column | Description | Range |
|--------|-------------|-------|
| `official_price_wan` | MSRP (10k CNY) | 2.98 ~ 130, mean 19 |
| `energy_type` | Energy type | Fuel / BEV / PHEV / EREV / HEV |
| `vehicle_class` | Body class | Compact SUV / Mid-size Car / MPV, etc. (18 classes) |
| `manufacturer` | Manufacturer | — |

**Price:** median 153.8k CNY for fuel & BEV; EREV most expensive (median 209.8k); priciest model 1.3M CNY (EREV).

#### Engine params (26 cols; present for Fuel/PHEV, NaN for BEV/EREV)

| Column | Description | Typical |
|--------|-------------|---------|
| `engine_displacement_l` | Displacement (L) | 1.2 ~ 3.0 |
| `engine_max_horsepower_ps` | Max horsepower (PS) | mean 181 |
| `engine_max_torque_nm` | Max torque (N·m) | mean 269 |
| `engine_intake_type` | Intake type | Turbo / NA |
| `fuel_form` | Fuel type | Gasoline / Diesel |
| `gearbox_type` | Gearbox | DCT / AT / CVT / Manual |

#### Motor params (9 cols; present for EV/Hybrid, NaN for Fuel)

| Column | Description | Typical |
|--------|-------------|---------|
| `motor_total_power_kw` | Total motor power (kW) | mean 191, max 880 |
| `motor_total_torque_nm` | Total motor torque (N·m) | — |
| `motor_front_power_kw` | Front motor power (kW) | FWD-dominated |
| `motor_rear_power_kw` | Rear motor power (kW) | Key for AWD / performance |

#### Battery params (4 cols; partial, heavily missing)

| Column | Description | Note |
|--------|-------------|------|
| `battery_range_km` | Electric range (km) | mean 424, well covered |
| `battery_capacity_kwh` | Capacity (kWh) | ⚠️ 97.4% missing, not a feature |
| `battery_type` | Battery type | ⚠️ 96.9% missing |
| `battery_warranty` | Battery warranty | ⚠️ 92.5% missing |

#### Dimensions (13 cols, no missing)

| Column | Description | Mean |
|--------|-------------|------|
| `length_mm` | Length | 4,761 mm |
| `width_mm` | Width | 1,876 mm |
| `height_mm` | Height | 1,656 mm |
| `wheelbase_mm` | **Wheelbase** (space metric) | 2,831 mm |
| `curb_weight_kg` | Curb weight | 1,782 kg |
| `body_structure` | Body structure | Sedan / SUV / MPV |

#### Safety (3 cols, no missing)

| Column | Description |
|--------|-------------|
| `driver_airbag` | Driver airbag |
| `side_airbag` | Side / curtain airbag |
| `knee_airbag` | Knee airbag (high-end, not standard) |

#### Comfort & Tech (13 cols, no missing)

| Column | Description |
|--------|-------------|
| `center_screen` | Center display |
| `seat_material` | Seat material (synthetic / leather / fabric) |
| `sound_brand` | Audio brand (Harman Kardon / BOSE, etc.) |
| `speaker_count` | Speaker count |
| `seat_heating` | Seat heating |
| `seat_massage` | Seat massage |
| `seat_ventilation` | Seat ventilation |
| `aircon_control` | A/C control (auto / manual) |

---

## 2. `sales.csv` — Monthly Sales Data

### Overview

| Item | Value |
|------|-------|
| Records | 33,845 |
| Series | 1,122 |
| Brands | 152 |
| Period | 2022-01 ~ 2026-05 |
| Total sales | ~96.52 million units |
| Quality | 0 negative sales ✅ 0 zero-sales anomalies ✅ |
| Main source | pcauto (98.5%) + dongchedi_fill interpolation (1.5%) |

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `year` | int | Year (2022~2026) |
| `month` | int | Month (1~12) |
| `series_id` | str | Series ID, joins vehicles.series_id |
| `series_name` | str | Series name |
| `brand` | str | Brand |
| `category` | str | SUV / Sedan / MPV |
| `monthly_sales` | int | **Monthly sales (units)** |
| `data_source` | str | pcauto / dongchedi_fill |

### Annual Trend

| Year | Total (M units) | Series covered | Full-12-month series |
|------|-----------------|---------------|----------------------|
| 2022 | 20.30 | 680 | 476 (70%) |
| 2023 | 21.87 | 725 | 526 (73%) |
| 2024 | 23.07 | 748 | 575 (77%) |
| 2025 | 24.04 | 796 | 596 (75%) |
| 2026 (Jan–May) | 7.24 | 733 | — |

**Trend:** SUV annual sales grew from 9.15M (2022) to 11.96M (2025), overtaking sedans as China's #1 category in 2024.

### Monthly Sales Distribution

| Percentile | Monthly sales | Interpretation |
|-----------|---------------|---------------|
| **P10** | 12 | Niche, sells <几十 units/month |
| P25 | 123 | Small-volume model |
| **P50 (median)** | 748 | Half of models sell < 1k/month |
| P75 | 3,043 | Hot-seller threshold |
| **P90** | 8,455 | Near-10k = top 10% |
| P95 | 13,055 | Top 5% evergreen |
| **P99** | 25,707 | "Legend" tier |

### Brand Landscape (2022–2026 cumulative)

| Rank | Brand | Total (M units) | Note |
|------|-------|-----------------|------|
| 1 | **BYD** | **11.63** | Dominant NEV leader |
| 2 | Volkswagen | 8.98 | Traditional fuel king |
| 3 | Toyota | 7.13 | Steady |
| 4 | Honda | 4.26 | |
| 5 | Geely | 4.13 | #2 domestic |
| 6 | Wuling | 3.45 | Mini-car king |
| 7 | Nissan | 2.83 | |
| 8 | Changan | 2.77 | |
| 9 | BMW | 2.69 | #1 luxury |
| 10 | Audi | 2.63 | |
| 11 | Tesla | 2.51 | Only pure-EV in top 10 |
| 12 | Mercedes-Benz | 2.44 | |

### Top-10 Series by Cumulative Sales

| Series | Brand | Cumulative (M units) |
|--------|-------|----------------------|
| Model Y | Tesla | 1.81 |
| Qin PLUS | BYD | 1.66 |
| Sylphy | Nissan | 1.54 |
| Song PLUS NEV | BYD | 1.47 |
| Hongguang MINIEV | Wuling | 1.40 |
| Lavida | Volkswagen | 1.38 |
| Sagitar | Volkswagen | 1.09 |
| Seagull | BYD | 1.05 |
| CS75 PLUS | Changan | 0.93 |
| Yuan PLUS | BYD | 0.91 |

---

## 3. `series_mapping.csv` — Series ID Mapping

### Purpose

Bridge table. Vehicle specs and monthly sales come from different platform ID systems; this table aligns them so the three core tables join on a unified `series_id`.

### Overview

| Item | Value |
|------|-------|
| Records | 1,139 |
| Brands | 155 |
| Quality | 100% complete, no missing |

### Columns

| Column | Description |
|--------|-------------|
| `统一后series_id` | **Final unified ID**, use this for JOINs |
| `车辆主表series_id` | vehicles original series_id |
| `销量表原series_id` | sales original series_id |
| `series_name` | Series name |
| `brand_name` | Brand name |
| `核对状态` | consistent (14) / unified (536) / series-level补采 (589) |
| `has_sales_data` | has sales data? (1,122 yes / 17 no) |

### Status Meaning

- **consistent (14):** IDs identical across sources, no action needed
- **unified (536):** name/ID differences (e.g. "型格-Integra"→"型格"), rule-aligned
- **series-level补采 (589):** vehicles ID from Dongchedi supplemental crawl, no pcauto match. **No impact on joins**

---

## 4. Series Missing Sales

17 series have spec data but no sales (flagged `has_sales_data=False`):

| Series | Brand | Energy | Reason |
|--------|-------|--------|--------|
| Avatr 06/07/11/12 | Avatr | EREV | New, not yet recorded |
| Arcfox 贝塔S3/T1/问道V9 | BAIC BJEV | BEV/EREV | New, not yet recorded |
| iCAR 超级V23 | Chery NEV | BEV | New |
| Star Era ET | Exeed | EREV | New |
| Tiggo 8L | Chery | Fuel | Launched 2024 |
| Ariya | Dongfeng Nissan | BEV | Source gap |
| CS35 / CS55 PLUS | Changan | Fuel | Source gap |
| S50 EV / D60 / T60 EV | Dongfeng | BEV | Source gap |
| ID. ERA 9X | SAIC Volkswagen | EREV | Not yet launched |

**Impact:** all are new models with < 24 months of data; no effect on ARIMA, just filter in XGBoost.

---

## 5. Usage Guide

### For XGBoost / LightGBM

1. Aggregate `sales.csv` by `series_id` (sum monthly sales as label, or take latest month)
2. Left-join `vehicles.csv`, one row per series
3. Filter `has_sales_data=True` → 1,122-row training set
4. Usable features: price, dimensions, energy type, horsepower/power, gearbox, safety, comfort
5. Conditional NaN needs no imputation; XGBoost handles it natively

### For ARIMA / Prophet / LSTM

1. Filter `data_source='pcauto'` and mature series with ≥ 36 months coverage
2. Build a date column from `year` + `month`
3. Model per `series_id` group
4. Note: 2026 only has Jan–May data

### Feature Engineering Tips

**Numeric features ready to use:**
`official_price_wan`, `length_mm`, `wheelbase_mm`, `curb_weight_kg`, `engine_max_horsepower_ps`, `engine_max_torque_nm`, `motor_total_power_kw`, `battery_range_km`, `seat_count`, `max_speed_kmh`

**Categorical features to encode:**
`energy_type`, `vehicle_class`, `manufacturer`, `gearbox_type`, `engine_intake_type`, `fuel_form`, `body_structure`, `seat_material`, `sound_brand`, `driver_airbag`, `side_airbag`

**Columns to avoid:**
`battery_type` (96.9% missing), `battery_capacity_kwh` (97.4% missing), `fuel_consumption_l_100km` (85.3% missing), and all ID/URL/crawler-metadata fields

### For Sentiment Impact-Factor Regression

1. Left-join `sentiment_summary.csv` to `sales.csv` by `series_id` (aggregate monthly sales as label Y)
2. Then left-join `vehicles.csv` (price / class / energy as features X)
3. Features: sentiment (positive-rate / avg-score / 8 dimensions) + vehicle (price / class / energy)
4. Use standardized coefficients + Bootstrap CI (more robust than p-values with limited series)
5. Sentiment only covers a subset of series; intersect with sales before regression

---

## 6. Sentiment / Review Data

**Source:** Dongchedi public review API (`dongchedi.com/motor/pc/car/series/get_review_list`),
pure `requests`. Crawler: `scripts/01_crawl_reviews.py`.

### File List

| File | Records | Period | Description |
|------|---------|--------|-------------|
| `data/sentiment/sentiment_reviews.csv` | 40,054 (490 series) | 2019-06 ~ 2026-07 | Review details |
| `data/sentiment/sentiment_summary.csv` | 490 series | — | Series-level sentiment aggregation |
| `data/README.md` Chapter 8 | — | — | Coverage & quality report (merged into this data README) |

> Regenerate summary / quality report: `python scripts/03_build_sentiment_summary.py` (reads
> `sentiment_reviews.csv`, outputs `sentiment_summary.csv` and
> `data_quality_report.json` / `.md` to `data/sentiment/`, reproducible).

### Collection Scope (full crawl completed 2026-07-10)

- **Target:** all **integer-ID series** in the sales table (Dongchedi API only accepts integer IDs) — 502 total
- **Actual:** **490 series / 40,054 reviews**, covering **95 brands** (the other 12 target series had no review data on the platform, recorded as empty)
- Review time span ~7 years (2019-06 ~ 2026-07), enabling long-term sentiment trend analysis

**Top-20 brands by coverage (by review count):**

| Brand | Series | Reviews | Avg score |
|-------|--------|---------|-----------|
| Volkswagen | 23 | 4,242 | 4.10 |
| Toyota | 18 | 1,548 | 4.11 |
| Honda | 16 | 1,461 | 3.99 |
| BYD | 19 | 1,444 | 4.09 |
| Audi | 17 | 1,425 | 4.18 |
| Changan | 14 | 1,264 | 4.14 |
| Hongqi | 17 | 1,169 | 4.27 |
| Mercedes-Benz | 13 | 1,120 | 4.15 |
| Chery | 11 | 915 | 4.04 |
| Exeed | 10 | 884 | 4.32 |
| Leapmotor | 9 | 854 | 4.27 |
| GAC Trumpchi | 9 | 823 | 4.22 |
| Geely | 9 | 817 | 4.26 |
| Lynk & Co | 8 | 800 | 4.32 |
| Cadillac | 10 | 787 | 4.24 |
| Chery Fulwin | 9 | 757 | 4.20 |
| Buick | 8 | 739 | 4.07 |
| Aion | 8 | 728 | 4.24 |
| Nissan | 8 | 725 | 4.06 |
| BMW | 9 | 713 | 4.24 |

> Full 95-brand breakdown: see Chapter 8 "Sentiment Data Quality & Coverage Report" in this file.

### Data Quality

| Metric | Value |
|--------|-------|
| Duplicate `review_id` | 558 (1.39%) |
| Missing overall score | 259 (0.65%) |
| Empty content | 0 |
| Series with no score | 0 |

→ Quality is good; duplication / missingness are negligible. Just `drop_duplicates(subset='review_id')` before analysis.

### Sentiment Polarity (rated subset, 39,795 reviews)

- Positive (≥4.5): 11,953 (**30.0%**)
- Neutral (3.5–4.5): 24,696 (**62.1%**)
- Negative (<3.5): 3,146 (**7.9%**)

> Note: Dongchedi reviews cluster at 4–5 stars, so high neutral share is normal. Text-level
> sentiment (NLP) will supplement the star rating to catch隐性 negative sentiment like
> "high score but low praise".

### `sentiment_reviews.csv` Columns

| Column | Description |
|--------|-------------|
| `series_id` / `series_name` | Series join key + name |
| `review_id` | Unique review ID (for dedup) |
| `platform` | Source platform (dongchedi) |
| `user_nickname` / `user_id` | User nickname / ID |
| `publish_time` | Publish time |
| `content` / `content_len` | Review text / length |
| `rating_overall` | Overall score (5-point) |
| `rating_appearance` ~ `rating_config` | 8-dimension scores (appearance / space / interior / power / handling / comfort / fuel / config) |
| `digg_count` / `comment_count` | Likes / comments |
| `car_model` / `buy_location` / `buy_price` / `buy_time` / `fuel_type` / `consumption` | Purchase info |

> ⚠️ `sentiment_reviews.csv` has **no brand column**. Recover brand via `series_id` join to
> `sales.csv` (`brand`) or `vehicles.csv` (`brand_name`); see `attach_brand()` in
> `scripts/03_build_sentiment_summary.py`.

### `sentiment_summary.csv` Columns

Series-level aggregation, one row per series:

- `review_count` / `avg_rating` / `median_rating` / `min_rating` / `max_rating`
- `avg_content_len` / `total_digg` / `total_comment`
- `earliest_review` / `latest_review`
- `positive_cnt` / `neutral_cnt` / `negative_cnt` + their `_ratio`
- `avg_rating_appearance` … `avg_rating_config` (8-dimension means)

### Relation to the Other Two Tables

```
vehicles ──series_id──┐
                      ├──> sentiment_reviews / sentiment_summary
sales    ──series_id──┘
```

- Sentiment × spec → explain "what kind of cars get good reviews"
- Sentiment × monthly sales → impact-factor regression (does reputation drive sales?)

---

## 7. Analysis-Ready Table (`analysis_input.csv`)

Generated by `scripts/02_clean_and_align.py` — the aligned analysis input after joining all three tables:

- **Input:** `sentiment_reviews.csv` (cleaned & deduped) → series-level aggregation ＋ `sales.csv` (sales aggregated by series) ＋ `vehicles.csv` (series-level spec features)
- **Output:** one row per series, columns include
  - Sentiment: `review_count` / `avg_rating` / `median_rating` / `positive_ratio` / `negative_ratio` / 8-dimension means (`avg_rating_*`)
  - Sales label: `total_sales` / `avg_monthly_sales` / `log_avg_monthly_sales` / `n_months` / `brand` / `category`
  - Vehicle features: `official_price_wan` / `vehicle_class` / `energy_type` / `manufacturer` / `brand_name`
- **Join logic:** all three `series_id` cast to `str` then left-joined; sentiment series matched to sales first (label Y), then to spec (feature X).
- **Current status** (full crawl done): 490 series → 489 matched sales, ready for regression; rerun `02_clean_and_align.py` to refresh with new data.

---

## 8. Sentiment Data Quality & Coverage Report (Full)

> The generation logic lives in `scripts/03_build_sentiment_summary.py`; rerunning outputs
> `data_quality_report.json` and `data_quality_report.md` to `data/sentiment/` (reproducible). The
> snapshot below is merged into this data README.

- Total reviews: **40,054**
- Series covered: **490**
- Brands covered: **95**
- Review time range: 2019-06-24 00:04 ~ 2026-07-10 18:32

### Data Quality

- Duplicate review_id: 558 (1.39%)
- Missing overall score: 259 (0.65%)
- Empty content: 0
- Series with no score: 0

### Sentiment Polarity (rated subset)

- Positive (≥4.5): 11,953 (30.0%)
- Neutral (3.5–4.5): 24,696
- Negative (<3.5): 3,146 (7.9%)

### Brand Coverage (full 95 brands)

| Brand | Series | Reviews | Avg score |
|-------|--------|---------|-----------|
| Volkswagen | 23 | 4,242 | 4.10 |
| Toyota | 18 | 1,548 | 4.11 |
| Honda | 16 | 1,461 | 3.99 |
| BYD | 19 | 1,444 | 4.09 |
| Audi | 17 | 1,425 | 4.18 |
| Changan | 14 | 1,264 | 4.14 |
| Hongqi | 17 | 1,169 | 4.27 |
| Mercedes-Benz | 13 | 1,120 | 4.15 |
| Chery | 11 | 915 | 4.04 |
| Exeed | 10 | 884 | 4.32 |
| Leapmotor | 9 | 854 | 4.27 |
| GAC Trumpchi | 9 | 823 | 4.22 |
| Geely | 9 | 817 | 4.26 |
| Lynk & Co | 8 | 800 | 4.32 |
| Cadillac | 10 | 787 | 4.24 |
| Chery Fulwin | 9 | 757 | 4.20 |
| Buick | 8 | 739 | 4.07 |
| Aion | 8 | 728 | 4.24 |
| Nissan | 8 | 725 | 4.06 |
| BMW | 9 | 713 | 4.24 |
| Deepal | 7 | 694 | 4.24 |
| Changan Nevo | 7 | 652 | 4.09 |
| NIO | 9 | 649 | 4.42 |
| Kia | 10 | 610 | 3.96 |
| Mazda | 6 | 600 | 4.26 |
| Ford | 8 | 591 | 4.20 |
| Geely Galaxy | 8 | 575 | 4.23 |
| Li Auto | 6 | 541 | 4.57 |
| Skoda | 5 | 500 | 3.87 |
| Haval | 5 | 500 | 4.28 |
| IM Motors | 6 | 499 | 4.44 |
| Venucia | 5 | 482 | 3.91 |
| Jetour | 5 | 474 | 4.06 |
| Roewe | 6 | 457 | 4.12 |
| Volvo | 6 | 436 | 4.29 |
| Chevrolet | 5 | 431 | 4.02 |
| Hyundai | 5 | 404 | 4.12 |
| Lincoln | 4 | 400 | 4.37 |
| Peugeot | 4 | 400 | 4.20 |
| Jetta | 5 | 332 | 3.86 |
| Dongfeng Aeolus | 4 | 326 | 3.98 |
| Dongfeng Forthing | 6 | 311 | 3.88 |
| Fangchengbao | 3 | 300 | 4.47 |
| Bestune | 5 | 287 | 3.86 |
| Dongfeng Fengguang | 5 | 263 | 3.61 |
| Ledo | 3 | 239 | 4.45 |
| ORA | 3 | 234 | 4.13 |
| Tank | 3 | 229 | 4.35 |
| smart | 3 | 207 | 4.23 |
| Livan | 2 | 200 | 4.06 |
| Citroën | 2 | 200 | 4.22 |
| Tesla | 2 | 200 | 4.24 |
| Feifan | 2 | 200 | 4.27 |
| Infiniti | 2 | 200 | 4.14 |
| Arcfox | 2 | 200 | 4.26 |
| Xiaomi | 2 | 200 | 4.53 |
| Wuling | 6 | 191 | 3.59 |
| Maxus | 5 | 187 | 3.97 |
| Polestar | 3 | 171 | 4.06 |
| Geely Geometry | 3 | 158 | 3.90 |
| Cowin | 3 | 153 | 3.80 |
| Jaguar | 2 | 145 | 4.25 |
| Mengshi | 2 | 140 | 4.38 |
| Beijing | 2 | 122 | 4.08 |
| Yangwang | 2 | 109 | 4.50 |
| Sihao | 3 | 106 | 3.70 |
| Zongheng | 1 | 100 | 4.29 |
| Land Rover | 1 | 100 | 4.07 |
| Denza | 1 | 100 | 4.38 |
| Changan Oshan | 1 | 100 | 4.15 |
| iCAR | 1 | 100 | 4.47 |
| Firefly | 1 | 95 | 4.30 |
| Audi (AUDI) | 2 | 93 | 4.42 |
| Huajing | 1 | 90 | 4.58 |
| JMEV | 4 | 70 | 3.84 |
| Levc | 5 | 70 | 4.10 |
| JAC Yiwei | 2 | 62 | 3.94 |
| JAC Refine | 2 | 55 | 3.74 |
| Shijie | 1 | 43 | 3.96 |
| Unknown | 1 | 42 | 4.51 |
| Changan Kaicheng | 2 | 36 | 3.50 |
| MINI | 2 | 33 | 4.14 |
| Haima | 1 | 32 | 4.00 |
| Dongfeng Fengdu | 1 | 23 | 4.13 |
| Zhidou | 1 | 22 | 4.15 |
| Lingbox | 2 | 15 | 3.08 |
| Caocao | 1 | 9 | 3.82 |
| Ruichi | 1 | 9 | 3.72 |
| Dayun | 2 | 9 | 3.59 |
| Skyworth | 1 | 8 | 3.75 |
| Aishang | 1 | 5 | 4.36 |
| Xiaohu | 1 | 5 | 3.42 |
| Foton | 1 | 4 | 2.72 |
| Lingxi | 1 | 3 | 3.83 |
| Dongfeng Fukang | 1 | 1 | 3.50 |

## 9. Dashboard Data Bridge

The Stage 6 web dashboard (`app/`) does not produce raw data itself. It reads the CSV files described above and pre-bakes them into JSON data bridges via `app/build_dashboard_data.py` for the frontend ECharts to render:

- Inputs: artifacts under `data/processed/stage3/`, `data/processed/stage4/`, `data/processed/stage5/`, and `data/sentiment/`.
- Outputs: `app/static/data/*.json` (overview, forecast, absa, attribution, relation, alerts, drilldown).
- Run: `python app/build_dashboard_data.py` to regenerate; `python app/app.py` to launch the dashboard.

These JSON files are pre-baked and committed with the code, so the dashboard works locally or via GitHub Pages static deployment without re-running the full data pipeline.

