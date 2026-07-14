// AutoPulse 中英双语字典与切换器
// 所有静态文案集中在此，模板元素用 data-i18n="key" 标记即可。
window.I18N = (function () {
  var DICT = {
    // nav / topbar
    "nav_overview": { zh: "项目概览", en: "Overview" },
    "nav_forecast": { zh: "销量预测", en: "Forecast" },
    "nav_absa": { zh: "舆情 ABSA", en: "ABSA" },
    "nav_attribution": { zh: "销量归因", en: "Attribution" },
    "nav_relation": { zh: "舆情↔销量", en: "Sentiment↔Sales" },
    "nav_alerts": { zh: "舆情预警", en: "Alerts" },
    "nav_drilldown": { zh: "钻取分析", en: "Drilldown" },
    "nav_group_analysis": { zh: "分析", en: "Analysis" },
    "nav_group_relation": { zh: "关系", en: "Relation" },
    "nav_group_drilldown": { zh: "钻取", en: "Drilldown" },
    "brand_sub": { zh: "汽车销量预测与用户舆情分析", en: "Automotive Sales Forecasting & User Sentiment Analysis" },
    "lang_btn": { zh: "English", en: "中文" },

    // overview
    "ov_title": { zh: "项目概览", en: "Project Overview" },
    "ov_lead": { zh: "AutoPulse 将汽车月度销量数据与懂车帝用户口碑进行对齐，系统评估舆情对销量预测与归因的价值。", en: "AutoPulse aligns monthly sales data with user reviews from Dongchedi to systematically evaluate the value of sentiment for sales forecasting and attribution." },
    "ov_stage_flow": { zh: "阶段流程", en: "Stage Pipeline" },
    "ov_key_findings": { zh: "核心发现", en: "Key Findings" },
    "ov_kpi_coverage": { zh: "覆盖车系", en: "Series Covered" },
    "ov_kpi_forecast": { zh: "预测 WMAPE", en: "Forecast WMAPE" },
    "ov_kpi_horizon": { zh: "预测区间", en: "Forecast Horizon" },
    "ov_kpi_alerts": { zh: "舆情预警", en: "Sentiment Alerts" },
    "ov_kpi_brands": { zh: "覆盖品牌", en: "Brands Covered" },
    "ov_kpi_reviews": { zh: "评论样本", en: "Review Samples" },
    "ov_unit_series": { zh: "车系", en: " series" },
    "ov_unit_horizon": { zh: "月", en: " months" },
    "ov_unit_alert": { zh: "条", en: " alerts" },
    "ov_unit_brand": { zh: "个", en: " brands" },
    "ov_unit_review": { zh: "条", en: " reviews" },
    "ov_monthly_trend": { zh: "全市场月度销量走势", en: "Market Monthly Sales Trend" },

    // forecast
    "fc_title": { zh: "销量预测", en: "Sales Forecasting" },
    "fc_lead": { zh: "销量预测回答车企的核心问题：下个月能卖多少？对比多种模型后发现，历史销量惯性是最强信号，而月度舆情作为动态特征的增益有限。", en: "Sales forecasting estimates next month's vehicle sales. Across models, historical sales momentum is the strongest signal, while monthly sentiment provides limited lift." },
    "fc_model_cmp": { zh: "模型对比（体积加权 WMAPE，越低越好）", en: "Model Comparison (Volume-Weighted WMAPE, Lower is Better)" },
    "fc_example": { zh: "预测示例：", en: "Prediction Example: " },
    "fc_example_sub": { zh: "（实际 vs 预测）", en: " (Actual vs Predicted)" },
    "fc_features": { zh: "特征重要性（Top 12）", en: "Feature Importance (Top 12)" },
    "fc_scatter": { zh: "预测 vs 实际散点图", en: "Predicted vs Actual Scatter" },
    "fc_scatter_insight": { zh: "每个点代表一个车系月的预测值与实际值。点越靠近对角线，模型预测越准确；偏离对角线的点是模型预测误差较大的样本。", en: "Each point is one series-month prediction vs actual. Points near the diagonal indicate accurate predictions; outliers are high-error samples." },
    "fc_class_wmape": { zh: "按车型级别的预测误差对比", en: "Forecast Error by Vehicle Class" },
    "fc_class_insight": { zh: "对比 SUV、轿车、MPV 三类车型的体积加权 WMAPE，评估模型在不同车型级别上的预测表现差异。", en: "Compares volume-weighted WMAPE across SUV, Sedan, and MPV classes to assess model performance by vehicle type." },
    "fc_error_dist": { zh: "预测误差分布（MAPE 箱线统计）", en: "Forecast Error Distribution (MAPE Box Stats)" },
    "fc_source_mc": { zh: "数据来源：data/processed/stage3/model_comparison.csv", en: "Source: data/processed/stage3/model_comparison.csv" },
    "fc_source_preds": { zh: "数据来源：data/processed/stage3/xgboost_preds.csv", en: "Source: data/processed/stage3/xgboost_preds.csv" },
    "fc_source_abl": { zh: "数据来源：data/processed/stage3/xgb_ablation.csv", en: "Source: data/processed/stage3/xgb_ablation.csv" },
    "fc_axis_wmape": { zh: "WMAPE %", en: "WMAPE %" },
    "fc_axis_mape": { zh: "MAPE", en: "MAPE" },
    "fc_legend_actual": { zh: "实际", en: "Actual" },
    "fc_legend_pred": { zh: "预测", en: "Predicted" },
    "fc_axis_sales": { zh: "销量", en: "Sales" },
    "fc_example_insight": { zh: "通过实际销量与预测销量的对比，判断模型对季节性与拐点的捕捉能力。", en: "Compare actual and predicted sales to assess the model's ability to capture seasonality and turning points." },
    "fc_error_insight": { zh: "误差分布反映模型在大部分车系上的稳定性，以及少数极端车系对整体指标的影响。", en: "Error distribution reflects model stability across most series and the impact of a few outliers on overall metrics." },
    "fc_feature_insight": { zh: "前两位均为销量自身滞后与移动平均，说明预测以历史惯性为主；能源类型、车辆级别等属性提供车型截面差异；舆情特征排名靠后，动态口碑不是主要预测信号。", en: "Top features are sales lags and moving averages, confirming the forecast relies on historical momentum. Energy type and vehicle class provide cross-sectional differences. Sentiment features rank low, indicating dynamic word-of-mouth is not a main predictor." },

    // absa
    "ab_title": { zh: "舆情 ABSA", en: "Sentiment ABSA" },
    "ab_lead": { zh: "ABSA（Aspect-Based Sentiment Analysis，基于维度的情感分析）将车主口碑拆分为 10 个评价维度（外观、内饰、空间、动力、操控、舒适、油耗、配置、智能化、性价比），分别计算情感分（−1~1），用于识别消费者在各维度上的正面与负面反馈。", en: "ABSA (Aspect-Based Sentiment Analysis) breaks down owner reviews into 10 evaluation dimensions (Appearance, Interior, Space, Power, Control, Comfort, Fuel, Config, Intelligence, Value), computing sentiment scores (−1~1) per dimension to identify positive and negative feedback." },
    "ab_trend": { zh: "十维度月度情感走势", en: "10-Dimension Monthly Sentiment Trends" },
    "ab_trend_insight": { zh: "折线展示各维度情感分随时间的变化，可观察哪些维度的口碑在改善、哪些在恶化。", en: "Lines show each dimension's sentiment score over time, revealing which aspects are improving or deteriorating." },
    "ab_bar": { zh: "各维度平均情感分", en: "Average Sentiment by Dimension" },
    "ab_dist": { zh: "情感正负分布（正/中/负）", en: "Positive / Neutral / Negative Distribution" },
    "ab_variance": { zh: "各维度情感波动（标准差）", en: "Sentiment Volatility by Dimension (Std Dev)" },
    "ab_source": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly.csv", en: "Source: data/processed/stage4/sentiment_sales_monthly.csv" },
    "ab_axis_sentiment": { zh: "情感分", en: "Sentiment Score" },
    "ab_radar_insight": { zh: "雷达图将 10 个维度放在同一坐标系下，用于比较各维度的满意度高低。", en: "The radar chart places all 10 dimensions on one coordinate system to compare satisfaction levels." },
    "ab_bar_insight": { zh: "平均分落在 −1 到 +1 之间，高于 0 表示整体正面，低于 0 表示负面反馈集中。", en: "Average scores range from −1 to +1; above 0 is generally positive, below 0 indicates concentrated negative feedback." },
    "ab_source_trend": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly.csv（按月聚合）", en: "Source: data/processed/stage4/sentiment_sales_monthly.csv (monthly aggregation)" },
    "ab_dist_insight": { zh: "堆叠图展示每个维度下正面、中性、负面评论的占比，用于识别情绪两极分化程度。", en: "The stacked chart shows the share of positive, neutral, and negative reviews per dimension to identify polarization." },
    "ab_var_insight": { zh: "标准差越大，说明该维度上用户评价的分歧越大。", en: "Higher standard deviation indicates greater disagreement among owners on that dimension." },
    "ab_pos": { zh: "正面", en: "Positive" },
    "ab_neu": { zh: "中性", en: "Neutral" },
    "ab_neg": { zh: "负面", en: "Negative" },

    // attribution
    "at_title": { zh: "销量归因", en: "Sales Attribution" },
    "at_lead": { zh: "在了解口碑分数后，需要进一步判断这些分数是否真正影响购车决策。用车系级 XGBoost+SHAP 将销量变化拆解到各舆情维度，量化各维度对销量解释力的贡献。", en: "After obtaining sentiment scores, the next step is to judge whether they actually affect purchase decisions. Series-level XGBoost+SHAP decomposes sales variation into each sentiment dimension and quantifies its contribution." },
    "at_shap": { zh: "舆情维度对销量的 SHAP 重要性", en: "SHAP Importance of Sentiment Dimensions" },
    "at_cmp_title": { zh: "加入舆情特征后的归因提升", en: "Attribution Lift After Adding Sentiment Features" },
    "at_example": { zh: "归因示例：预测误差最小的车系", en: "Attribution Example: Series with Smallest Prediction Error" },
    "at_source_shap": { zh: "数据来源：data/processed/stage4/aspect_shap_ranking.csv", en: "Source: data/processed/stage4/aspect_shap_ranking.csv" },
    "at_source_metr": { zh: "数据来源：data/processed/stage4/attribution_metrics.csv", en: "Source: data/processed/stage4/attribution_metrics.csv" },
    "at_shap_insight": { zh: "SHAP 值反映该维度对销量预测准确度的贡献，条形越长代表解释力越强。", en: "SHAP values reflect each dimension's contribution to sales prediction accuracy; longer bars indicate stronger explanatory power." },
    "at_cmp_insight": { zh: "对比含舆情与不含舆情两套模型的 R²/MAPE，量化舆情特征对销量归因的真实贡献。", en: "Compare models with and without sentiment features to quantify sentiment's real contribution to sales attribution." },
    "at_example_insight": { zh: "展示模型拟合最优的车系示例：绿色线为实际销量，蓝色虚线为预测，红色柱为绝对误差。", en: "Best-fit series example: green line = actual sales, blue dashed line = predicted, red bars = absolute error." },
    "at_label_r2": { zh: "R²（决定系数）", en: "R²" },
    "at_label_mape": { zh: "MAPE（平均绝对百分比误差）", en: "MAPE" },
    "at_better": { zh: "✓ 舆情带来提升", en: "✓ Sentiment improves" },
    "at_worse": { zh: "✗ 未改善", en: "✗ No improvement" },
    "at_label_without": { zh: "不含舆情", en: "Without sentiment" },
    "at_label_with": { zh: "含舆情", en: "With sentiment" },
    "at_legend_actual": { zh: "实际", en: "Actual" },
    "at_legend_pred": { zh: "预测", en: "Predicted" },
    "at_legend_error": { zh: "绝对误差", en: "Absolute Error" },

    // relation
    "rl_title": { zh: "舆情↔销量", en: "Sentiment ↔ Sales" },
    "rl_lead": { zh: "本项目核心问题：舆情能否预测销量？Granger 因果、情感融合预测与双轴时序共同表明，舆情更接近销量的同期或滞后指标，而非先行指标。", en: "Core question: can sentiment predict sales? Granger causality, fusion experiments, and dual-axis series all indicate sentiment is a coincident or lagging indicator, not a leading one." },
    "rl_granger": { zh: "品牌级 Granger 因果显著率（各维度）", en: "Brand-Level Granger Causality Significance Rate" },
    "rl_fusion": { zh: "把月度情感塞回销量模型：预测误差变化", en: "Adding Monthly Sentiment to Sales Model: Error Change" },
    "rl_ts": { zh: "市场级销量 ↔ 综合口碑时序", en: "Market-Level Sales vs Overall Sentiment" },
    "rl_corr": { zh: "各维度情感与销量的相关系数", en: "Correlation Between Dimension Sentiment and Sales" },
    "rl_source_granger": { zh: "数据来源：data/processed/stage4/granger_brand_summary.csv", en: "Source: data/processed/stage4/granger_brand_summary.csv" },
    "rl_source_fusion": { zh: "数据来源：data/processed/stage5/forecast_comparison.csv", en: "Source: data/processed/stage5/forecast_comparison.csv" },
    "rl_source_ts": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly_brand.csv", en: "Source: data/processed/stage4/sentiment_sales_monthly_brand.csv" },
    "rl_source_corr": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly.csv", en: "Source: data/processed/stage4/sentiment_sales_monthly.csv" },
    "rl_axis_sig_rate": { zh: "显著率", en: "Significance Rate" },
    "rl_axis_wmape": { zh: "WMAPE %", en: "WMAPE %" },
    "rl_axis_corr": { zh: "皮尔逊相关系数", en: "Pearson Correlation" },
    "rl_legend_sales": { zh: "月度销量", en: "Monthly Sales" },
    "rl_ts_insight": { zh: "双轴时序将全市场月度总销量与平均口碑叠加，用于判断两者同步变动还是存在领先-滞后关系。品牌级时序可在「钻取分析」中查看。", en: "Dual-axis series overlays total market sales and average sentiment to judge whether they move together or have a lead-lag relationship. Brand-level series available in Drilldown." },
    "rl_legend_sentiment": { zh: "综合口碑", en: "Overall Sentiment" },

    // alerts
    "al_title": { zh: "舆情预警", en: "Sentiment Alerts" },
    "al_lead": { zh: "由于舆情更适合监测而非预测，将其转化为声誉风险预警：识别口碑正在恶化、需要运营侧关注的车系。", en: "Since sentiment is more suitable for monitoring than forecasting, it is converted into reputation-risk alerts: identifying series whose word-of-mouth is deteriorating and require attention." },
    "al_rule": { zh: "预警规则", en: "Alert Rule" },
    "al_trend": { zh: "预警月度趋势", en: "Monthly Alert Trend" },
    "al_risk": { zh: "风险等级分布", en: "Risk Level Distribution" },
    "al_list": { zh: "预警车系列表", en: "Alert Series List" },
    "al_source": { zh: "数据来源：data/processed/stage5/sentiment_alerts.csv", en: "Source: data/processed/stage5/sentiment_alerts.csv" },
    "al_tbl_series": { zh: "车系", en: "Series" },
    "al_tbl_brand": { zh: "品牌", en: "Brand" },
    "al_tbl_period": { zh: "月份", en: "Period" },
    "al_tbl_overall": { zh: "综合情感", en: "Overall" },
    "al_tbl_drop": { zh: "环比下降", en: "MoM Drop" },
    "al_risk_high": { zh: "高危", en: "High" },
    "al_risk_mid": { zh: "中危", en: "Medium" },
    "al_risk_low": { zh: "低危", en: "Low" },
    "al_axis_count": { zh: "预警条数", en: "Alert Count" },
    "al_rule_insight": { zh: "舆情作为声誉健康度的监测入口，规则聚焦：综合口碑已为负面且环比继续下降。", en: "Sentiment serves as a reputation-health monitor. The rule focuses on cases where overall sentiment is already negative and keeps worsening month-over-month." },
    "al_trend_insight": { zh: "趋势图用于识别口碑危机集中爆发的月份，区分偶发波动与系统性恶化。", en: "The trend chart identifies months when reputation crises cluster, distinguishing one-off spikes from systematic deterioration." },
    "al_risk_insight": { zh: "按综合情感分将预警划分为高、中、低三档，便于按风险等级排序处理。", en: "Alerts are classified into high, medium, and low risk by overall sentiment to enable prioritized handling." },
    "al_list_insight": { zh: "明细列出触发规则的车系、月份与情感变化，供运营侧跟进。", en: "Detailed list of series, month, and sentiment change for operations follow-up." },

    // drilldown
    "dd_title": { zh: "钻取分析", en: "Drilldown Analysis" },
    "dd_lead": { zh: "从宏观到微观两层钻取：品牌钻取展示品牌级销量/口碑走势与维度画像（数据连续，完成度高）；车型钻取下探到单个车系的十维度口碑热力图。点击品牌排名中的车系可直接跳转至车型钻取。", en: "Two-level drilldown from macro to micro: Brand tab shows brand-level sales/sentiment trends and aspect profiles (near-continuous data); Series tab drills into individual series with 10-dimension sentiment heatmaps. Click a series in the brand ranking to jump directly to the series tab." },
    "dd_tab_brand": { zh: "品牌钻取", en: "Brand" },
    "dd_tab_series": { zh: "车型钻取", en: "Series" },
    "dd_select_brand": { zh: "品牌筛选：", en: "Brand: " },
    "dd_select_series": { zh: "选择车系：", en: "Series: " },
    "dd_search": { zh: "搜索车系...", en: "Search series..." },
    "dd_meta": { zh: "共 {n} 个月", en: "{n} months" },
    "dd_empty": { zh: "该车系暂缺对齐数据", en: "No aligned data for this series" },
    "dd_sales_sentiment": { zh: "月度销量 vs 综合口碑", en: "Monthly Sales vs Overall Sentiment" },
    "dd_aspects": { zh: "十维度情感走势", en: "10-Dimension Sentiment Trends" },
    "dd_source": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly.csv（已对齐的口碑+销量）", en: "Source: data/processed/stage4/sentiment_sales_monthly.csv (aligned reviews + sales)" },
    "dd_all_brands": { zh: "全部品牌", en: "All brands" },
    "dd_heatmap_positive": { zh: "正面", en: "Positive" },
    "dd_heatmap_negative": { zh: "负面", en: "Negative" },
    "dd_legend_sales": { zh: "月度销量", en: "Monthly Sales" },
    "dd_legend_sentiment": { zh: "综合口碑", en: "Overall Sentiment" },
    "dd_insight": { zh: "选择一个品牌与车系，查看该车系 2022-2026 年逐月销量与十维度口碑走势，用于观察销量波动前各维度口碑的先行变化。", en: "Select a brand and series to see monthly sales and 10-dimension sentiment from 2022-2026, to observe the lead-lag relationship between sentiment changes and sales fluctuations." },
    "dd_sales_insight": { zh: "左轴为销量，右轴为综合口碑。两者同步变动说明口碑是销量的同期指标；若销量下滑而口碑平稳，则销量另有驱动因素。", en: "Left axis = sales, right axis = overall sentiment. Synchronous movement means sentiment is a coincident indicator; if sales drop while sentiment stays flat, sales has other drivers." },
    "dd_aspects_insight": { zh: "热力图颜色越红代表该维度越负面，越绿代表越正面。用于识别销量波动前口碑率先转负的维度。", en: "Heatmap: redder = more negative, greener = more positive. Use it to identify dimensions where sentiment deteriorated ahead of sales fluctuations." },
    "dd_axis_sentiment": { zh: "口碑", en: "Sentiment" },
    "dd_sparsity_note": { zh: "注：热力图中空白格表示当月无足够口碑样本（评论数为0或ABSA聚合后样本过少），不代表系统异常。", en: "Note: blank cells in the heatmap indicate months with insufficient review samples (zero reviews or too few for ABSA aggregation), not a system error." },
    "dd_filtered": { zh: "已筛选：仅保留销量与口碑数据均完整的车系", en: "Filtered: only series with complete sales and sentiment data are shown" },
    "dd_not_in_filtered": { zh: "该车系口碑样本不足，未纳入车型钻取", en: "This series has insufficient sentiment samples and is not included in series drilldown" },
    "dd_click_hint": { zh: "提示：点击柱子可跳转到该车型的钻取详情", en: "Tip: click a bar to drill into that series" },

    // brand drilldown
    "nav_brand_drilldown": { zh: "品牌钻取", en: "Brand Drilldown" },
    "bd_title": { zh: "品牌钻取", en: "Brand Drilldown" },
    "bd_lead": { zh: "从宏观视角查看单一品牌的销量与口碑整体走势：品牌级时序、维度画像，以及旗下车系的贡献结构。品牌级聚合后舆情数据基本连续，完成度高于车型级。", en: "A macro view of a single brand's sales and sentiment: brand-level time series, aspect profile, and the contribution structure of its series. Brand-level aggregation yields near-continuous sentiment data with higher completeness than the series level." },
    "bd_select_brand": { zh: "选择品牌：", en: "Brand: " },
    "bd_all_brands": { zh: "全部品牌", en: "All brands" },
    "bd_insight": { zh: "选择一个品牌，查看其月度销量与综合口碑走势、十维度口碑画像，以及旗下主力车系的销量与口碑排名。", en: "Select a brand to see its monthly sales vs overall sentiment, 10-dimension sentiment profile, and the sales/sentiment ranking of its flagship series." },
    "bd_sales_sentiment": { zh: "品牌月度销量 vs 综合口碑", en: "Brand Monthly Sales vs Overall Sentiment" },
    "bd_sales_insight": { zh: "左轴为销量，右轴为综合口碑（十维度均值）。品牌级样本更多，时序基本连续；若销量与口碑背离，说明品牌当期表现由非舆情因素驱动。", en: "Left axis = sales, right axis = overall sentiment (mean of 10 aspects). Brand-level aggregation gives near-continuous series; a divergence suggests non-sentiment drivers are at play." },
    "bd_legend_sales": { zh: "月度销量", en: "Monthly Sales" },
    "bd_legend_sentiment": { zh: "综合口碑", en: "Overall Sentiment" },
    "bd_radar": { zh: "品牌维度画像 vs 市场平均", en: "Brand Profile vs Market Average" },
    "bd_radar_insight": { zh: "雷达图展示该品牌十维度平均口碑，灰色虚线为全市场品牌均值。外扩维度是品牌口碑优势，内缩维度是短板。", en: "Radar shows the brand's 10-dimension average sentiment; the grey dashed line is the all-market brand average. Outward dimensions are strengths, inward are weaknesses." },
    "bd_radar_brand": { zh: "本品牌", en: "This brand" },
    "bd_radar_market": { zh: "市场平均", en: "Market avg" },
    "bd_rank": { zh: "旗下主力车系排名（按平均月销量）", en: "Flagship Series Ranking (by avg monthly sales)" },
    "bd_rank_insight": { zh: "按平均月销量排序的 Top 15 车系，柱高为平均月销量，标签为该车系平均综合口碑。可识别品牌的销量支柱与口碑洼地。", en: "Top 15 series by average monthly sales; bar height = avg monthly sales, label = avg overall sentiment. Identifies the brand's sales pillars and sentiment lows." },
    "bd_rank_axis_sales": { zh: "平均月销量", en: "Avg monthly sales" },
    "bd_rank_axis_sent": { zh: "平均综合口碑", en: "Avg sentiment" },
    "bd_rank_empty": { zh: "该品牌暂缺车型级对齐数据", en: "No model-level aligned data for this brand" },
    "bd_source_brand": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly_brand.csv", en: "Source: data/processed/stage4/sentiment_sales_monthly_brand.csv" },
    "bd_source_model": { zh: "数据来源：data/processed/stage4/sentiment_sales_monthly.csv", en: "Source: data/processed/stage4/sentiment_sales_monthly.csv" },

    // footer / source
    "source_prefix": { zh: "数据来源：", en: "Source: " },
    "conclusion_prefix": { zh: "结论：", en: "Conclusion: " }
  };

  function get(key, lang) {
    if (!key) return key;
    lang = lang || localStorage.getItem("autopulse_lang") || "zh";
    if (DICT[key] && DICT[key][lang]) return DICT[key][lang];
    return key;
  }

  function apply(lang) {
    if (lang) {
      localStorage.setItem("autopulse_lang", lang);
    } else {
      lang = localStorage.getItem("autopulse_lang") || "zh";
    }
    window.AUTOPULSE = window.AUTOPULSE || {};
    window.AUTOPULSE.lang = lang;
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    // 切换 data-i18n 元素
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var k = el.getAttribute("data-i18n");
      var text = get(k, lang);
      if (el.tagName === "INPUT" && el.hasAttribute("placeholder")) el.setAttribute("placeholder", text);
      else el.textContent = text;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
      var k = el.getAttribute("data-i18n-placeholder");
      el.setAttribute("placeholder", get(k, lang));
    });
    // 切换按钮文字
    var btn = document.getElementById("langSwitch");
    if (btn) btn.textContent = lang === "zh" ? "English" : "中文";
    // 触发语言切换事件，供各页图表重绘
    document.dispatchEvent(new Event("i18nChanged"));
  }

  function toggle() {
    var cur = localStorage.getItem("autopulse_lang") || "zh";
    apply(cur === "zh" ? "en" : "zh");
  }

  return { get: get, apply: apply, toggle: toggle };
})();

document.addEventListener("DOMContentLoaded", function () {
  var btn = document.getElementById("langSwitch");
  if (btn) btn.addEventListener("click", window.I18N.toggle);
  window.I18N.apply();
});
