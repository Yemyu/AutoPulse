#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AutoPulse 看板服务（Flask + ECharts）。

页面只 render 模板；数据全部来自 app/static/data/*.json（由
build_dashboard_data.py 预烘焙生成）。前端 fetch JSON 后用 ECharts 渲染。
重新生成数据：python app/build_dashboard_data.py
启动服务：python app/app.py  ->  http://127.0.0.1:5001/
"""
from flask import Flask, render_template

app = Flask(__name__)

PAGES = [
    ("/", "index.html", "overview", "项目概览 · Overview"),
    ("/forecast", "forecast.html", "forecast", "销量预测 · Sales Forecasting"),
    ("/absa", "absa.html", "absa", "舆情 ABSA · Aspect Sentiment"),
    ("/attribution", "attribution.html", "attribution", "销量归因 · Attribution"),
    ("/relation", "relation.html", "relation", "舆情↔销量 · Relation"),
    ("/alerts", "alerts.html", "alerts", "舆情预警 · Alerts"),
    ("/drilldown", "drilldown.html", "drilldown", "钻取分析 · Drilldown"),
]


def make_view(route, template, key, title):
    def view():
        return render_template(template, title=title, active=key)
    view.__name__ = key
    return view


for _route, _tpl, _key, _title in PAGES:
    app.add_url_rule(_route, view_func=make_view(_route, _tpl, _key, _title))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
