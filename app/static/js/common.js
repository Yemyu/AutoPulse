// AutoPulse 看板通用前端助手
function loadJSON(name) {
  return fetch('/static/data/' + name + '.json').then(function (r) { return r.json(); });
}

function initChart(id) {
  var el = document.getElementById(id);
  var c = echarts.init(el, null, { renderer: 'canvas' });
  window.addEventListener('resize', function () { c.resize(); });
  return c;
}

function baseGrid(extra) {
  extra = extra || {};
  return Object.assign({
    left: 72, right: 36, top: 64, bottom: 68, containLabel: true
  }, extra);
}

function axisStyle() {
  return {
    axisLine: { lineStyle: { color: window.AUTOPULSE.axisLine } },
    axisLabel: { color: window.AUTOPULSE.muted },
    splitLine: { lineStyle: { color: window.AUTOPULSE.splitLine } }
  };
}

function titleStyle(text) {
  // 看板统一用卡片 <h3> 作为标题，不再让 ECharts 内部重复画标题，避免中英文切换时双重标题重叠。
  return { show: false };
}

function tooltipStyle() {
  return {
    trigger: 'axis', backgroundColor: '#fff', borderColor: '#e3e8ee',
    textStyle: { color: '#1f2d3d' }
  };
}

function labelLang(d) {
  return window.AUTOPULSE.lang === 'en' ? (d.name_en || d.name || d.name_zh) : (d.name_zh || d.name || d.name_en);
}

function nameFromAspect(d) {
  if (window.AUTOPULSE.lang === 'en') return d.name_en || d.name || d.name_zh;
  return d.name_zh || d.name || d.name_en;
}

// 辅助：给所有图表绑定语言切换重绘（页面自己存 render 函数到 window.__renderers）
document.addEventListener('i18nChanged', function () {
  if (window.__renderers) {
    Object.values(window.__renderers).forEach(function (fn) { try { fn(); } catch (e) {} });
  }
});
window.__renderers = window.__renderers || {};
