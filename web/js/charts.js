/* ============================================================================
 * charts.js —— 轻量 SVG 图表（零依赖，断网可用）
 *   提供：折线图（含外推虚线）、仪表盘、横向条形图、历史趋势图。
 * ========================================================================== */
(function (global) {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  /** 数值格式化：去掉多余的浮点尾巴 */
  function fmt(v, digits) {
    if (v == null || Number.isNaN(v)) return "—";
    const d = digits == null ? 2 : digits;
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    const r = Math.round(n * Math.pow(10, d)) / Math.pow(10, d);
    return String(r);
  }

  /** 自动选择"漂亮"的刻度步长 */
  function niceStep(span, targetTicks) {
    const raw = span / (targetTicks || 5);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    let step;
    if (norm < 1.5) step = 1;
    else if (norm < 3) step = 2;
    else if (norm < 7) step = 5;
    else step = 10;
    return step * mag;
  }

  /**
   * 折线图（通用）
   * opts: {
   *   series: [{ name, color, points: [{x, y}], dash?: bool, dashSegments?: [{x0,y0,x1,y1}] }]
   *   width, height, pad, xLabels?, yFormat?, threshold?: {value, color, label}
   * }
   * x 与 y 均可为任意数值（按数值坐标渲染）。
   */
  function lineChart(container, opts) {
    const W = opts.width || 720;
    const H = opts.height || 300;
    const padL = opts.padL != null ? opts.padL : 52;
    const padR = 18, padT = 18, padB = 34;

    // 收集所有点以确定范围
    let xs = [], ys = [];
    (opts.series || []).forEach((s) => {
      (s.points || []).forEach((p) => { xs.push(p.x); ys.push(p.y); });
    });
    if (opts.threshold != null && opts.threshold.value != null) ys.push(opts.threshold.value);

    if (xs.length === 0) {
      container.innerHTML = '<div class="empty-state"><span class="big">📉</span>暂无数据可绘图</div>';
      return;
    }

    let xMin = opts.xMin != null ? opts.xMin : Math.min.apply(null, xs);
    let xMax = opts.xMax != null ? opts.xMax : Math.max.apply(null, xs);
    let yMin = opts.yMin != null ? opts.yMin : Math.min.apply(null, ys);
    let yMax = opts.yMax != null ? opts.yMax : Math.max.apply(null, ys);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (opts.yMin != null) yMin = opts.yMin;
    if (opts.yMax != null) yMax = opts.yMax;

    const innerW = W - padL - padR;
    const innerH = H - padT - padB;
    const sx = (x) => padL + ((x - xMin) / (xMax - xMin)) * innerW;
    const sy = (y) => padT + ((yMax - y) / (yMax - yMin)) * innerH;

    // 刻度
    const yStep = niceStep(yMax - yMin, 5);
    let grid = [], gy = [], i;
    const y0 = Math.ceil(yMin / yStep) * yStep;
    for (i = y0; i <= yMax + 1e-9; i += yStep) {
      gy.push(i);
      grid.push(`<line x1="${padL}" y1="${sy(i).toFixed(1)}" x2="${W - padR}" y2="${sy(i).toFixed(1)}" stroke="#eef1f4" stroke-width="1"/>`);
      grid.push(`<text x="${padL - 8}" y="${(sy(i) + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="#9aa3af">${esc(fmt(i, 2))}</text>`);
    }

    // X 轴标签（最多约 8 个）
    const xTickN = Math.min(8, xs.length);
    let xticks = [];
    if (xTickN > 1) {
      for (let k = 0; k < xTickN; k++) {
        const v = xMin + ((xMax - xMin) / (xTickN - 1)) * k;
        xticks.push(v);
      }
    } else if (xs.length === 1) {
      xticks = [xs[0]];
    }

    const path = (pts, closed) => {
      if (!pts || pts.length === 0) return "";
      let d = `M ${sx(pts[0].x).toFixed(1)} ${sy(pts[0].y).toFixed(1)}`;
      for (let k = 1; k < pts.length; k++) d += ` L ${sx(pts[k].x).toFixed(1)} ${sy(pts[k].y).toFixed(1)}`;
      return d;
    };

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
    svg += grid.join("");
    xticks.forEach((v) => {
      svg += `<text x="${sx(v).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#9aa3af">${esc(opts.xLabel ? opts.xLabel(v) : fmt(v, 1))}</text>`;
    });

    // 阈值线
    if (opts.threshold != null && opts.threshold.value != null) {
      svg += `<line x1="${padL}" y1="${sy(opts.threshold.value).toFixed(1)}" x2="${W - padR}" y2="${sy(opts.threshold.value).toFixed(1)}" stroke="${opts.threshold.color || '#c0392b'}" stroke-width="1.5" stroke-dasharray="6 4"/>`;
      svg += `<text x="${W - padR}" y="${(sy(opts.threshold.value) - 5).toFixed(1)}" text-anchor="end" font-size="10" fill="${opts.threshold.color || '#c0392b'}">${esc(opts.threshold.label || '')}</text>`;
    }

    // 系列线
    (opts.series || []).forEach((s) => {
      const main = (s.points || []).filter((p) => !p.extrapolated);
      const extra = (s.points || []).filter((p) => p.extrapolated);
      const dash = s.dash;
      const color = s.color || "#0e7c86";
      if (main.length) {
        svg += `<path d="${path(main)}" fill="none" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="5 4"' : ""}/>`;
        if (main.length > 0) {
          const last = main[main.length - 1];
          svg += `<circle cx="${sx(last.x).toFixed(1)}" cy="${sy(last.y).toFixed(1)}" r="3" fill="${color}"/>`;
        }
      }
      // 外推段：虚线，从最后一个实测点连到外推点
      if (extra.length && main.length) {
        const lastReal = main[main.length - 1];
        const pts = [lastReal].concat(extra);
        svg += `<path d="${path(pts)}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="5 5" opacity="0.75"/>`;
        const last = pts[pts.length - 1];
        svg += `<circle cx="${sx(last.x).toFixed(1)}" cy="${sy(last.y).toFixed(1)}" r="3" fill="${color}" opacity="0.6"/>`;
      }
    });

    svg += `</svg>`;

    // 图例
    let legend = "";
    if (opts.series && opts.series.length > 1) {
      legend = '<div class="chart-legend">' + opts.series.map((s) =>
        `<span class="lg"><span class="sw" style="background:${s.color || '#0e7c86'}"></span>${esc(s.name || '')}${s.dash ? '（外推）' : ''}</span>`
      ).join("") + '</div>';
    } else if (opts.series && opts.series.length === 1 && (opts.series[0].extrapolatedCount > 0)) {
      const s = opts.series[0];
      legend = `<div class="chart-legend"><span class="lg"><span class="sw" style="background:${s.color || '#0e7c86'}"></span>实测</span><span class="lg"><span class="sw" style="background:${s.color || '#0e7c86'};opacity:.6"></span>线性外推（≤12 小时）</span></div>`;
    }

    container.innerHTML = `<div class="chart-box">${svg}</div>${legend}`;
  }

  /** 仪表盘：value(0-100)，marks 刻度位置数组，riskLevel 决定配色 */
  function gauge(container, value, marks, riskLevel) {
    if (value == null || Number.isNaN(value)) {
      container.innerHTML = '<div class="empty-state">—</div>';
      return;
    }
    const W = 320, H = 190;
    const cx = W / 2, cy = 150, r = 120;
    const a0 = Math.PI, a1 = 2 * Math.PI; // 半圆从左到右
    const ang = (pct) => a0 + (a1 - a0) * Math.max(0, Math.min(100, pct)) / 100;
    const pt = (pct, rr) => {
      const a = ang(pct);
      return [cx + rr * Math.cos(a), cy + rr * Math.sin(a)];
    };

    // 三个色段：0-40 绿，40-70 橙，70-100 红
    const seg = (s, e, color) => {
      const [x1, y1] = pt(s, r - 26);
      const [x2, y2] = pt(e, r - 26);
      const large = (e - s) > 50 ? 1 : 0;
      return `<path d="M ${x1} ${y1} A ${r - 26} ${r - 26} 0 ${large} 1 ${x2} ${y2}" fill="none" stroke="${color}" stroke-width="22" stroke-linecap="butt"/>`;
    };

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
    svg += seg(0, 40, "#27ae60");
    svg += seg(40, 70, "#e67e22");
    svg += seg(70, 100, "#c0392b");

    // 刻度线 40 / 70（以及 0/100 端点）
    [0, 40, 70, 100].forEach((m) => {
      const [x1, y1] = pt(m, r - 38);
      const [x2, y2] = pt(m, r - 14);
      svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#9aa3af" stroke-width="1.5"/>`;
      svg += `<text x="${pt(m, r - 50)[0]}" y="${pt(m, r - 50)[1] + 4}" text-anchor="middle" font-size="10" fill="#9aa3af">${m}</text>`;
    });

    // 指针
    const v = Math.max(0, Math.min(100, value));
    const [nx, ny] = pt(v, r - 26);
    svg += `<line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#1f2937" stroke-width="3" stroke-linecap="round"/>`;
    svg += `<circle cx="${cx}" cy="${cy}" r="7" fill="#1f2937"/>`;

    // 中心文字
    const riskColor = { "正常": "#1e8449", "预警": "#d9822b", "高危": "#c0392b" }[riskLevel] || "#1f2937";
    svg += `<text x="${cx}" y="${cy - 34}" text-anchor="middle" font-size="26" font-weight="800" fill="${riskColor}" font-family="Consolas,Menlo,monospace">${esc(fmt(value, 1))}</text>`;
    svg += `<text x="${cx}" y="${cy - 14}" text-anchor="middle" font-size="11" fill="#9aa3af">风险分（0–100）</text>`;
    svg += `</svg>`;
    container.innerHTML = `<div class="chart-box">${svg}</div>`;
  }

  /** 横向条形图：items = [{ label, value(0-1), color? }] */
  function hbar(container, items, opts) {
    opts = opts || {};
    if (!items || items.length === 0) {
      container.innerHTML = '<div class="empty-state">—</div>';
      return;
    }
    const max = opts.max != null ? opts.max : 1;
    const rows = items.map((it) => {
      const pct = Math.max(0, Math.min(1, (it.value || 0) / max)) * 100;
      const color = it.color || "#0e7c86";
      const valText = opts.valueText ? opts.valueText(it.value) : fmt(it.value, 2);
      return `<div class="sev-row">
        <div class="sev-label" title="${esc(it.label || '')}">${esc(it.label || '—')}</div>
        <div class="sev-bar"><div class="sev-fill" style="width:${pct.toFixed(1)}%;background:${color}"></div></div>
        <div class="sev-tag" style="width:auto;min-width:52px">${esc(valText)}</div>
      </div>`;
    }).join("");
    container.innerHTML = rows;
  }

  /**
   * 历史风险趋势：scores = 按时间正序排列的 fused_score 数组（调用方已 reverse）
   * 纵轴 0–100，网格线 40/70；横轴为评估次序（历史 items 无时间戳字段）。
   */
  function historyTrend(container, scores) {
    if (!scores || scores.length === 0) {
      container.innerHTML = '<div class="empty-state"><span class="big">📊</span>还没有评估记录</div>';
      return;
    }
    const W = 720, H = 260, padL = 52, padR = 18, padT = 18, padB = 34;
    const innerW = W - padL - padR, innerH = H - padT - padB;
    const n = scores.length;
    const sx = (i) => padL + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
    const sy = (y) => padT + ((100 - y) / 100) * innerH;

    let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">`;
    [0, 20, 40, 60, 70, 80, 100].forEach((g) => {
      const hl = g === 40 || g === 70;
      svg += `<line x1="${padL}" y1="${sy(g).toFixed(1)}" x2="${W - padR}" y2="${sy(g).toFixed(1)}" stroke="${hl ? '#d9822b' : '#eef1f4'}" stroke-width="${hl ? 1.5 : 1}"/>`;
      svg += `<text x="${padL - 8}" y="${(sy(g) + 4).toFixed(1)}" text-anchor="end" font-size="10" fill="${hl ? '#d9822b' : '#9aa3af'}">${g}</text>`;
    });

    // X 轴序号
    const tickN = Math.min(6, n);
    for (let k = 0; k < tickN; k++) {
      const idx = Math.round((n - 1) * (k / (tickN - 1)));
      svg += `<text x="${sx(idx).toFixed(1)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#9aa3af">#${idx + 1}</text>`;
    }

    // 折线
    let d = "";
    scores.forEach((sc, k) => {
      d += `${k === 0 ? "M" : "L"} ${sx(k).toFixed(1)} ${sy(sc).toFixed(1)} `;
    });
    svg += `<path d="${d}" fill="none" stroke="#0e7c86" stroke-width="2"/>`;
    scores.forEach((sc, k) => {
      const c = sc >= 70 ? "#c0392b" : sc >= 40 ? "#d9822b" : "#1e8449";
      svg += `<circle cx="${sx(k).toFixed(1)}" cy="${sy(sc).toFixed(1)}" r="3.5" fill="${c}"/>`;
    });
    svg += `</svg>`;
    const legend = `<div class="chart-legend">
      <span class="lg"><span class="sw" style="background:#0e7c86"></span>fused_score 融合分（越早越靠左）</span>
      <span class="lg"><span class="sw" style="background:#d9822b"></span>40 / 70 预警·高危线</span>
    </div>`;
    container.innerHTML = `<div class="chart-box">${svg}</div>${legend}`;
  }

  function fmtTime(d) {
    if (!d || Number.isNaN(d.getTime())) return "—";
    const p = (n) => String(n).padStart(2, "0");
    return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.charts = {
    esc, fmt, lineChart, gauge, hbar, historyTrend, fmtTime,
  };
})(window);
