/* ============================================================================
 * ui.js —— 通用渲染辅助 + 完整评估报告渲染器
 *   所有渲染函数都必须能吃 null / 缺字段，缺则显示 "—" 或隐藏区块，
 *   绝不输出 undefined / NaN / [object Object]。
 * ========================================================================== */
(function (global) {
  "use strict";

  const dict = global.Fisheries.dict;
  const charts = global.Fisheries.charts;
  const md = global.Fisheries.markdown;

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmt(v, digits) {
    if (v == null || v === "") return "—";
    if (typeof v === "number" && Number.isNaN(v)) return "—";
    const d = digits == null ? 2 : digits;
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    return String(Math.round(n * Math.pow(10, d)) / Math.pow(10, d));
  }

  function pct(confidence) {
    if (confidence == null) return "—";
    return fmt(confidence * 100, 1) + "%";
  }

  function badge(text, cls) {
    return `<span class="badge ${cls || "badge-muted"}">${esc(text)}</span>`;
  }

  function riskBadge(level) {
    const c = dict.RISK_LEVELS[level] || dict.RISK_LEVELS["正常"];
    return `<span class="badge ${c.badge}">${c.icon} ${esc(level || "—")}</span>`;
  }

  /* ---------- 结论区（大色块） ---------- */
  function renderVerdict(data) {
    const level = data.risk_level || "—";
    const c = dict.RISK_LEVELS[level] || { cls: "lv-正常" };
    return `
      <div class="verdict ${c.cls}">
        <div class="verdict-score">
          <div class="num">${fmt(data.fused_score, 1)}</div>
          <div class="lbl">融合分 / 100</div>
        </div>
        <div class="verdict-body">
          <div class="verdict-level">风险等级：${esc(level)}</div>
          <div class="verdict-headline">${esc(data.headline || "—")}</div>
          <div class="verdict-reason">${esc(data.reason || "")}</div>
        </div>
      </div>`;
  }

  /* ---------- 告警面板（critical > warning > info 排序） ---------- */
  function renderAlerts(alerts) {
    const list = Array.isArray(alerts) ? alerts.slice() : [];
    list.sort((a, b) => (dict.ALERT_ORDER[a.level] ?? 9) - (dict.ALERT_ORDER[b.level] ?? 9));
    if (list.length === 0) {
      return `<div class="alert-box ok">本次没有触发告警。</div>`;
    }
    return list.map((a) => {
      const lv = dict.ALERT_LEVELS[a.level] || { cls: "info", label: "信息" };
      const codeLabel = dict.fusionCodeLabel(a.code);
      return `<div class="alert-item ${lv.cls}">
        <div style="flex:1">
          <div class="alert-title">${esc(a.title || codeLabel)}</div>
          <div style="font-size:13px;margin-top:2px">${esc(a.detail || "")}</div>
          <div class="alert-code">${esc(a.code || "")}${a.level ? " · " + esc(lv.label) : ""}</div>
        </div>
      </div>`;
    }).join("");
  }

  /* ---------- 动作列表（水质调控 / 病害处置，分开渲染） ---------- */
  function renderActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) {
      return `<div class="empty-state" style="padding:18px">暂无动作</div>`;
    }
    return actions.map((a) => {
      const p = a.priority;
      const cls = p === 1 ? "p1" : p === 2 ? "p2" : "";
      return `<div class="action-item">
        <div class="action-priority ${cls}">${esc(p == null ? "·" : p)}</div>
        <div class="action-main">
          <div class="action-target">${esc(a.target || "—")}</div>
          <div>${esc(a.action || "—")}</div>
          ${a.timing ? `<div class="action-timing">⏱ ${esc(a.timing)}</div>` : ""}
        </div>
      </div>`;
    }).join("");
  }

  /* ---------- 证据链（按 source 分色） ---------- */
  function renderEvidence(evidence) {
    if (!Array.isArray(evidence) || evidence.length === 0) {
      return `<div class="empty-state" style="padding:16px">暂无证据</div>`;
    }
    return evidence.map((e) => {
      const src = dict.EVIDENCE_SOURCES[e.source] || { label: e.source || "?", color: "#9aa3af" };
      const codeLabel = dict.fusionCodeLabel(e.code);
      const val = e.value != null ? fmt(e.value, 2) : "";
      const unit = e.unit ? " " + esc(e.unit) : "";
      return `<div class="evidence-item">
        <span class="evidence-src ${esc(e.source)}" style="background:${src.color}" title="${esc(src.label)}"></span>
        <div style="flex:1">
          <div>${esc(e.label || codeLabel)}</div>
          <div class="evidence-code">${esc(e.code || "")}</div>
        </div>
        <div class="evidence-val">${val}${unit}</div>
      </div>`;
    }).join("");
  }

  /* ---------- 严重度条形（severity_map） ---------- */
  function renderSeverityMap(severityMap) {
    if (!severityMap || typeof severityMap !== "object") return `<div class="empty-state">—</div>`;
    const items = [];
    Object.keys(dict.INDICATORS).forEach((key) => {
      if (severityMap[key] == null) return;
      const v = severityMap[key];
      const label = dict.indicatorLabel(key);
      const sevLabel = dict.severityLabel(v);
      const color = v === 1 ? "#c0392b" : v === 0.5 ? "#e67e22" : "#27ae60";
      items.push({ label: `${label}`, value: Math.max(v, 0.05), color, _sev: sevLabel, _raw: v });
    });
    if (items.length === 0) return `<div class="empty-state">—</div>`;
    const html = items.map((it) => {
      const p = Math.max(0, Math.min(1, it._raw)) * 100;
      return `<div class="sev-row">
        <div class="sev-label">${esc(it.label)}</div>
        <div class="sev-bar"><div class="sev-fill ${it._raw === 1 ? "s1" : it._raw === 0.5 ? "s05" : "s0"}" style="width:${p.toFixed(1)}%"></div></div>
        <div class="sev-tag" style="color:${it.color}">${esc(it._sev)}</div>
      </div>`;
    }).join("");
    return `<div>${html}</div>
      <div class="chart-legend"><span class="lg"><span class="sw" style="background:#27ae60"></span>0.0 正常</span><span class="lg"><span class="sw" style="background:#e67e22"></span>0.5 警戒</span><span class="lg"><span class="sw" style="background:#c0392b"></span>1.0 危险</span></div>`;
  }

  /* ---------- 规则命中（rule_hits → 中文清单 + 红线醒目） ---------- */
  function renderRuleHits(ruleHits, redlineHits) {
    const rules = Array.isArray(ruleHits) ? ruleHits : [];
    const redlines = Array.isArray(redlineHits) ? redlineHits : [];
    if (rules.length === 0 && redlines.length === 0) {
      return `<div class="empty-state" style="padding:16px">未命中任何阈值规则</div>`;
    }
    let html = "";
    if (redlines.length) {
      html += `<div style="margin-bottom:8px">${redlines.map((c) => `<span class="badge badge-danger">⛔ ${esc(dict.redlineLabel(c))}</span>`).join(" ")}</div>`;
    }
    if (rules.length) {
      html += rules.map((c) => {
        const sev = dict.ruleSeverity(c);
        const cls = sev >= 1 ? "badge-danger" : sev >= 0.5 ? "badge-warn" : "badge-info";
        return `<span class="badge ${cls}" style="margin:2px">${esc(dict.ruleLabel(c))} <span class="mono" style="opacity:.65">(${esc(c)})</span></span>`;
      }).join(" ");
    }
    return html;
  }

  /* ---------- 融合算式（怎么算出来的，一目了然） ---------- */
  function renderFusionFormula(fusion) {
    if (!fusion || typeof fusion !== "object") return `<div class="empty-state">—</div>`;
    const w = fusion.effective_weights || {};
    const terms = [];
    if (fusion.water_score != null && w.water != null) {
      terms.push(`<span class="term">水质 ${fmt(fusion.water_score, 4)} × ${fmt(w.water, 2)}</span>`);
    }
    if (fusion.vision_score != null && w.vision != null) {
      terms.push(`<span class="term">图像 ${fmt(fusion.vision_score, 4)} × ${fmt(w.vision, 2)}</span>`);
    }
    const synergy = fusion.synergy_score != null ? fusion.synergy_score : 0;

    let html = `<div class="fusion-formula">`;
    if (terms.length) {
      html += terms.join('<span class="op">+</span>');
    } else {
      html += `<span class="term">无有效模态分数</span>`;
    }
    html += `<span class="op">→</span><span class="term">加权</span>`;
    html += `<span class="op">+</span><span class="term">协同 ${fmt(synergy, 1)}</span>`;
    html += `<span class="op">=</span><span class="res">weighted_score ${fmt(fusion.weighted_score, 1)}</span>`;
    if (fusion.score_floor != null) {
      html += `<span class="op">→</span><span class="term">被下限抬到 ${fmt(fusion.score_floor, 1)}</span>`;
    } else {
      html += `<span class="op">→</span><span class="term">未触发分数下限</span>`;
    }
    html += `<span class="op">=</span><span class="final">fused_score ${fmt(fusion.fused_score, 1)}</span>`;
    html += `</div>`;

    // 命中的协同规则
    if (Array.isArray(fusion.synergy_hits) && fusion.synergy_hits.length) {
      html += `<div class="mt-1">协同规则：` + fusion.synergy_hits.map((s) =>
        `<span class="badge badge-info">${esc(dict.fusionCodeLabel(s))}</span>`).join(" ") + `</div>`;
    }

    // 等级下限高亮
    if (Array.isArray(fusion.level_floors) && fusion.level_floors.length) {
      html += `<div class="mt-1">等级下限：` + fusion.level_floors.map((f) => {
        const label = dict.LEVEL_FLOORS[f] || f;
        return `<span class="badge badge-danger">${esc(label)}</span>`;
      }).join(" ") + `</div>`;
    }
    // 模态与失败模态
    if (Array.isArray(fusion.modalities) && fusion.modalities.length) {
      html += `<div class="mt-1 small muted">参与定级的模态：${esc(fusion.modalities.join("、"))}</div>`;
    }
    if (fusion.failed_modalities && Object.keys(fusion.failed_modalities).length) {
      html += `<div class="mt-1 small" style="color:#c0392b">失败模态：` +
        Object.keys(fusion.failed_modalities).map((k) => `${esc(k)}（${esc(fusion.failed_modalities[k])}）`).join("、") +
        `</div>`;
    }
    return html;
  }

  /* ---------- 推理链路时间轴（trace） ---------- */
  function renderTrace(trace) {
    if (!Array.isArray(trace) || trace.length === 0) {
      return `<div class="empty-state" style="padding:16px">本次未记录推理链路</div>`;
    }
    return `<div class="timeline">` + trace.map((t) => {
      const kind = dict.TRACE_KINDS[t.kind] || { label: t.kind || "步骤", icon: "·" };
      const hasData = t.data && typeof t.data === "object" && Object.keys(t.data).length > 0;
      const dataBlock = hasData
        ? `<details class="fold" style="margin-top:6px"><summary>查看原始 data</summary><div class="fold-body"><pre class="json">${esc(JSON.stringify(t.data, null, 2))}</pre></div></details>`
        : "";
      return `<div class="trace-item" data-kind="${esc(t.kind)}">
        <div class="trace-kind">${kind.icon} ${esc(kind.label)}${t.step != null ? " · 步骤 " + t.step : ""}</div>
        <div class="trace-title">${esc(t.title || "—")}</div>
        ${t.detail ? `<div class="trace-detail">${esc(t.detail)}</div>` : ""}
        ${dataBlock}
      </div>`;
    }).join("") + `</div>`;
  }

  /* ---------- 引用列表（未核对标记警示） ---------- */
  function renderReferences(refs) {
    if (!Array.isArray(refs) || refs.length === 0) {
      return `<div class="empty-state" style="padding:16px">无知识库引用</div>`;
    }
    return refs.map((r) => {
      const unverified = /未核对标准原文/.test(r);
      const cls = unverified ? "reference-item unverified" : "reference-item";
      return `<div class="${cls}">${esc(r)}${unverified ? ` <span class="ref-flag">（未与标准原文核对）</span>` : ""}</div>`;
    }).join("");
  }

  /* ---------- 知识摘要 + 知识块 ---------- */
  function renderKnowledge(knowledge_summary, knowledge_block) {
    let html = "";
    if (knowledge_summary && typeof knowledge_summary === "object") {
      const purposes = knowledge_summary.purposes && Object.keys(knowledge_summary.purposes).length
        ? Object.entries(knowledge_summary.purposes).map(([k, v]) => `${k}×${v}`).join("、")
        : "—";
      html += `<div class="kv"><span class="k">检索块数</span><span class="v">${esc(knowledge_summary.n_chunks ?? "—")}</span></div>`;
      html += `<div class="kv"><span class="k">用途分布</span><span class="v">${esc(purposes)}</span></div>`;
      html += `<div class="kv"><span class="k">后端</span><span class="v small">${esc(knowledge_summary.backend || "—")}</span></div>`;
      if (knowledge_summary.degraded) {
        html += `<div class="alert-box warn" style="margin-top:8px">知识库处于降级状态。</div>`;
      }
      if (Array.isArray(knowledge_summary.warnings) && knowledge_summary.warnings.length) {
        html += `<div class="alert-box warn" style="margin-top:8px">${knowledge_summary.warnings.map(esc).join("<br>")}</div>`;
      }
    }
    if (knowledge_block && String(knowledge_block).trim()) {
      html += `<details class="fold" style="margin-top:8px"><summary>查看注入提示词的知识块原文（Markdown）</summary><div class="fold-body">${md.render(knowledge_block)}</div></details>`;
    } else {
      html += `<div class="small faint mt-1">（本次响应未携带知识块原文，可能已通过 include_knowledge_block=false 关闭）</div>`;
    }
    return html;
  }

  /* ---------- 演示模式 / LLM 后端披露 ---------- */
  function renderDegraded(data) {
    let html = "";
    if (data.degraded_mode) {
      const reasons = Array.isArray(data.degraded_reasons) && data.degraded_reasons.length
        ? data.degraded_reasons.join("；")
        : "当前为确定性模拟输出";
      html += `<div class="alert-box warn">⚠ <div><strong>演示模式</strong>（degraded_mode=true）：${esc(reasons)}</div></div>`;
    }
    if (data.llm_backends && typeof data.llm_backends === "object") {
      const entries = Object.entries(data.llm_backends).filter(([k]) => k !== "prompt_lang");
      html += `<div class="small faint">LLM 后端：${entries.map(([k, v]) => `${esc(k)} = ${esc(v)}`).join("；") || "—"}</div>`;
    }
    return html;
  }

  /* ==========================================================================
   * 完整评估报告渲染器（智能评估页核心：结论 + 融合报告）
   * ========================================================================== */
  function renderReport(data) {
    if (!data || typeof data !== "object") return `<div class="empty-state">无评估结果</div>`;
    return renderVerdict(data) + `<div class="mt-2">${renderDegraded(data)}</div>` + renderReportBody(data);
  }

  /** 报告主体（告警 / 动作 / 融合 / 证据 / trace / 引用 / 知识块 / 原始 JSON），不含结论区 */
  function renderReportBody(data) {
    const report = data.report;
    const fusion = data.fusion;
    let html = "";

    // 2) 告警面板
    html += `<div class="card mt-2"><div class="card-head"><h3 class="card-title">① 告警面板</h3>
      <span class="card-sub">${riskBadge(data.risk_level)}</span></div>
      <div class="card-body">${renderAlerts(data.alerts)}</div></div>`;

    // 3) 水质调控动作 + 病害处置动作（分开显示）
    const waterActions = report && Array.isArray(report.water_actions) ? report.water_actions : null;
    const diseaseActions = report && Array.isArray(report.disease_actions) ? report.disease_actions : null;
    if (waterActions == null && report == null) {
      // report 为 null 时用顶层 advisories 兜底
      html += `<div class="card"><div class="card-head"><h3 class="card-title">② 处置建议</h3>
        <span class="card-sub">report 为 null，使用顶层 advisories 兜底</span></div>
        <div class="card-body">${renderActions(data.advisories)}</div></div>`;
    } else {
      html += `<div class="card"><div class="card-head"><h3 class="card-title">② 水质调控动作</h3>
        <span class="card-sub">去调水</span></div>
        <div class="card-body">${renderActions(waterActions)}</div></div>`;
      html += `<div class="card"><div class="card-head"><h3 class="card-title">③ 病害处置动作</h3>
        <span class="card-sub">去捞病鱼用药</span></div>
        <div class="card-body">${renderActions(diseaseActions)}</div></div>`;
    }

    // 4) 观察窗口 + 可信度说明
    html += `<div class="card"><div class="card-head"><h3 class="card-title">④ 观察窗口与可信度</h3></div>
      <div class="card-body">
        <div class="kv"><span class="k">观察窗口</span><span class="v">${esc(data.observation_window || "—")}</span></div>
        <div class="kv"><span class="k">可信度说明</span><span class="v">${esc(data.confidence_note || "—")}</span></div>
      </div></div>`;

    // 5) 融合解释区
    html += `<div class="card"><div class="card-head"><h3 class="card-title">融合是怎么算出来的</h3></div>
      <div class="card-body">
        ${renderFusionFormula(fusion)}
        ${renderEvidence(data.evidence)}
      </div></div>`;

    // 6) 证据链 + 推理链路 + 引用 + 知识块
    html += `<div class="card"><div class="card-head"><h3 class="card-title">推理链路（trace）</h3></div>
      <div class="card-body">${renderTrace(data.trace)}</div></div>`;

    html += `<div class="card"><div class="card-head"><h3 class="card-title">知识库引用</h3>
      <span class="card-sub">含「未核对标准原文」标记的已警示</span></div>
      <div class="card-body">
        ${renderReferences(data.references)}
        <div class="mt-2" style="border-top:1px solid var(--border);padding-top:10px">${renderKnowledge(data.knowledge_summary, data.knowledge_block)}</div>
      </div></div>`;

    // 7) 原始 JSON（折叠）
    html += `<details class="fold"><summary>原始 JSON（供评审核对）</summary>
      <div class="fold-body"><pre class="json">${esc(JSON.stringify(data, null, 2))}</pre></div></details>`;

    return html;
  }

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.ui = {
    esc, fmt, pct, badge, riskBadge,
    renderVerdict, renderAlerts, renderActions, renderEvidence,
    renderSeverityMap, renderRuleHits, renderFusionFormula, renderTrace,
    renderReferences, renderKnowledge, renderDegraded, renderReport, renderReportBody,
  };
})(window);
