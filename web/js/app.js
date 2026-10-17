/* ============================================================================
 * app.js —— 应用主逻辑：路由 + 六个页面 + 交互
 * ========================================================================== */
(function (global) {
  "use strict";

  const api = global.Fisheries.api;
  const ui = global.Fisheries.ui;
  const charts = global.Fisheries.charts;
  const dict = global.Fisheries.dict;
  const md = global.Fisheries.markdown;
  const parse = global.Fisheries.parse;
  const esc = ui.esc;
  const fmt = ui.fmt;
  const pct = ui.pct;

  const state = {
    health: null,
    thresholds: null,
    tools: null,
    templates: undefined,   // undefined = 还没拉过；null = 拉失败了
    lastAssess: null,
    lastWater: null,        // { readings, species, pondId, intervalMinutes }
    lastVision: null,       // ingest/image 响应
    uploadId: null,
    imagePreviewUrl: null,
    question: "",
    waterParsed: null,      // 解析后待提交的水质结果
    selectedMetric: null,
    chatAnswer: null,
    // 展示开关：默认开，交上去的还是完整报告
    includeTrace: true,
    includeKnowledge: true,
    waterParams: { pondId: "POND-03", species: "鲤科", intervalMinutes: 60, limit: 288 },
  };

  const $ = (s, root) => (root || document).querySelector(s);
  const $$ = (s, root) => Array.from((root || document).querySelectorAll(s));

  /* ================= 路由 ================= */
  const ROUTES = {
    home:      { title: "总览",       desc: "系统状态与历史风险趋势" },
    evaluate:  { title: "智能评估",   desc: "上传水质数据 / 鱼体图片，一键生成处置报告" },
    qa:        { title: "知识问答",   desc: "检索水产养殖知识库，回答养殖问题" },
    standards: { title: "数据与标准", desc: "阈值、权重、红线与工具契约（供评委核对）" },
  };

  function currentRoute() {
    const h = location.hash.replace(/^#\/?/, "").split("?")[0];
    return ROUTES[h] ? h : "home";
  }

  function navigate() {
    const route = currentRoute();
    const meta = ROUTES[route];
    $("#view-title").textContent = meta.title;
    $("#view-desc").textContent = meta.desc;
    $$(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.route === route));
    renderView(route);
  }

  /* ================= 演示模式横幅 ================= */
  function updateDemoBanner() {
    const b = $("#demo-banner");
    const t = $("#demo-banner-text");
    if (state.lastAssess && state.lastAssess.degraded_mode) {
      const reasons = (state.lastAssess.degraded_reasons || []).join("；") || "当前为确定性模拟输出";
      t.textContent = reasons;
      b.hidden = false;
    } else if (state.health && state.health.status === "degraded") {
      const mockTools = (state.health.tools && typeof state.health.tools === "object")
        ? Object.entries(state.health.tools).filter(([, v]) => /mock|模拟/.test(v)).length : 0;
      t.textContent = mockTools > 0
        ? "图像识别 / 建议生成等环节当前为确定性模拟（MockLLM / mock 模型），非真实模型输出，已在报告中如实标注。"
        : "当前服务处于降级模式。";
      b.hidden = false;
    } else {
      b.hidden = true;
    }
  }

  /* ================= 连接状态 ================= */
  async function pingHealth(quiet) {
    const dot = $("#conn-dot");
    const badge = $("#topbar-health");
    try {
      const h = await api.health();
      state.health = h;
      dot.classList.add("ok"); dot.classList.remove("err");
      const mock = (h.tools && typeof h.tools === "object")
        ? Object.values(h.tools).some((v) => /mock|模拟/.test(v)) : false;
      badge.textContent = mock ? "演示模式" : (h.status === "ok" ? "正常" : "降级");
      badge.className = "badge " + (mock ? "badge-warn" : h.status === "ok" ? "badge-ok" : "badge-warn");
    } catch (e) {
      if (!quiet) { /* 静默失败，仅状态点 */ }
      dot.classList.add("err"); dot.classList.remove("ok");
      badge.textContent = "后端未启动";
      badge.className = "badge badge-danger";
    }
    updateDemoBanner();
  }

  /* ================= 视图分发 ================= */
  function renderView(route) {
    const c = $("#content");
    c.innerHTML = '<div class="loading-veil"><span class="spinner"></span>加载中…</div>';
    const fn = {
      evaluate: viewEvaluate, qa: viewQA, standards: viewStandards, home: viewHome,
    }[route];
    fn(c);
  }

  /* ================= 通用渲染包装 ================= */
  function card(title, bodyHtml, opts) {
    opts = opts || {};
    const sub = opts.sub ? `<span class="card-sub">${opts.sub}</span>` : "";
    return `<div class="card"><div class="card-head"><h3 class="card-title">${title}</h3>${sub}</div><div class="card-body">${bodyHtml}</div></div>`;
  }

  /* ================= 内置水质模板 ================= */
  // 模板由后端 /api/v1/config/templates 提供（12 个场景），CSV 由
  // /api/v1/config/templates/{key}/csv 直出。前端不再自己造样例数据 ——
  // 自己造的那份没有「预期等级」，也没有任何回归测试钉着它。
  async function loadTemplates() {
    if (state.templates !== undefined) return state.templates;
    try {
      state.templates = await api.templates();
    } catch (e) {
      state.templates = null;
    }
    return state.templates;
  }

  function fallbackSpecies() {
    const cat = state.templates;
    if (cat && cat.templates) {
      const seen = [];
      cat.templates.forEach((t) => { if (t.species && seen.indexOf(t.species) < 0) seen.push(t.species); });
      if (seen.length) return seen;
    }
    return ["鲤科", "罗非鱼", "虾蟹"];
  }

  function templateOptionsHtml() {
    const cat = state.templates;
    if (!cat || !cat.templates || !cat.templates.length) {
      return '<option value="">（模板目录不可用，请确认接口服务已启动）</option>';
    }
    return cat.templates
      .map((t) => {
        const note = [t.expected_risk, t.species].filter(Boolean).join(" · ");
        return `<option value="${esc(t.key)}">${esc(t.name)}${note ? "（" + esc(note) + "）" : ""}</option>`;
      })
      .join("");
  }

  function findTemplate(key) {
    const cat = state.templates;
    if (!cat || !cat.templates) return null;
    for (let i = 0; i < cat.templates.length; i++) {
      if (cat.templates[i].key === key) return cat.templates[i];
    }
    return null;
  }

  async function applyTemplate(key) {
    const msg = $("#water-template-msg");
    if (!msg) return;
    if (!key) { msg.innerHTML = ""; return; }
    msg.innerHTML = '<span class="spinner"></span>拉取模板…';
    try {
      const resp = await fetch(api.templateCsvUrl(key));
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const text = await resp.text();
      $("#water-paste").value = text;

      // 模板同时决定品种档位与采样间隔 —— 只换数据不换品种，
      // 等于拿虾蟹塘的数字按鲤科的档位判级。
      const t = findTemplate(key);
      if (t) {
        state.waterParams.species = t.species;
        const sel = $("#water-species");
        if (sel && Array.from(sel.options).some((o) => o.value === t.species)) sel.value = t.species;
        if (t.interval_minutes) {
          state.waterParams.intervalMinutes = t.interval_minutes;
          const iv = $("#water-interval");
          if (iv) iv.value = String(t.interval_minutes);
        }
      }
      msg.innerHTML = t
        ? `<span class="ok">已载入「${esc(t.name)}」：${t.n_points} 条 · ${esc(t.species)} · 预期 ${esc(t.expected_risk)}</span>`
        : "";
      parseWaterPaste();
    } catch (e) {
      msg.innerHTML = `<span class="bad">模板拉取失败：${esc(e.message)}</span>`;
    }
  }

  function updateTemplateDownload(key) {
    const a = $("#water-template-dl");
    if (!a) return;
    a.href = key ? api.templateCsvUrl(key) : "#";
  }

  // 模板目录到达后，把「模板下拉」与「品种下拉」都换成真实内容
  function refreshTemplateControls() {
    const sel = $("#water-template");
    if (sel && state.templates) {
      sel.innerHTML = templateOptionsHtml();
      updateTemplateDownload(sel.value);
    }
    const sp = $("#water-species");
    if (sp) {
      const want = state.waterParams.species;
      sp.innerHTML = fallbackSpecies().map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
      if (Array.from(sp.options).some((o) => o.value === want)) sp.value = want;
    }
  }

  /* 把当前解析结果写进服务端塘口会话（POST /ingest/sensor）。
     写进去之后首页的「历史风险趋势」才有数据 —— 之前那条曲线永远是空的。 */
  async function pushToBuffer() {
    const msg = $("#water-parse-msg");
    const p = state.waterParsed;
    if (!p || !p.ok || !p.readings.length) {
      msg.innerHTML = '<span class="bad">请先解析一份水质数据。</span>';
      return;
    }
    msg.innerHTML = '<span class="spinner"></span>写入服务端会话…';
    try {
      const r = await api.ingestSensor({
        pond_id: (state.waterParams.pondId || "").trim() || "unknown",
        readings: p.readings.slice(0, state.waterParams.limit || 288),
        replace: true,
      });
      msg.innerHTML = `<span class="ok">已写入 ${r.accepted} 条（会话中共 ${r.buffered_total} 条，${esc(r.first_timestamp || "—")} → ${esc(r.last_timestamp || "—")}）。可回「总览」查看历史趋势。</span>`;
    } catch (e) {
      msg.innerHTML = `<span class="bad">写入失败：${esc(e.message)}</span>`;
    }
  }

  /* ================= 首页 / 总览 ================= */
  function viewHome(c) {
    // ★ 立即渲染外壳，不 await 任何接口，避免整页灰屏/卡加载
    const h = state.health || {};

    c.innerHTML = `
      <div class="grid grid-3">
        <div class="card"><div class="card-head"><h3 class="card-title">服务状态</h3>
          <button class="btn btn-sm" id="btn-home-refresh" type="button">刷新</button></div>
          <div class="card-body" id="home-health">${renderHealthCard(h)}</div></div>
        <div class="card"><div class="card-head"><h3 class="card-title">工具状态</h3><span class="card-sub">真实 / 模拟</span></div>
          <div class="card-body" id="home-tools">${renderToolsCard(h.tools)}</div></div>
        <div class="card"><div class="card-head"><h3 class="card-title">知识库</h3></div>
          <div class="card-body" id="home-kb">${renderKbCard(h.knowledge)}</div></div>
      </div>

      <div class="card"><div class="card-head"><h3 class="card-title">历史风险趋势</h3>
        <div class="flex gap-8" style="align-items:center">
          <input id="home-pond" type="text" value="${esc(state.lastAssess?.pond_id || 'POND-03')}" placeholder="塘口 ID" style="width:150px">
          <button class="btn btn-sm" id="btn-home-hist" type="button">刷新</button>
        </div></div>
        <div class="card-body">
          <div id="history-chart" class="chart-box"><div class="loading-veil"><span class="spinner"></span>加载历史…</div></div>
        </div></div>

      <div class="card"><div class="card-body pad-lg" style="text-align:center">
        <div style="font-size:28px">⌘</div>
        <div class="mt-1" style="font-weight:700;font-size:16px">上传数据，一键生成处置报告</div>
        <div class="small muted mt-1">支持水质 CSV / 粘贴文本 + 鱼体图片，水质趋势、病害识别、融合定级一站式完成</div>
        <a class="btn btn-primary btn-lg mt-2" href="#/evaluate">立即开始评估 →</a>
      </div></div>
      <div class="card"><div class="card-body pad-lg" style="text-align:center">
        <div style="font-size:28px">?</div>
        <div class="mt-1" style="font-weight:700;font-size:16px">知识库问答</div>
        <div class="small muted mt-1">检索水产养殖知识库，回答投料、水质、病害等问题</div>
        <a class="btn mt-2" href="#/qa">去提问 →</a>
      </div></div>`;

    $("#btn-home-refresh").addEventListener("click", refreshHomeHealth);
    $("#btn-home-hist").addEventListener("click", () => loadHistory($("#home-pond").value.trim()));
    // 异步填充（不阻塞渲染）
    refreshHomeHealth();
    loadHistory($("#home-pond").value.trim() || "POND-03");
  }

  function renderHealthCard(h) {
    h = h || {};
    if (!h.status) {
      return `<div class="loading-veil"><span class="spinner"></span>连接中…</div>`;
    }
    const notes = Array.isArray(h.notes) ? h.notes : [];
    const notesHtml = notes.length
      ? `<div class="mt-2 small faint">${notes.map((n) => `<div>· ${esc(n)}</div>`).join("")}</div>`
      : "";
    return `
      <div class="kv"><span class="k">状态</span><span class="v">${esc(h.status || "—")}</span></div>
      <div class="kv"><span class="k">版本</span><span class="v">${esc(h.version || "—")}</span></div>
      <div class="kv"><span class="k">运行时长</span><span class="v">${fmt(h.uptime_seconds, 0)} 秒</span></div>
      <div class="kv"><span class="k">服务器时间</span><span class="v small">${esc(h.server_time || "—")}</span></div>${notesHtml}`;
  }

  function renderToolsCard(tools) {
    tools = tools || {};
    if (!Object.keys(tools).length) {
      return `<div class="loading-veil"><span class="spinner"></span>连接中…</div>`;
    }
    return Object.entries(tools).map(([k, v]) => {
      const isMock = /mock|模拟/.test(v);
      return `<div class="kv"><span class="k">${esc(k)}</span>
        <span class="v">${isMock ? `<span class="badge badge-warn">模拟</span>` : `<span class="badge badge-ok">真实</span>`} <span class="small muted">${esc(v)}</span></span></div>`;
    }).join("");
  }

  function renderKbCard(kb) {
    kb = kb || {};
    if (!kb.status) {
      return `<div class="loading-veil"><span class="spinner"></span>连接中…</div>`;
    }
    return `
      <div class="kv"><span class="k">状态</span><span class="v">${esc(kb.status || "—")}</span></div>
      <div class="kv"><span class="k">分块数</span><span class="v">${esc(kb.n_chunks ?? "—")}</span></div>
      <div class="kv"><span class="k">后端</span><span class="v small">${esc(kb.backend || "—")}</span></div>`;
  }

  async function refreshHomeHealth() {
    const statusEl = $("#home-health");
    const toolsEl = $("#home-tools");
    const kbEl = $("#home-kb");
    try {
      const h = await api.health();
      state.health = h;
      if (statusEl) statusEl.innerHTML = renderHealthCard(h);
      if (toolsEl) toolsEl.innerHTML = renderToolsCard(h.tools);
      if (kbEl) kbEl.innerHTML = renderKbCard(h.knowledge);
      const mock = (h.tools && typeof h.tools === "object") ? Object.values(h.tools).some((v) => /mock|模拟/.test(v)) : false;
      const badge = $("#topbar-health");
      badge.textContent = mock ? "演示模式" : (h.status === "ok" ? "正常" : "降级");
      badge.className = "badge " + (mock ? "badge-warn" : h.status === "ok" ? "badge-ok" : "badge-warn");
    } catch (e) {
      const errHtml = `<div class="alert-box error">后端未连接：${esc(e.message)}</div>`;
      if (statusEl) statusEl.innerHTML = errHtml;
      if (toolsEl) toolsEl.innerHTML = `<div class="empty-state">—</div>`;
      if (kbEl) kbEl.innerHTML = `<div class="empty-state">—</div>`;
    }
    updateDemoBanner();
  }

  async function loadHistory(pondId) {
    const box = $("#history-chart");
    if (!box) return;
    box.innerHTML = '<div class="loading-veil"><span class="spinner"></span>加载历史…</div>';
    try {
      const h = await api.pondHistory(pondId || "POND-03");
      const items = Array.isArray(h.items) ? h.items : [];
      const scores = items.slice().reverse().map((it) => (it && it.fused_score != null ? it.fused_score : null)).filter((v) => v != null);
      charts.historyTrend(box, scores);
    } catch (e) {
      box.innerHTML = `<div class="alert-box error">加载历史失败：${esc(e.message)}</div>`;
    }
  }

  /* ================= 智能评估页（水质 + 病害 + 融合 合一） ================= */
  function viewEvaluate(c) {
    const parsed = state.waterParsed;
    const hasWater = parsed && parsed.ok && parsed.readings.length >= 3;
    const hasImage = !!state.uploadId && !!state.lastVision;
    const last = state.lastAssess;

    c.innerHTML = `
      ${card("上传数据", `
        <div class="grid grid-2">
          <div>
            <div class="sub-head">≋ 水质数据（可选）</div>
            <div class="field"><label class="field-label">内置水质模板（12 个场景，一键填充 + 自动切品种）</label>
              <div class="flex gap-8 wrap" style="align-items:center">
                <select id="water-template" style="flex:1;min-width:220px">${templateOptionsHtml()}</select>
                <button class="btn" id="btn-water-template" type="button">用该模板填充</button>
                <a class="btn" id="water-template-dl" href="#" download>下载 CSV</a>
              </div>
              <div class="hint mt-1" id="water-template-msg"></div>
            </div>
            <input type="file" id="water-file" accept=".csv,.tsv,.txt,.xlsx,.xlsm" style="padding:6px">
            <div class="hint mt-1">或直接粘贴表格文本（首行为表头：水温 / pH / 溶解氧 / 氨氮；亚硝酸盐可选）</div>
            <textarea id="water-paste" class="mt-1" style="min-height:92px" placeholder="timestamp,water_temp,ph,dissolved_oxygen,ammonia_n,nitrite_n&#10;2026-10-13 00:00,27.5,7.9,6.4,0.1,0.05"></textarea>
            <div class="form-row">
              <div class="field"><label class="field-label">塘口编号</label><input type="text" id="water-pond" value="POND-03" maxlength="64"></div>
              <div class="field"><label class="field-label">品种（决定阈值档位）</label>
                <select id="water-species">${fallbackSpecies().map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("")}</select></div>
            </div>
            <div class="form-row">
              <div class="field" style="flex:2"><label class="field-label">行数上限（12–2000）</label>
                <div class="range-wrap"><input type="range" id="water-limit" min="12" max="2000" value="288"><span class="range-val" id="water-limit-val">288</span></div></div>
              <div class="field"><label class="field-label">采样间隔(分)</label><input type="number" id="water-interval" value="60" min="1"></div>
            </div>
            <div class="flex gap-8 wrap">
              <button class="btn" id="btn-water-parse" type="button">解析并预校验</button>
              <button class="btn" id="btn-water-buffer" type="button">送入塘口会话</button>
            </div>
            <div id="water-parse-msg" class="mt-1"></div>
            <div id="water-preview"></div>
          </div>
          <div>
            <div class="sub-head">◉ 鱼体图片（可选）</div>
            <div class="dropzone" id="disease-drop">
              <span class="dz-icon">🖼</span>
              <div><strong>点击选择 / 拖拽图片到此处</strong></div>
              <div class="small faint">支持 JPG / PNG / WebP，单张 ≤ 10 MB</div>
              <input type="file" id="disease-file" accept="image/*" hidden>
              <div class="mt-1 small">或 <label class="small" style="color:var(--primary);cursor:pointer">拍照<input type="file" id="disease-camera" accept="image/*" capture="environment" hidden></label></div>
            </div>
            <div id="disease-preview" class="mt-2"></div>
            <div id="image-status" class="mt-1"></div>
          </div>
        </div>
        <div class="field mt-2">
          <label class="field-label">补充问题（可选，≤500 字）</label>
          <textarea id="evaluate-question" style="min-height:60px" maxlength="500" placeholder="例如：夜间需要特别注意什么？">${esc(state.question || "")}</textarea>
        </div>
        <div class="form-row">
          <div class="field">
            <label class="field-label">报告内容</label>
            <div class="flex gap-8 wrap">
              <label class="small"><input type="checkbox" id="opt-trace" ${state.includeTrace ? "checked" : ""}> 显示推理链路（trace）</label>
              <label class="small"><input type="checkbox" id="opt-knowledge" ${state.includeKnowledge ? "checked" : ""}> 附带知识库原文块</label>
            </div>
          </div>
        </div>
        <div class="flex gap-8 wrap" style="align-items:center">
          <button class="btn btn-primary btn-lg" id="btn-evaluate-run" type="button" ${(hasWater || hasImage) ? "" : "disabled"}>▶ 运行评估</button>
          <button class="btn" id="btn-evaluate-clear" type="button">清空重来</button>
          <span class="hint">水质与图片至少提供一种。${(hasWater || hasImage) ? "" : "（当前两者都没有，请先上传水质数据或图片）"}</span>
        </div>
      `, { sub: "上传即评估" })}

      <div id="evaluate-result">
        ${last ? renderEvaluationHTML(last, state.lastWater) : '<div class="empty-state"><span class="big">⌘</span>上传水质数据或鱼体图片，点「运行评估」生成决策报告</div>'}
      </div>`;

    // 水质表单绑定
    $("#water-limit").addEventListener("input", (e) => { $("#water-limit-val").textContent = e.target.value; state.waterParams.limit = parseInt(e.target.value, 10) || 288; });
    $("#water-pond").addEventListener("input", (e) => { state.waterParams.pondId = e.target.value; });
    $("#water-species").addEventListener("change", (e) => { state.waterParams.species = e.target.value; });
    $("#water-interval").addEventListener("input", (e) => { state.waterParams.intervalMinutes = parseInt(e.target.value, 10) || 60; });
    $("#water-file").addEventListener("change", (e) => { const f = e.target.files && e.target.files[0]; if (f) parseWaterFile(f); });
    $("#btn-water-parse").addEventListener("click", () => parseWaterPaste());
    $("#water-template").addEventListener("change", (e) => updateTemplateDownload(e.target.value));
    $("#btn-water-template").addEventListener("click", () => applyTemplate($("#water-template").value));
    $("#btn-water-buffer").addEventListener("click", () => pushToBuffer());
    $("#opt-trace").addEventListener("change", (e) => { state.includeTrace = e.target.checked; });
    $("#opt-knowledge").addEventListener("change", (e) => { state.includeKnowledge = e.target.checked; });

    // 模板目录是异步拉的：先按已知状态渲染，拿到之后再补一次（含品种下拉）
    if (state.templates === undefined) {
      loadTemplates().then(() => refreshTemplateControls());
    } else {
      updateTemplateDownload($("#water-template").value);
    }

    // 图片上传绑定
    const dz = $("#disease-drop");
    const file = $("#disease-file");
    dz.addEventListener("click", (e) => { if (e.target.tagName !== "INPUT" && e.target.tagName !== "LABEL") file.click(); });
    file.addEventListener("change", (e) => { if (e.target.files[0]) handleImageFile(e.target.files[0]); });
    $("#disease-camera").addEventListener("change", (e) => { if (e.target.files[0]) handleImageFile(e.target.files[0]); });
    ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => { const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]; if (f) handleImageFile(f); });

    // 运行 / 清空
    $("#btn-evaluate-run").addEventListener("click", runEvaluate);
    $("#btn-evaluate-clear").addEventListener("click", clearEvaluate);
    $("#evaluate-question").addEventListener("input", (e) => { state.question = e.target.value; });

    // 恢复状态
    if (hasWater) renderWaterParsePreview();
    if (last) renderEvaluationCharts(last, state.lastWater);
  }

  function clearEvaluate() {
    state.waterParsed = null;
    state.uploadId = null;
    state.lastVision = null;
    state.lastAssess = null;
    state.lastWater = null;
    state.question = "";
    if (state.imagePreviewUrl) { URL.revokeObjectURL(state.imagePreviewUrl); state.imagePreviewUrl = null; }
    viewEvaluate($("#content"));
  }

  /* ---------- 评估结果组装（结论 + 水质详情 + 图像详情 + 融合报告主体） ---------- */
  function renderEvaluationHTML(data, wInput, wallMs) {
    let html = ui.renderVerdict(data);
    html += `<div class="mt-2">${ui.renderDegraded(data)}</div>`;
    html += renderTimingNote(data, wallMs);

    const water = data.water;
    if (water && water.status === "ok") {
      html += `<h2 class="section-h">≋ 水质模态详情</h2>` + waterSectionHTML(data);
    } else if (water && water.status === "error") {
      html += `<div class="alert-box error">水质分析失败：${esc(water.error_message || dict.errorLabel(water.error_code))}</div>`;
    }

    const vision = data.vision;
    if (vision && vision.status === "ok") {
      html += `<h2 class="section-h">◉ 图像模态详情</h2>` + visionSectionHTML(vision);
    } else if (vision && vision.status === "error") {
      html += `<div class="alert-box error">图像分析失败：${esc(vision.error_message || dict.errorLabel(vision.error_code))}</div>`;
    }

    html += `<h2 class="section-h">⌘ 融合决策报告</h2>` + ui.renderReportBody(data);
    return html;
  }

  /* 服务端自己报的耗时（elapsed_seconds）与浏览器实测的往返时间。
     两个都写出来：差值大说明时间花在网络或排队上，不是算得慢。 */
  function renderTimingNote(data, wallMs) {
    const bits = [];
    if (typeof data.elapsed_seconds === "number") bits.push(`服务端耗时 ${fmt(data.elapsed_seconds, 3)} s`);
    if (typeof wallMs === "number") bits.push(`浏览器往返 ${fmt(wallMs / 1000, 3)} s`);
    if (!bits.length) return "";
    return `<div class="hint mt-1">${esc(bits.join(" · "))}</div>`;
  }

  function renderEvaluationCharts(data, wInput) {
    if (data.water && data.water.status === "ok") renderWaterSectionCharts(data, wInput);
    if (data.vision && data.vision.status === "ok") renderVisionSectionCharts(data.vision);
  }

  /* ---------- 水质详情 ---------- */
  function waterSectionHTML(data) {
    const water = data.water;
    state.selectedMetric = state.selectedMetric || water.limiting_factor || "dissolved_oxygen";
    const dq = water.data_quality || {};
    const eta = water.projected_do_alert_eta_hours;

    let html = `<div class="grid grid-2">
      <div class="card"><div class="card-head"><h3 class="card-title">风险分仪表盘</h3><span class="card-sub">40 预警 · 70 高危</span></div>
        <div class="card-body"><div id="water-gauge"></div></div></div>
      <div class="card"><div class="card-head"><h3 class="card-title">主控风险因子</h3></div>
        <div class="card-body">
          <div style="font-size:26px;font-weight:800;color:var(--danger)">${esc(water.limiting_factor_label || "—")}</div>
          <div class="small muted mt-1">${esc(water.degradation_summary || "—")}</div>
        </div></div>
    </div>`;

    const indicators = Object.keys(dict.INDICATORS).filter((k) => water.trends && water.trends[k]);
    const segBtns = indicators.map((k) => `<button class="${state.selectedMetric === k ? "active" : ""}" data-metric="${k}">${esc(dict.indicatorLabel(k))}</button>`).join("");
    html += card("指标趋势（外推段为虚线，≤12 小时有意义）",
      `<div class="seg" id="water-trend-seg">${segBtns}</div><div id="water-trend-chart" class="mt-2"></div>`);

    html += `<div class="grid grid-2">
      ${card("指标严重度", `<div id="water-severity"></div>`)}
      ${card("命中的规则", `<div id="water-rules"></div><div class="small faint mt-2">红线命中必然为【高危】。</div>`)}
    </div>`;

    const etaText = eta != null
      ? `预计 <strong>${fmt(eta, 1)}</strong> 小时后溶氧跌破 <strong>${fmt(water.projected_do_alert_threshold, 2)} mg/L</strong>（${esc(water.projected_do_alert_label || "阈值")}）`
      : "预计不会跌破阈值";
    const intervalSrc = dq.interval_source === "timestamp" ? "由时间戳推断" : "由参数传入";

    html += `<div class="grid grid-2">
      <div class="card"><div class="card-head"><h3 class="card-title">溶氧跌破预测（线性外推）</h3></div>
        <div class="card-body">
          <div>${etaText}</div>
          <div class="small faint mt-1">这是线性外推，最多 12 小时有意义，只能当趋势提示，非确定预测。</div>
          <div class="small mt-1">超标指标：${(water.exceeded_indicators || []).map((x) => `<span class="badge badge-warn">${esc(x)}</span>`).join(" ") || "无"}</div>
        </div></div>
      <div class="card"><div class="card-head"><h3 class="card-title">数据质量</h3></div>
        <div class="card-body">
          <div class="kv"><span class="k">数据点数</span><span class="v">${esc(dq.n_points ?? "—")}</span></div>
          <div class="kv"><span class="k">采样间隔</span><span class="v">${esc(dq.interval_minutes ?? "—")} 分钟（${intervalSrc}）</span></div>
          <div class="kv"><span class="k">时间跨度</span><span class="v">${fmt(dq.span_hours, 1)} 小时</span></div>
          <div class="kv"><span class="k">缺失指标</span><span class="v">${(dq.missing_indicators || []).map(esc).join("、") || "无"}</span></div>
          ${(dq.warnings || []).length ? `<div class="alert-box warn mt-1">${dq.warnings.map(esc).join("<br>")}</div>` : ""}
        </div></div>
    </div>`;

    return html;
  }

  function renderWaterSectionCharts(data, wInput) {
    const water = data.water;
    charts.gauge($("#water-gauge"), water.risk_score != null ? water.risk_score : data.fused_score, [40, 70], water.risk_level);
    $("#water-severity").innerHTML = ui.renderSeverityMap(water.severity_map);
    $("#water-rules").innerHTML = ui.renderRuleHits(water.rule_hits, water.redline_hits);
    renderWaterTrend(data, wInput, water);
    $$("#water-trend-seg button").forEach((b) => b.addEventListener("click", () => {
      state.selectedMetric = b.dataset.metric;
      renderWaterTrend(data, wInput, water);
      $$("#water-trend-seg button").forEach((x) => x.classList.toggle("active", x.dataset.metric === state.selectedMetric));
    }));
  }

  /* ---------- 图像详情 ---------- */
  function visionSectionHTML(vision) {
    const group = dict.diseaseGroup(vision.disease_type);
    const groupBadge = vision.is_healthy ? "badge-ok" : group === "细菌性" ? "badge-danger" : group === "真菌性" ? "badge-warn" : "badge-info";

    let html = `<div class="grid grid-2">
      <div class="card"><div class="card-head"><h3 class="card-title">识别结果</h3>
        <span class="card-sub">${vision.degraded_mode ? `<span class="badge badge-warn">模拟输出</span>` : ""}</span></div>
        <div class="card-body">
          <div class="flex gap-12 wrap" style="align-items:center">
            <div>
              <div style="font-size:30px;font-weight:800">${esc(vision.disease_type || "—")}</div>
              <div class="mt-1">${`<span class="badge ${groupBadge}">${esc(group)}</span>`} <span class="badge badge-muted">严重度权重 ${fmt(vision.severity_weight, 2)}</span></div>
            </div>
            <div>
              <div class="small muted">置信度 <span style="opacity:.7">（仅在这 4 类之间比较）</span></div>
              <div style="font-size:26px;font-weight:800;font-family:var(--mono)">${pct(vision.confidence)}</div>
            </div>
          </div>
          ${vision.is_low_confidence ? `<div class="alert-box warn mt-2">置信度偏低，建议人工复核，不要强行下结论。</div>` : ""}
          ${vision.degraded_mode ? `<div class="alert-box warn mt-1">当前为模拟输出（model_version=${esc(vision.model_version || "—")}），非真实模型结果。</div>` : ""}
        </div></div>
      <div class="card"><div class="card-head"><h3 class="card-title">模型犹豫程度（all_scores）</h3></div>
        <div class="card-body"><div id="vision-scores"></div></div></div>
    </div>`;
    html += card("说明", `<div>${md.render(vision.note || "—")}</div>
      ${vision.raw_class ? `<div class="small mt-1">原始类别 raw_class：<span class="mono">${esc(vision.raw_class)}</span></div>` : ""}`);
    return html;
  }

  function renderVisionSectionCharts(vision) {
    const allScores = vision.all_scores || {};
    const items = Object.entries(allScores).sort((a, b) => b[1] - a[1]).map(([label, val]) => ({
      label, value: val, color: label === vision.disease_type ? "#0e7c86" : "#b8c2cc",
    }));
    const sc = $("#vision-scores");
    if (sc) charts.hbar(sc, items, { max: 1, valueText: (x) => pct(x) });
  }

  /* ---------- 运行评估 ---------- */
  async function runEvaluate() {
    state.question = $("#evaluate-question").value.trim() || null;
    const hasWater = state.waterParsed && state.waterParsed.ok && state.waterParsed.readings.length >= 3;
    const hasImage = !!state.uploadId;

    const box = $("#evaluate-result");
    box.innerHTML = '<div class="loading-veil"><span class="spinner"></span>正在运行评估…（接真实大模型时可能需数秒）</div>';

    const body = {
      pond_id: (state.waterParams.pondId || "").trim() || "unknown",
      species: state.waterParams.species || "鲤科",
      include_trace: !!state.includeTrace,
      include_knowledge_block: !!state.includeKnowledge,
    };
    if (hasWater) {
      const limit = state.waterParams.limit || 288;
      body.readings = state.waterParsed.readings.slice(0, limit);
    }
    if (hasImage) body.upload_id = state.uploadId;
    if (state.question) body.question = state.question;

    const t0 = (global.performance && global.performance.now) ? global.performance.now() : Date.now();
    try {
      const data = await api.assess(body);
      const wallMs = ((global.performance && global.performance.now) ? global.performance.now() : Date.now()) - t0;
      state.lastAssess = data;
      if (hasWater) {
        state.lastWater = {
          readings: body.readings || [],
          species: body.species,
          pondId: body.pond_id,
          intervalMinutes: state.waterParams.intervalMinutes || 60,
        };
      }
      box.innerHTML = renderEvaluationHTML(data, state.lastWater, wallMs);
      renderEvaluationCharts(data, state.lastWater);
      updateDemoBanner();
    } catch (e) {
      box.innerHTML = `<div class="alert-box error">评估失败：${esc(e.message)}</div>`;
    }
  }

  async function parseWaterFile(file) {
    const msg = $("#water-parse-msg");
    msg.innerHTML = '<div class="loading-veil"><span class="spinner"></span>解析文件…</div>';
    const r = await parse.parseFile(file);
    applyWaterParse(r);
  }

  function parseWaterPaste() {
    const text = $("#water-paste").value;
    if (!text || !text.trim()) {
      $("#water-parse-msg").innerHTML = `<div class="alert-box warn">请先上传文件或粘贴表格文本。</div>`;
      return;
    }
    const r = parse.parseText(text);
    applyWaterParse(r);
  }

  function applyWaterParse(r) {
    state.waterParsed = r;
    const msg = $("#water-parse-msg");
    const runBtn = $("#btn-evaluate-run");

    if (!r.ok) {
      if (runBtn) runBtn.disabled = true;
      msg.innerHTML = r.errors.map((e) => `<div class="alert-box error">✕ ${esc(e)}</div>`).join("") +
        (r.warnings.length ? r.warnings.map((w) => `<div class="alert-box warn">! ${esc(w)}</div>`).join("") : "");
      return;
    }

    if (runBtn) runBtn.disabled = false;
    msg.innerHTML = `<div class="alert-box ok">✓ 解析成功：${r.readings.length} 条有效读数；识别列：${colsText(r.columns)}。</div>` +
      r.warnings.map((w) => `<div class="alert-box warn">! ${esc(w)}</div>`).join("");
    renderWaterParsePreview();
  }

  function colsText(cols) {
    const names = [];
    if (cols.water_temp != null) names.push("水温");
    if (cols.ph != null) names.push("pH");
    if (cols.dissolved_oxygen != null) names.push("溶氧");
    if (cols.ammonia_n != null) names.push("氨氮");
    if (cols.nitrite_n != null) names.push("亚硝酸盐");
    if (cols.timestamp != null) names.push("时间");
    return names.join("、") || "—";
  }

  function renderWaterParsePreview() {
    const r = state.waterParsed;
    const box = $("#water-preview");
    if (!box || !r || !r.readings) return;
    const head = r.stats.header || [];
    const rows = r.readings.slice(0, 5);
    const table = `<div class="table-wrap"><table class="tbl"><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((rd) => `<tr>${head.map((h, i) => {
        const f = fieldOfHeader(h);
        const v = f ? rd[f] : (r.stats.header[i] ? "" : "");
        return `<td class="num">${v == null ? "—" : fmt(v, 3)}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table></div>`;
    box.innerHTML = card("数据预览（前 5 条）", table + `<div class="small faint mt-1">共 ${r.readings.length} 条读数。</div>`);
  }

  function fieldOfHeader(h) {
    const f = parse.matchColumn(h);
    return f;
  }

  function renderWaterTrend(data, wInput, water) {
    const box = $("#water-trend-chart");
    if (!box) return;
    const key = state.selectedMetric;
    const trend = water.trends && water.trends[key];
    const proj = water.projections && water.projections[key];
    const readings = (wInput && wInput.readings) || [];
    if (!trend) { box.innerHTML = '<div class="empty-state">—</div>'; return; }

    // x 轴：有时间戳用时间，否则用序号
    const hasTs = readings.length && readings[0].timestamp != null;
    const points = readings.map((r, i) => ({
      x: hasTs ? (Date.parse(r.timestamp) || i) : i,
      y: r[key],
    })).filter((p) => p.y != null);

    if (!points.length) { box.innerHTML = '<div class="empty-state">—</div>'; return; }

    const series = [{
      name: dict.indicatorLabel(key),
      color: "#0e7c86",
      points,
      extrapolatedCount: 0,
    }];

    // 外推段：最后一个实测点 → 未来 horizon 小时
    const lastReal = points[points.length - 1];
    const horizon = proj ? (proj.horizon_used_hours != null ? proj.horizon_used_hours : water.extrapolation_horizon_hours) : water.extrapolation_horizon_hours;
    if (horizon && proj && proj.projected_at_horizon != null) {
      const extrapX = lastReal.x + (hasTs ? horizon * 3600 * 1000 : horizon);
      series[0].points.push({ x: extrapX, y: proj.projected_at_horizon, extrapolated: true });
      series[0].extrapolatedCount = 1;
    }

    const threshold = (proj && proj.threshold_used != null)
      ? { value: proj.threshold_used, color: "#c0392b", label: proj.threshold_label || "阈值" }
      : null;

    const xMin = points[0].x;
    const xMax = series[0].points[series[0].points.length - 1].x;

    charts.lineChart(box, {
      width: 720, height: 260, series, threshold,
      xMin, xMax,
      xLabel: (v) => hasTs ? charts.fmtTime(new Date(v)) : fmt(v, 0),
    });

    // 下方趋势信息
    const info = `<div class="small muted mt-1">当前 <strong>${fmt(trend.current, 2)}</strong> ${esc(trend.unit || "")} ·
      斜率 ${fmt(trend.slope_per_hour, 3)}/小时 · 方向 <strong>${esc(trend.direction || "—")}</strong> ·
      区间 ${fmt(trend.minimum, 2)}~${fmt(trend.maximum, 2)}</div>`;
    box.insertAdjacentHTML("beforeend", info);
  }

  /* ---------- 图片上传与识别 ---------- */
  async function handleImageFile(file) {
    const preview = $("#disease-preview");
    const status = $("#image-status");
    // 本地预览（objectURL，不显示 stored_path）
    if (state.imagePreviewUrl) URL.revokeObjectURL(state.imagePreviewUrl);
    state.imagePreviewUrl = URL.createObjectURL(file);
    if (preview) preview.innerHTML = `<img class="img-preview" src="${state.imagePreviewUrl}" alt="预览"><div class="small faint mt-1">${esc(file.name)} · ${fmt(file.size / 1024, 1)} KB</div>`;

    if (status) status.innerHTML = '<div class="loading-veil"><span class="spinner"></span>上传并识别中…</div>';
    try {
      const v = await api.ingestImage(file);
      state.lastVision = v;
      state.uploadId = v.upload_id;
      const vision = v.vision || v;
      const disease = vision.disease_type || "—";
      const group = dict.diseaseGroup(disease);
      if (status) status.innerHTML = `<div class="alert-box ok">✓ 已识别：<strong>${esc(disease)}</strong>（${esc(group)}）· 置信度 ${pct(vision.confidence)}${vision.degraded_mode ? ' · <span class="badge badge-warn">模拟</span>' : ""}</div>`;
      const runBtn = $("#btn-evaluate-run");
      if (runBtn) runBtn.disabled = false;
    } catch (e) {
      if (status) status.innerHTML = `<div class="alert-box error">识别失败：${esc(e.message)}</div>`;
    }
  }

  /* ================= 知识问答页 ================= */
  function viewQA(c) {
    const last = state.chatAnswer;
    c.innerHTML = `
      ${card("提问", `
        <div class="field">
          <label class="field-label">问题</label>
          <textarea id="qa-query" style="min-height:70px" placeholder="例如：溶氧低怎么办"></textarea>
        </div>
        <div class="form-row">
          <div class="field">
            <label class="field-label">检索条数 k（1–20）</label>
            <div class="range-wrap"><input type="range" id="qa-k" min="1" max="20" value="5"><span class="range-val" id="qa-k-val">5</span></div>
          </div>
          <div class="field">
            <label class="field-label">分类筛选</label>
            <select id="qa-category"><option value="">全部</option>
              <option>水质</option><option>病害</option><option>投料</option><option>应急</option><option>标准</option></select>
          </div>
          <div class="field">
            <label class="field-label">病害筛选</label>
            <select id="qa-disease"><option value="">全部</option>
              <option>烂鳃</option><option>出血</option><option>水霉</option><option>其他</option></select>
          </div>
        </div>
        <div class="flex gap-8" style="align-items:center">
          <button class="btn btn-primary" id="btn-qa-ask" type="button">提问</button>
          <label class="small" style="display:flex;align-items:center;gap:6px;cursor:pointer">
            <input type="checkbox" id="qa-llm"> 调用大模型润色回答（默认关；未配置大模型时后端会自动降级为原文拼接）
          </label>
        </div>
      `)}

      <div id="qa-result">${last ? renderChatAnswer(last) : '<div class="empty-state"><span class="big">?</span>输入问题，检索养殖知识库</div>'}</div>`;

    $("#qa-k").addEventListener("input", (e) => { $("#qa-k-val").textContent = e.target.value; });
    $("#btn-qa-ask").addEventListener("click", runQA);
    $("#qa-query").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runQA(); });
  }

  async function runQA() {
    const query = $("#qa-query").value.trim();
    if (!query) {
      $("#qa-result").innerHTML = '<div class="alert-box warn">请输入问题。</div>';
      return;
    }
    const box = $("#qa-result");
    box.innerHTML = '<div class="loading-veil"><span class="spinner"></span>检索中…</div>';
    const body = {
      query,
      k: parseInt($("#qa-k").value, 10) || 5,
      answer_with_llm: $("#qa-llm").checked,
    };
    const cat = $("#qa-category").value;
    const dis = $("#qa-disease").value;
    if (cat) body.category = cat;
    if (dis) body.disease = dis;
    try {
      const data = await api.chat(body);
      state.chatAnswer = data;
      box.innerHTML = renderChatAnswer(data);
    } catch (e) {
      box.innerHTML = `<div class="alert-box error">问答失败：${esc(e.message)}</div>`;
    }
  }

  function renderChatAnswer(data) {
    let html = "";
    html += card("回答", `
      ${data.answer_is_raw_retrieval ? `<div class="alert-box warn">⚠ 以下为检索原文拼接，非模型生成（后端未配置大模型时故意不编答案）。</div>` : ""}
      <div class="chat-answer">${md.render(data.answer || "—")}</div>
      ${(data.warnings || []).length ? `<div class="alert-box warn mt-2">${data.warnings.map(esc).join("<br>")}</div>` : ""}
    `);

    const chunks = Array.isArray(data.chunks) ? data.chunks : [];
    html += card("引用来源（由 chunks 拼出）", chunks.length
      ? chunks.map((ch) => {
        const cite = `《${esc(ch.title || "—")}》${ch.heading_text ? " > " + esc(ch.heading_text) : ""}`;
        const matched = Array.isArray(ch.matched_by) ? ch.matched_by.join(" / ") : (ch.matched_by || "—");
        return `<div class="chunk-item">
          <div class="chunk-head">
            ${ch.verified === false ? `<span class="badge badge-warn">未与标准原文核对</span>` : `<span class="badge badge-ok">已核对</span>`}
            <span>${cite}</span>
          </div>
          <div class="chunk-meta">
            <span>score ${fmt(ch.score, 3)}</span>
            <span>dense ${fmt(ch.dense_score, 3)}</span>
            <span>sparse ${fmt(ch.sparse_score, 3)}</span>
            <span>匹配方式：${esc(matched)}</span>
            ${ch.disease ? `<span>病害：${esc(ch.disease)}</span>` : ""}
            ${ch.category ? `<span>分类：${esc(ch.category)}</span>` : ""}
          </div>
          <details class="fold" style="margin-top:8px"><summary>查看原文</summary><div class="fold-body">${md.render(ch.text || "—")}</div></details>
        </div>`;
      }).join("")
      : `<div class="empty-state">未检索到相关内容</div>`);

    html += `<div class="small faint">检索后端：${esc(data.backend || "—")}</div>`;
    return html;
  }

  /* ================= 数据与标准页 ================= */
  async function viewStandards(c) {
    c.innerHTML = '<div class="loading-veil"><span class="spinner"></span>加载阈值标准…</div>';
    let th = null, tools = null;
    try {
      th = await api.thresholds();
      state.thresholds = th;
    } catch (e) { c.innerHTML = `<div class="alert-box error">加载阈值失败：${esc(e.message)}</div>`; return; }
    try { tools = await api.tools(); state.tools = tools; } catch (e) { /* 可选 */ }

    const species = Array.isArray(th.species) ? th.species : [];
    const profiles = th.profiles || {};
    const weights = th.weights || {};
    const redlines = th.redlines || {};
    const actionThresholds = th.action_thresholds || {};

    // 品种适宜/警戒区间表
    const profileRows = species.map((sp) => {
      const p = profiles[sp] || {};
      const range = (arr) => Array.isArray(arr) && arr.length === 2 ? `${arr[0]}–${arr[1]}` : "—";
      return `<tr><th>${esc(sp)}</th>
        <td class="num">${range(p.temp_opt)}</td><td class="num">${range(p.temp_warn)}</td>
        <td class="num">${range(p.ph_opt)}</td><td class="num">${range(p.ph_warn)}</td>
        <td class="num">${fmt(p.do_opt, 1)}</td><td class="num">${fmt(p.do_warn, 1)}</td>
        <td class="num">${fmt(p.ammonia_opt, 2)}</td><td class="num">${fmt(p.ammonia_warn, 2)}</td>
        <td class="num">${fmt(p.nitrite_opt, 2)}</td><td class="num">${fmt(p.nitrite_warn, 2)}</td></tr>`;
    }).join("");

    const profileTable = `<div class="table-wrap"><table class="tbl"><thead><tr>
      <th>品种</th><th>水温适宜</th><th>水温警戒</th><th>pH适宜</th><th>pH警戒</th>
      <th>溶氧适宜</th><th>溶氧警戒</th><th>氨氮适宜</th><th>氨氮警戒</th><th>亚硝酸盐适宜</th><th>亚硝酸盐警戒</th>
      </tr></thead><tbody>${profileRows}</tbody></table></div>`;

    // 权重
    const weightRows = Object.entries(weights).map(([k, v]) =>
      `<div class="kv"><span class="k">${esc(dict.indicatorLabel(k))}</span><span class="v">${fmt(v, 2)}</span></div>`).join("");

    // 红线
    const redlineLabels = {
      DO_REDLINE: "溶氧红线", NH3_REDLINE: "氨氮红线", NO2_REDLINE: "亚硝酸盐红线",
      PH_LOW_REDLINE: "pH 下限红线", PH_HIGH_REDLINE: "pH 上限红线",
      DO_CRASH_DROP: "溶氧骤降幅度", DO_CRASH_WINDOW_H: "溶氧骤降窗口",
    };
    const redlineRows = Object.entries(redlines).map(([k, v]) =>
      `<tr><td><span class="mono">${esc(k)}</span></td><td>${esc(redlineLabels[k] || "—")}</td><td class="num">${fmt(v, 2)}</td></tr>`).join("");

    // action_thresholds：形状是 指标 → 数组[{value,label}]，必须按数组渲染
    const actionRows = Object.entries(actionThresholds).map(([k, arr]) => {
      const lines = Array.isArray(arr) ? arr.map((a) => `${fmt(a.value, 2)}（${esc(a.label)}）`).join("、") : "—";
      return `<tr><td>${esc(dict.indicatorLabel(k))}</td><td>${lines}</td></tr>`;
    }).join("");

    c.innerHTML = `
      ${card("免责声明（必须原样显示）", `<div class="alert-box warn">${esc(th.disclaimer || "—")}</div>`)}

      <div class="grid grid-3">
        <div class="stat-card"><div class="k">外推最大时长</div><div class="v">${fmt(th.extrapolation_horizon_hours, 1)}</div><div class="s">小时（超过部分仅趋势提示）</div></div>
        <div class="stat-card"><div class="k">可靠趋势最小点数</div><div class="v">${esc(th.min_reliable_points ?? "—")}</div><div class="s">条读数</div></div>
        <div class="stat-card"><div class="k">持久性基线上限</div><div class="v">${fmt(th.persistence_cap, 2)}</div><div class="s">persistence_cap</div></div>
      </div>

      ${card("品种阈值档案（profiles）", profileTable)}

      <div class="grid grid-2">
        ${card("基础权重（weights）", `<div>${weightRows}</div><div class="small faint mt-1">${esc(th.effective_note || "")}</div>`)}
        ${card("红线（redlines）", `<div class="table-wrap"><table class="tbl"><thead><tr><th>键</th><th>含义</th><th class="num">值</th></tr></thead><tbody>${redlineRows}</tbody></table></div>`)}
      </div>

      ${card("处置阈值（action_thresholds）", `<div class="small faint">一个指标可能有多条线，按数组渲染。</div>
        <div class="table-wrap mt-1"><table class="tbl"><thead><tr><th>指标</th><th>阈值线（value · label）</th></tr></thead><tbody>${actionRows}</tbody></table></div>`)}
    `;

    // 工具契约（可选）
    if (tools && Array.isArray(tools.tools) && tools.tools.length) {
      const toolBlocks = tools.tools.map((t) => {
        const fn = t.function || {};
        const params = fn.parameters || {};
        return `<details class="fold"><summary>工具：${esc(fn.name || "—")}</summary>
          <div class="fold-body">
            <div class="small muted">${esc(fn.description || "")}</div>
            <pre class="json mt-1">${esc(JSON.stringify(params, null, 2))}</pre>
          </div></details>`;
      }).join("");
      c.insertAdjacentHTML("beforeend", card("工具契约（function calling schema，给技术评委看）", toolBlocks));
    }
  }

  /* ================= 设置模态 ================= */
  function openConfig() {
    const b = $("#modal-backdrop");
    b.hidden = false;
    b.innerHTML = `<div class="modal">
      <h3 class="modal-title">接口基地址设置</h3>
      <p class="small muted">前端与服务由同一个进程托管，默认<b>不需要改这里</b>。<br>
      · 页面从服务打开（<span class="mono">http://…/app/</span>）时留空即为同源；<br>
      · 直接双击 <span class="mono">index.html</span> 打开时，需要填后端地址；<br>
      · 后端另起在别的机器/端口时，填那个地址。</p>
      <div class="field"><label class="field-label">基地址（留空 = 与本页面同源）</label>
        <input type="text" id="config-url" value="${esc(api.getBaseURL())}" placeholder="（同源）"></div>
      <div class="small faint">当前解析为：<span class="mono">${esc(api.describeBase())}</span></div>
      <div class="flex gap-8" style="justify-content:flex-end">
        <button class="btn" id="config-cancel" type="button">取消</button>
        <button class="btn btn-primary" id="config-save" type="button">保存并重连</button>
      </div>
    </div>`;
    $("#config-cancel").addEventListener("click", () => { b.hidden = true; });
    $("#config-save").addEventListener("click", () => {
      const url = $("#config-url").value.trim();
      api.setBaseURL(url);
      $("#conn-url").textContent = api.getBaseURL();
      b.hidden = true;
      pingHealth(false).then(() => { navigate(); });
    });
  }

  /* ================= 初始化 ================= */
  function init() {
    $("#conn-url").textContent = api.describeBase();
    $("#btn-config").addEventListener("click", openConfig);
    window.addEventListener("hashchange", navigate);
    loadTemplates();          // 预取模板目录，评估页就不必等
    pingHealth(false);
    navigate();
    // 定期刷新连接状态
    setInterval(() => pingHealth(true), 30000);
  }

  // 暴露给模块内/调试
  global.App = {
    state,
    openConfig,
    parseWaterPaste,
    runEvaluate,
    clearEvaluate,
    runQA,
    handleImageFile,
    loadTemplates,
    applyTemplate,
    pushToBuffer,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
