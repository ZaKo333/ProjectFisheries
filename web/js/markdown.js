/* ============================================================================
 * markdown.js —— 轻量 Markdown 渲染器（零依赖，先转义 HTML 再套 Markdown）
 * 用于渲染 knowledge_block / vision.note / chat.answer 等字段。
 * ========================================================================== */
(function (global) {
  "use strict";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** 行内元素：先转义 HTML，再套 Markdown（code / bold / italic / links） */
  function inline(text) {
    text = esc(text);
    // 先把行内代码占位，避免其内容被后续规则干扰
    const codes = [];
    text = text.replace(/`([^`]+)`/g, (m, c) => {
      codes.push(`<code class="mono">${c}</code>`);
      return `\u0000${codes.length - 1}\u0000`;
    });
    // 链接
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // 粗体
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // 斜体
    text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    // 还原代码
    text = text.replace(/\u0000(\d+)\u0000/g, (m, i) => codes[Number(i)]);
    return text;
  }

  function render(md) {
    if (md == null) return "";
    const src = String(md).replace(/\r\n/g, "\n");
    // 切出代码块，保护之
    const blocks = [];
    let text = src.replace(/```([\s\S]*?)```/g, (m, code) => {
      blocks.push(`<pre class="json"><code>${esc(code.replace(/^\n/, ""))}</code></pre>`);
      return `\u0001${blocks.length - 1}\u0001`;
    });

    const lines = text.split("\n");
    const out = [];
    let i = 0;
    let inList = null; // 'ul' | 'ol'
    let inTable = false;
    let tableRows = [];

    function closeList() {
      if (inList) { out.push(`</${inList}>`); inList = null; }
    }
    function closeTable() {
      if (inTable) {
        out.push('<div class="table-wrap"><table class="tbl">' + tableRows.join("") + "</table></div>");
        inTable = false; tableRows = [];
      }
    }

    function flushBlock() { closeList(); closeTable(); }

    while (i < lines.length) {
      let line = lines[i];

      // 代码块占位
      if (/\u0001\d+\u0001/.test(line)) {
        flushBlock();
        out.push(line.replace(/\u0001(\d+)\u0001/g, (m, k) => blocks[Number(k)]));
        i++; continue;
      }

      // 空行
      if (line.trim() === "") { flushBlock(); i++; continue; }

      // 表格：| a | b | ... | 分隔行 |---|---|
      if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
        const cells = splitRow(line);
        if (!inTable) { inTable = true; tableRows = []; }
        if (isTableSep(cells)) {
          // 分隔行：跳过（首个表头已生成 thead 之外，这里直接略过）
          i++; continue;
        }
        const isHead = tableRows.length === 0;
        const tag = isHead ? "th" : "td";
        tableRows.push("<tr>" + cells.map((c) => `<${tag}>${inline(c.trim())}</${tag}>`).join("") + "</tr>");
        i++; continue;
      }

      flushBlock();

      // 标题
      let m = line.match(/^(#{1,6})\s+(.*)$/);
      if (m) {
        const level = m[1].length;
        out.push(`<h${level} style="margin:14px 0 6px">${inline(m[2])}</h${level}>`);
        i++; continue;
      }

      // 引用
      if (/^>\s?/.test(line)) {
        const quoteLines = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^>\s?/, ""));
          i++;
        }
        out.push(`<blockquote style="margin:8px 0;padding:8px 14px;border-left:3px solid #0e7c86;background:#f2f7f8;color:#41545b">${render(quoteLines.join("\n"))}</blockquote>`);
        continue;
      }

      // 无序列表
      m = line.match(/^\s*[-*+]\s+(.*)$/);
      if (m) {
        if (inList !== "ul") { flushBlock(); out.push("<ul style='margin:6px 0;padding-left:22px'>"); inList = "ul"; }
        out.push(`<li>${inline(m[1])}</li>`);
        i++; continue;
      }

      // 有序列表
      m = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
      if (m) {
        if (inList !== "ol") { flushBlock(); out.push("<ol style='margin:6px 0;padding-left:22px'>"); inList = "ol"; }
        out.push(`<li>${inline(m[2])}</li>`);
        i++; continue;
      }

      // 水平线
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        out.push("<hr style='border:none;border-top:1px solid #e5e9ee;margin:12px 0'>");
        i++; continue;
      }

      // 普通段落（合并连续非空行）
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim() !== "" &&
             !/^(#{1,6})\s/.test(lines[i]) &&
             !/^\s*[-*+]\s+/.test(lines[i]) &&
             !/^\s*(\d+)[.)]\s+/.test(lines[i]) &&
             !/^>\s?/.test(lines[i]) &&
             !/^\|/.test(lines[i]) &&
             !/\u0001\d+\u0001/.test(lines[i]) &&
             !/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i])) {
        para.push(lines[i]); i++;
      }
      out.push(`<p style="margin:6px 0">${inline(para.join("<br>"))}</p>`);
    }
    flushBlock();

    return `<div class="md">${out.join("")}</div>`;
  }

  function splitRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|")) s = s.slice(0, -1);
    // 简单按 | 切（不处理转义，够用）
    return s.split("|");
  }
  function isTableSep(cells) {
    return cells.every((c) => /^:?-{2,}:?$/.test(c.trim()) || /^-{2,}$/.test(c.trim())) && cells.length > 0;
  }

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.markdown = { render, esc };
})(window);
