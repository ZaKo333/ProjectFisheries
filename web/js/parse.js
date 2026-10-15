/* ============================================================================
 * parse.js —— 水质数据解析（CSV / TSV / XLSX / 粘贴文本）+ 列名模糊识别 + 前端预校验
 * 依据《前端要求》7.2 的预校验规则。
 * ========================================================================== */
(function (global) {
  "use strict";

  /* ---------------- 列名识别 ---------------- */

  // 归一化：小写、去括号内容、去空格/下划线/连字符/点/单位
  function normName(s) {
    return String(s == null ? "" : s)
      .toLowerCase()
      .replace(/[（(][^）)]*[）)]/g, "")   // 去掉中文/英文括号内容
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "") // 去除非字母数字中文（含空格、下划线、连字符、点）
      .replace(/℃|°c|度/g, "");
  }

  const ALIASES = {
    water_temp: ["watertemp", "temp", "temperature", "watertemperature", "水温", "温度", "水温temp", "tempwater"],
    ph: ["ph", "ph值", "phvalue", "酸碱度", "ph值"],
    dissolved_oxygen: ["dissolvedoxygen", "do", "溶解氧", "溶氧", "溶解氧do", "溶氧量", "do值", "dioxygen"],
    ammonia_n: ["ammonian", "ammonia", "nh3n", "nh3", "nh4n", "nh4", "氨氮", "氨氮nh3n", "总氨氮", "ammonianitrogen"],
    nitrite_n: ["nitriten", "nitrite", "no2n", "no2", "亚硝酸盐", "亚硝", "亚硝酸盐氮", "no2亚硝酸盐", "亚硝酸盐no2n"],
    timestamp: ["timestamp", "time", "datetime", "date", "时间", "日期", "采样时间", "时刻", "时间戳"],
  };

  function matchColumn(rawHeader) {
    const key = normName(rawHeader);
    if (!key) return null;
    for (const field in ALIASES) {
      if (ALIASES[field].includes(key)) return field;
    }
    // 模糊匹配：包含关系（用于纠正拼写错误，如 "dissolvedoxygen" 已覆盖，这里兜底）
    for (const field in ALIASES) {
      if (ALIASES[field].some((a) => key.includes(a) || a.includes(key))) return field;
    }
    return null;
  }

  /* ---------------- CSV / TSV 解析 ---------------- */

  function parseDelimited(text, delimiter) {
    const rows = [];
    let row = [], cell = "", inQ = false;
    const t = String(text).replace(/^\uFEFF/, "");
    for (let i = 0; i < t.length; i++) {
      const c = t[i];
      if (inQ) {
        if (c === '"') {
          if (t[i + 1] === '"') { cell += '"'; i++; }
          else inQ = false;
        } else cell += c;
      } else {
        if (c === '"') inQ = true;
        else if (c === delimiter) { row.push(cell); cell = ""; }
        else if (c === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
        else if (c === "\r") { /* 忽略 */ }
        else cell += c;
      }
    }
    if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
    // 去掉全空尾行
    while (rows.length && rows[rows.length - 1].every((c) => c.trim() === "")) rows.pop();
    return rows;
  }

  function detectDelimiter(text) {
    const head = text.split("\n")[0] || "";
    const tab = (head.match(/\t/g) || []).length;
    const comma = (head.match(/,/g) || []).length;
    const semi = (head.match(/;/g) || []).length;
    if (tab >= comma && tab >= semi && tab > 0) return "\t";
    if (comma >= semi && comma > 0) return ",";
    if (semi > 0) return ";";
    return "\t";
  }

  /* ---------------- XLSX 解析（best-effort，DecompressionStream） ---------------- */

  async function parseXlsx(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    const dv = new DataView(arrayBuffer);
    const te = new TextDecoder("utf-8");

    // 找中央目录结束记录（0x06054b50）
    let eocd = -1;
    const minLen = Math.min(bytes.length, 1024 * 64);
    for (let i = bytes.length - minLen; i < bytes.length - 21; i++) {
      if (bytes[i] === 0x50 && bytes[i + 1] === 0x4b && bytes[i + 2] === 0x05 && bytes[i + 3] === 0x06) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error("不是有效的 XLSX 文件");
    const cdCount = dv.getUint16(eocd + 10, true);
    let cdOffset = dv.getUint32(eocd + 16, true);

    const entries = [];
    for (let k = 0; k < cdCount; k++) {
      if (dv.getUint32(cdOffset, true) !== 0x02014b50) break;
      const method = dv.getUint16(cdOffset + 10, true);
      const compSize = dv.getUint32(cdOffset + 20, true);
      const uncompSize = dv.getUint32(cdOffset + 24, true);
      const nameLen = dv.getUint16(cdOffset + 28, true);
      const extraLen = dv.getUint16(cdOffset + 30, true);
      const commentLen = dv.getUint16(cdOffset + 32, true);
      const localOff = dv.getUint32(cdOffset + 42, true);
      const name = te.decode(bytes.subarray(cdOffset + 46, cdOffset + 46 + nameLen));
      entries.push({ name, method, compSize, uncompSize, localOff });
      cdOffset += 46 + nameLen + extraLen + commentLen;
    }

    function readEntry(name) {
      const e = entries.find((x) => x.name === name || x.name.endsWith("/" + name));
      if (!e) return null;
      const off = e.localOff;
      if (dv.getUint32(off, true) !== 0x04034b50) return null;
      const lnameLen = dv.getUint16(off + 26, true);
      const lextraLen = dv.getUint16(off + 28, true);
      const dataStart = off + 30 + lnameLen + lextraLen;
      const comp = bytes.subarray(dataStart, dataStart + e.compSize);
      if (e.method === 0) return te.decode(comp.subarray(0, e.uncompSize));
      if (e.method === 8) {
        // DEFLATE
        if (typeof DecompressionStream !== "undefined") {
          const ds = new DecompressionStream("deflate-raw");
          const stream = new Blob([comp]).stream().pipeThrough(ds);
          return new Response(stream).text();
        }
        throw new Error("浏览器不支持 DecompressionStream，无法解析 XLSX（请改用 CSV）");
      }
      throw new Error("不支持的压缩方式");
    }

    const sharedStrXml = await readEntry("xl/sharedStrings.xml");
    const sheetXml = await readEntry("xl/worksheets/sheet1.xml") || await readEntry("xl/worksheets/sheet.xml");
    if (!sheetXml) throw new Error("未找到工作表");

    const sharedStrings = [];
    if (sharedStrXml) {
      const doc = new DOMParser().parseFromString(sharedStrXml, "application/xml");
      doc.querySelectorAll("si").forEach((si) => {
        // 拼接所有 <t>（含富文本 <r><t>）
        sharedStrings.push(Array.from(si.querySelectorAll("t")).map((t) => t.textContent).join(""));
      });
    }

    const doc = new DOMParser().parseFromString(sheetXml, "application/xml");
    const grid = [];
    let maxCol = 0;
    doc.querySelectorAll("sheetData > row").forEach((r) => {
      const rowIdx = parseInt(r.getAttribute("r"), 10) - 1;
      if (rowIdx < 0) return;
      const rowArr = [];
      r.querySelectorAll(":scope > c").forEach((c) => {
        const ref = c.getAttribute("r") || "";
        const colIdx = colFromRef(ref);
        const t = c.getAttribute("t");
        const v = c.querySelector("v");
        const is = c.querySelector("is t");
        let val;
        if (t === "s" && v) val = sharedStrings[parseInt(v.textContent, 10)] ?? "";
        else if (t === "inlineStr" && is) val = is.textContent;
        else if (v) val = v.textContent;
        else if (t === "str") val = c.querySelector("v")?.textContent || "";
        else val = "";
        rowArr[colIdx] = val;
        if (colIdx > maxCol) maxCol = colIdx;
      });
      grid[rowIdx] = rowArr;
    });

    // 规整成矩形
    const rows = [];
    for (let r = 0; r < grid.length; r++) {
      const arr = [];
      for (let c = 0; c <= maxCol; c++) arr.push(grid[r] && grid[r][c] !== undefined ? grid[r][c] : "");
      if (arr.some((x) => String(x).trim() !== "")) rows.push(arr);
    }
    return rows;
  }

  function colFromRef(ref) {
    const m = ref.match(/^([A-Z]+)/);
    if (!m) return 0;
    let n = 0;
    for (let i = 0; i < m[1].length; i++) n = n * 26 + (m[1].charCodeAt(i) - 64);
    return n - 1;
  }

  /* ---------------- 哨兵值 / 数值解析 ---------------- */

  const SENTINELS = ["-127", "-999", "-9999", "-9999.0", "n/a", "na", "null", "none", "nan", "#n/a", "#value!", "—", "暂无"];

  function toNumber(raw) {
    if (raw == null) return null;
    const s = String(raw).trim();
    if (s === "") return null;
    if (SENTINELS.includes(s.toLowerCase())) return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }

  function excelSerialToISO(serial) {
    // Excel 日期序列（1899-12-30 为 0）
    const ms = Math.round((serial - 25569) * 86400 * 1000);
    const d = new Date(ms);
    if (Number.isNaN(d.getTime())) return null;
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  /* ---------------- 主入口：把原始行数组转成 readings + 校验 ---------------- */

  function buildReadings(rows) {
    const result = {
      ok: false,
      readings: [],
      columns: {},
      errors: [],
      warnings: [],
      stats: { rows: 0, header: null, delimiter: null, hasTimestamp: false, nitriteHasMissing: false, source: null },
    };
    if (!rows || rows.length === 0) {
      result.errors.push("没有读到任何数据行");
      return result;
    }

    // 找表头行：第一行含可识别列名
    const headerIdx = 0;
    const header = rows[headerIdx].map((h) => String(h == null ? "" : h).trim());
    const colMap = {}; // field -> colIndex
    header.forEach((h, ci) => {
      const f = matchColumn(h);
      if (f && colMap[f] == null) colMap[f] = ci;
    });
    result.columns = colMap;
    result.stats.header = header;

    const required = ["water_temp", "ph", "dissolved_oxygen", "ammonia_n"];
    const missingCols = required.filter((f) => colMap[f] == null);
    if (missingCols.length) {
      const cn = { water_temp: "水温", ph: "pH", dissolved_oxygen: "溶解氧/溶氧", ammonia_n: "氨氮" };
      result.errors.push("无法识别以下必需列：" + missingCols.map((f) => `「${cn[f]}」`).join("、") +
        "。请检查表头列名（如 水温、pH、溶解氧、氨氮），并确认不是空格或拼写错误。");
      return result;
    }

    const dataRows = rows.slice(headerIdx + 1).filter((r) => r.some((c) => String(c == null ? "" : c).trim() !== ""));

    // 检测时间列是否全有
    let tsCount = 0, tsPresent = 0;
    if (colMap.timestamp != null) {
      dataRows.forEach((r) => {
        const raw = String(r[colMap.timestamp] == null ? "" : r[colMap.timestamp]).trim();
        if (raw !== "") tsPresent++;
      });
      tsCount = dataRows.length;
    }

    // 亚硝酸盐缺失检测
    let nitriteHasMissing = false;
    if (colMap.nitrite_n != null) {
      nitriteHasMissing = dataRows.some((r) => toNumber(r[colMap.nitrite_n]) == null);
    }
    result.stats.nitriteHasMissing = nitriteHasMissing;
    if (nitriteHasMissing) {
      result.warnings.push("亚硝酸盐列存在缺失/空值：后端会把整列剔除、不参与打分（其余指标权重按比例归一化）。");
    }

    // 时间列「要么全有要么全无」校验
    if (colMap.timestamp != null && tsPresent > 0 && tsPresent < tsCount) {
      result.warnings.push("时间列存在部分缺失：后端要求时间戳要么全给要么全不给，缺失处将按等间隔处理。");
    }

    const readings = [];
    dataRows.forEach((r, idx) => {
      const rec = {};
      let hasAny = false;
      required.concat(["nitrite_n"]).forEach((f) => {
        const ci = colMap[f];
        if (ci == null) return;
        const n = toNumber(r[ci]);
        if (n != null) { rec[f] = n; hasAny = true; }
        else if (f === "nitrite_n") { /* 亚硝酸盐可缺，留空由后端整列剔除 */ }
      });
      if (colMap.timestamp != null) {
        const raw = String(r[colMap.timestamp] == null ? "" : r[colMap.timestamp]).trim();
        if (raw !== "") {
          // 数字 → 可能是 Excel 序列日期
          const n = Number(raw);
          if (Number.isFinite(n) && n > 20000 && n < 60000) {
            const iso = excelSerialToISO(n);
            if (iso) rec.timestamp = iso;
          } else {
            rec.timestamp = raw;
          }
        }
      }
      // 四个必填项必须都有值，否则该行无效
      const okRow = required.every((f) => rec[f] != null);
      if (okRow) {
        if (rec.timestamp == null) delete rec.timestamp;
        readings.push(rec);
      }
      if (hasAny && !okRow) {
        result.warnings.push(`第 ${idx + 2} 行数据缺少必填指标，已跳过。`);
      }
    });

    result.stats.rows = readings.length;
    result.stats.hasTimestamp = colMap.timestamp != null && tsPresent > 0;
    result.readings = readings;

    // 预校验
    if (readings.length < 3) {
      result.errors.push(`至少需要 3 条读数才能做趋势判断，当前只有 ${readings.length} 条有效数据。`);
      return result;
    }

    // pH 范围校验（后端不校验，前端自己拦）
    const badPh = readings.filter((r) => r.ph != null && (r.ph < 0 || r.ph > 14));
    if (badPh.length) {
      result.warnings.push(`有 ${badPh.length} 条记录的 pH 不在 0–14 范围内（可能是量纲或单位错误），请核对。`);
    }
    // 溶氧异常值（>20 视为传感器错误）
    const badDo = readings.filter((r) => r.dissolved_oxygen != null && (r.dissolved_oxygen < 0 || r.dissolved_oxygen > 20));
    if (badDo.length) {
      result.warnings.push(`有 ${badDo.length} 条记录的溶氧超出 0–20 mg/L 合理范围（可能是传感器故障值），请核对。`);
    }

    result.ok = true;
    return result;
  }

  /**
   * 从文本（CSV/TSV/粘贴）解析
   */
  function parseText(text) {
    const delimiter = detectDelimiter(text);
    const rows = parseDelimited(text, delimiter);
    const r = buildReadings(rows);
    r.stats.delimiter = delimiter === "\t" ? "制表符(Tab)" : delimiter;
    r.stats.source = "text";
    return r;
  }

  /**
   * 从文件解析（自动判断 CSV/TSV/XLSX）
   */
  async function parseFile(file) {
    const name = (file.name || "").toLowerCase();
    if (name.endsWith(".xlsx") || name.endsWith(".xlsm")) {
      try {
        const buf = await file.arrayBuffer();
        const rows = await parseXlsx(buf);
        const r = buildReadings(rows);
        r.stats.source = "xlsx";
        return r;
      } catch (e) {
        return { ok: false, readings: [], columns: {}, errors: ["XLSX 解析失败：" + (e.message || e) + "。建议将 Excel 另存为 CSV 后上传。"], warnings: [], stats: {} };
      }
    }
    const text = await file.text();
    return parseText(text);
  }

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.parse = { parseText, parseFile, buildReadings, matchColumn, normName, toNumber };
})(window);
