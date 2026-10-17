/* ============================================================================
 * api.js —— 接口客户端（消费 12 个端点）
 *   · baseURL 默认同源（页面由接口服务托管时不用配任何东西）
 *   · 直接用文件双击打开时（file://）回落到 http://127.0.0.1:8000
 *   · 显式配置存 localStorage，可随时在页面左下角「设置」里改与清空
 *   · normalizeError：统一处理 detail 的三种形状（字符串 / 数组 / 对象）
 *   · 状态码 → 中文提示
 * ========================================================================== */
(function (global) {
  "use strict";

  const STORAGE_KEY = "fisheries_base_url";

  /** 用 file:// 直接打开页面时的默认后端地址。 */
  const FILE_FALLBACK_BASE = "http://127.0.0.1:8000";

  /**
   * 旧版本把「交付包里的 8887 mock 服务」地址写进了 localStorage。
   * 那个 mock 已经不随项目分发，同源部署下继续用它只会得到一个连不上的地址，
   * 所以这里显式识别并忽略掉，而不是让用户自己去设置里清。
   */
  const LEGACY_BASES = ["http://127.0.0.1:8787", "http://localhost:8787"];

  function isHttpPage() {
    const proto = (global.location && global.location.protocol) || "";
    return proto === "http:" || proto === "https:";
  }

  function detectBase() {
    const saved = (localStorage.getItem(STORAGE_KEY) || "").trim().replace(/\/+$/, "");
    if (saved) {
      if (isHttpPage() && LEGACY_BASES.indexOf(saved) !== -1) return "";
      return saved;
    }
    // 页面本身就是接口服务发出来的 ⇒ 用相对路径，不需要也不应该配地址。
    return isHttpPage() ? "" : FILE_FALLBACK_BASE;
  }

  const state = { baseURL: detectBase() };

  /** 空串表示同源；传空即清除本地覆盖。返回值是规范化后的地址。 */
  function setBase(url) {
    let u = (url || "").trim();
    if (u === "/" || u === "." || u === "(同源)") u = "";
    u = u.replace(/\/+$/, "");
    state.baseURL = u;
    if (u) localStorage.setItem(STORAGE_KEY, u);
    else localStorage.removeItem(STORAGE_KEY);
    return u;
  }

  /** 给人看的地址：同源时显示当前页面的来源。 */
  function describeBase() {
    if (state.baseURL) return state.baseURL;
    const origin = (global.location && global.location.origin) || "";
    return origin ? origin + "（与本页面同源）" : "(同源)";
  }

  /* ---------------- 错误归一化（8.1 三种 detail 形状） ---------------- */

  const STATUS_MSG = {
    400: "上传内容为空。",
    403: "服务器已禁用「按本地路径读图」，请改用图片上传。",
    404: "请求的资源不存在（如 upload_id 或图片路径、工具名、模板 key）。",
    409: "声明使用会话缓冲，但缓冲里没有读数。请先上传传感器数据，或直接传入 readings。",
    410: "上传的临时文件已被清理，请重新上传图片。",
    413: "上传文件超过大小上限（默认 10 MB）。",
    415: "上传类型不受支持（不是图片）。",
    422: "参数校验失败。",
    500: "后端执行失败，请稍后重试。",
    503: "知识库不可用，暂时无法问答。",
  };

  function normalizeError(status, body) {
    let message = "";
    let code = null;
    const detail = body && body.detail;

    if (detail === undefined || detail === null) {
      message = STATUS_MSG[status] || `请求失败（HTTP ${status}）`;
    } else if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      // FastAPI 422：取每项 msg，用 loc 拼出字段
      message = detail.map((d) => {
        const loc = (d.loc || []).filter((x) => x !== "body").join(".");
        const hint = global.Fisheries.dict.VALIDATION_HINTS[d.msg || d.type] || d.msg || d.type || "参数不合法";
        return loc ? `字段「${loc}」：${hint}` : hint;
      }).join("；");
    } else if (typeof detail === "object") {
      message = detail.error_message || detail.message || JSON.stringify(detail);
      code = detail.error_code || null;
    }

    if (!message) message = STATUS_MSG[status] || `请求失败（HTTP ${status}）`;
    return { message, code, status };
  }

  /** 抛出一个带中文 message 的 Error */
  function makeError(status, body) {
    const e = normalizeError(status, body);
    const err = new Error(e.message);
    err.code = e.code;
    err.status = status;
    err.normalized = e;
    return err;
  }

  /* ---------------- 底层请求 ---------------- */

  async function request(method, path, options) {
    options = options || {};
    const timeoutMs = options.timeoutMs != null ? options.timeoutMs : (path.includes("/assess") ? 60000 : 30000);
    const url = state.baseURL + path;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);

    const headers = Object.assign({}, options.headers || {});
    const init = { method, headers, signal: ctrl.signal };
    if (options.body != null) init.body = options.body;
    if (options.json != null) {
      init.body = JSON.stringify(options.json);
      headers["Content-Type"] = "application/json";
    }

    let resp;
    try {
      resp = await fetch(url, init);
    } catch (e) {
      clearTimeout(timer);
      if (e && e.name === "AbortError") {
        const err = new Error("请求超时（超过 " + Math.round(timeoutMs / 1000) + " 秒无响应）。评估接真实大模型时可能较慢，请稍后重试。");
        err.status = 0; err.isNetwork = true;
        throw err;
      }
      const err = new Error(
        "连不上后端服务 " + describeBase() +
        "。请先启动接口服务（双击 启动接口服务.bat，或运行 python -m fisheries.api），" +
        "确认真实地址后再点左下角「设置」核对。"
      );
      err.status = 0; err.isNetwork = true;
      throw err;
    }
    clearTimeout(timer);

    let body = null;
    const ct = resp.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      try { body = await resp.json(); } catch (e) { body = null; }
    } else {
      const txt = await resp.text();
      try { body = txt ? JSON.parse(txt) : null; } catch (e) { body = null; }
    }

    if (!resp.ok) {
      throw makeError(resp.status, body);
    }
    return body;
  }

  function qs(params) {
    if (!params) return "";
    const p = new URLSearchParams();
    Object.keys(params).forEach((k) => {
      if (params[k] != null && params[k] !== "") p.set(k, params[k]);
    });
    const s = p.toString();
    return s ? "?" + s : "";
  }

  /* ---------------- 端点 ---------------- */

  const api = {
    getBaseURL: () => state.baseURL,
    setBaseURL: setBase,
    describeBase: describeBase,
    isSameOrigin: () => !state.baseURL,

    health() { return request("GET", "/api/v1/health", { timeoutMs: 5000 }); },

    assess(payload) { return request("POST", "/api/v1/assess", { json: payload }); },

    toolsCall(payload) { return request("POST", "/api/v1/tools/call", { json: payload }); },

    ingestSensor(payload) { return request("POST", "/api/v1/ingest/sensor", { json: payload }); },

    pondReadings(pondId, limit) { return request("GET", "/api/v1/ponds/" + encodeURIComponent(pondId) + "/readings" + qs({ limit })); },

    pondHistory(pondId, limit) { return request("GET", "/api/v1/ponds/" + encodeURIComponent(pondId) + "/history" + qs({ limit })); },

    ingestImage(file) {
      const fd = new FormData();
      fd.append("file", file);
      return request("POST", "/api/v1/ingest/image", { body: fd });
    },

    /** 鱼种识别是否可用。前端据此显示或隐藏「品种识别」模块。 */
    speciesAvailable() { return request("GET", "/api/v1/species", { timeoutMs: 8000 }); },

    speciesDetect(file) {
      const fd = new FormData();
      fd.append("file", file);
      return request("POST", "/api/v1/species/detect", { body: fd });
    },

    chat(payload) { return request("POST", "/api/v1/chat", { json: payload }); },

    thresholds() { return request("GET", "/api/v1/config/thresholds"); },

    tools() { return request("GET", "/api/v1/config/tools"); },

    templates() { return request("GET", "/api/v1/config/templates"); },

    /** 模板 CSV 的直链。用 <a download> 交给浏览器下载，不走 fetch。 */
    templateCsvUrl(key) {
      return state.baseURL + "/api/v1/config/templates/" + encodeURIComponent(key) + "/csv";
    },
  };

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.api = api;
  global.Fisheries.normalizeError = normalizeError;
})(window);
