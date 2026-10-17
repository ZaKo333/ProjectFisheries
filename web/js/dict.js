/* ============================================================================
 * dict.js —— 枚举字典（后端返回英文/代码，中文由前端配置）
 * 依据《前端要求》第 6 章逐条实现，code 逐字对齐。
 * ========================================================================== */
(function (global) {
  "use strict";

  /** 风险等级三档 → 颜色 / 徽章样式 */
  const RISK_LEVELS = {
    "正常": { color: "#1e8449", cls: "lv-正常", badge: "badge-ok", icon: "✓" },
    "预警": { color: "#d9822b", cls: "lv-预警", badge: "badge-warn", icon: "!" },
    "高危": { color: "#c0392b", cls: "lv-高危", badge: "badge-danger", icon: "✕" },
  };

  /**
   * 6.2 rule_hits 规则码（18 个，一个不能漏）
   * { label, severity }  severity: 1 危险 / 0.5 警戒 / 0 提示
   */
  const RULE_HITS = {
    DO_BELOW_OPTIMAL:          { label: "溶氧低于适宜下限", severity: 0.5 },
    DO_REDLINE:                { label: "溶氧跌破泛塘临界值（1.5 mg/L）", severity: 1 },
    DO_REDLINE_PROJECTED:      { label: "预计溶氧将跌破泛塘临界值", severity: 1 },
    DO_CRASH_TREND:            { label: "溶氧短期内骤降", severity: 1 },
    DO_TREND_DOWN_FAST:        { label: "溶氧下降过快", severity: 0.5 },
    NH3N_ABOVE_WARNING:        { label: "氨氮超过警戒值", severity: 0.5 },
    NH3N_TREND_UP_FAST:        { label: "氨氮上升过快", severity: 0.5 },
    NH3_REDLINE:               { label: "氨氮触及红线（1.0 mg/L）", severity: 1 },
    UNIONIZED_AMMONIA_REDLINE: { label: "非离子氨（毒性氨）超限（0.02 mg/L）", severity: 1 },
    UNIONIZED_AMMONIA_RISK:    { label: "非离子氨逼近限值（0.01–0.02 mg/L）", severity: 0.5 },
    NO2N_ABOVE_WARNING:        { label: "亚硝酸盐超过警戒值", severity: 0.5 },
    NO2N_TREND_UP_FAST:        { label: "亚硝酸盐上升过快", severity: 0.5 },
    NO2_REDLINE:               { label: "亚硝酸盐触及红线（0.5 mg/L）", severity: 1 },
    PH_OUT_OF_OPTIMAL:         { label: "pH 偏离适宜区间", severity: 0.5 },
    PH_REDLINE:                { label: "pH 触及红线（<5.5 或 >9.5）", severity: 1 },
    PH_FLUCTUATION_HIGH:       { label: "pH 波动过大", severity: 0.5 },
    TEMP_OUT_OF_OPTIMAL:       { label: "水温偏离适宜区间", severity: 0.5 },
    TEMP_FLUCTUATION_HIGH:     { label: "水温波动过大", severity: 0.5 },
  };

  /**
   * 6.3 redline_hits 红线码
   *
   * 这里是 `water_risk.py::_check_redlines` 实际会发出的**全部 7 个码**，一个不多一个不少。
   * 曾经漏过三个（DO_REDLINE_PROJECTED / DO_CRASH_TREND / PH_REDLINE）、多了两个
   * （PH_LOW_REDLINE / PH_HIGH_REDLINE —— 它们只是后端 REDLINE 阈值表的字典键，
   * pH 越界时后端发的是 PH_REDLINE），结果那三个码在界面上直接显示英文。
   * `UNIONIZED_AMMONIA_REDLINE` 不在后端 REDLINE 阈值表里（表里的键按总氨氮 mg/L 存，
   * 这条按换算后的非离子氨 0.02 mg/L 判），所以这里必须单独登记。
   */
  const REDLINES = {
    DO_REDLINE:           { label: "溶氧跌破泛塘临界值", value: "1.5 mg/L" },
    DO_REDLINE_PROJECTED: { label: "按趋势外推，溶氧将跌破泛塘临界值", value: "1.5 mg/L" },
    DO_CRASH_TREND:       { label: "溶氧短期内骤降", value: "3 h 内降幅 ≥ 2.0 mg/L" },
    NH3_REDLINE:          { label: "氨氮触及红线", value: "1.0 mg/L" },
    UNIONIZED_AMMONIA_REDLINE: { label: "非离子氨（毒性氨）超限", value: "0.02 mg/L（GB 11607-1989）" },
    NO2_REDLINE:          { label: "亚硝酸盐触及红线", value: "0.5 mg/L" },
    PH_REDLINE:           { label: "pH 触及红线", value: "低于 5.5 或高于 9.5" },
  };

  /**
   * 6.4 融合层 code → 中文（alerts[].code / evidence[].code / synergy）
   * 返回 { label, detail? }。detail 用于在不知道具体文本时兜底说明。
   */
  const FUSION_CODES = {
    MODALITY_UNAVAILABLE:    { label: "某一路模态失败" },
    SEVERE_DISEASE:          { label: "高致病性病害" },
    LEVEL_FLOOR:             { label: "等级被下限抬升" },
    DO_ALERT_ETA:            { label: "溶氧预计跌破阈值" },
    VISION_HEALTHY:          { label: "图像未检出明显病害" },
    VISION_DISEASE:          { label: "图像检出病害" },
    VISION_LOW_CONF:         { label: "图像置信度偏低，建议人工复核" },
    VISION_RAW_CLASS:        { label: "图像原始类别" },
    SYN_BACTERIAL_ON_WARNING:{ label: "细菌性病害 + 水质预警（协同 +12）" },
    SYN_NITROGEN_WITH_DISEASE:{ label: "氮素超标 + 病害（协同 +8）" },
    SYN_LOW_DO_WITH_GILL:    { label: "低溶氧 + 烂鳃（协同 +15）" },
    SYN_SAPRO_ON_COLD:       { label: "低温 + 水霉（协同 +10）" },
  };

  /** SEV_<指标大写> 的前缀解析 */
  const SEV_PREFIX = {
    DISSOLVED_OXYGEN: "溶氧",
    AMMONIA_N: "氨氮",
    NITRITE_N: "亚硝酸盐",
    PH: "pH",
    WATER_TEMP: "水温",
  };

  /** alerts[].level 枚举 → 配色 */
  const ALERT_LEVELS = {
    critical: { label: "高危", cls: "critical", badge: "badge-danger" },
    warning:  { label: "预警", cls: "warning", badge: "badge-warn" },
    info:     { label: "信息", cls: "info", badge: "badge-info" },
  };
  /** 排序权重（critical > warning > info） */
  const ALERT_ORDER = { critical: 0, warning: 1, info: 2 };

  /** 6.4 fusion.level_floors 取值 → 中文 */
  const LEVEL_FLOORS = {
    "REDLINE":            "命中水质红线",
    "SEVERE_DISEASE":     "检出高致病性病害",
    "WATER_LEVEL:高危":   "水质工具自身判定为高危",
    "WATER_LEVEL:预警":   "水质工具自身判定为预警",
    "DISEASE_DETECTED":   "检出病害（严重度 ≥0.70 且置信度 ≥0.40）",
  };

  /** 6.5 severity_map：指标英文键 → 中文标签 + 单位 */
  const INDICATORS = {
    dissolved_oxygen: { label: "溶氧", unit: "mg/L" },
    ammonia_n:        { label: "氨氮", unit: "mg/L" },
    nitrite_n:        { label: "亚硝酸盐", unit: "mg/L" },
    ph:               { label: "pH", unit: "" },
    water_temp:       { label: "水温", unit: "℃" },
  };
  const SEVERITY_LABEL = { 0: "正常", 0.5: "警戒", 1: "危险" };

  /** 6.6 病害标签分组（后端 `DISEASE_LABELS` 的 5 个标签一一对应） */
  const DISEASE_GROUPS = {
    "烂鳃": "细菌性",
    "出血": "细菌性",
    "水霉": "真菌性",
    "健康": "健康",
    "其他": "其他",
  };

  /** 6.7 trace[].kind → 图标 + 中文名 */
  const TRACE_KINDS = {
    llm:          { label: "大模型决策", icon: "🧠" },
    tool_call:    { label: "调用工具", icon: "🔧" },
    tool_result:  { label: "工具返回", icon: "📦" },
    guardrail:    { label: "护栏介入", icon: "🛡" },
    fusion:       { label: "融合定级", icon: "🧮" },
    advisor:      { label: "生成建议", icon: "📝" },
    error:        { label: "出错", icon: "⛔" },
  };

  /** 证据来源 → 中文名 + 颜色 */
  const EVIDENCE_SOURCES = {
    water:  { label: "水质", color: "#2e86de" },
    vision: { label: "图像", color: "#8e44ad" },
    fusion: { label: "融合", color: "#e67e22" },
  };

  /**
   * 工具层 / 调度层错误码 → 中文
   *
   * 只覆盖图像工具自己发的那 3 个码是不够的：`/assess` 里的图像走的是工具调用，
   * 调度层的 TOOL_CRASHED / BAD_ARGUMENTS / UNKNOWN_TOOL 与水质的
   * INVALID_INPUT / COMPUTE_FAILED 都可能出现在 `vision.error_code` 上，
   * 漏一个就会在界面上看到英文码。
   */
  const TOOL_ERRORS = {
    IMAGE_NOT_FOUND:   "找不到图片文件",
    NOT_A_FILE:        "上传的不是文件",
    UNSUPPORTED_IMAGE: "无法识别的图片格式",
    INVALID_INPUT:     "输入数据不合法（缺必需指标、点数不足，或含 NaN / inf）",
    COMPUTE_FAILED:    "计算失败，请检查数据后重试",
    UNKNOWN_TOOL:      "调用了未注册的工具",
    BAD_ARGUMENTS:     "工具参数不合法",
    TOOL_CRASHED:      "工具执行时抛出异常",
    DUPLICATE_CALL:    "同一工具在一次评估里被重复调用",
    UNKNOWN:           "原因未知",
  };

  /** FastAPI 422 常见 msg → 中文说明表（8.1） */
  const VALIDATION_HINTS = {
    "String should have at least 2 characters": "字符串长度不足（至少 2 个字符）",
    "value_error": "字段取值不合法",
    "string_too_short": "字符串长度不足",
    "greater_than_equal": "数值小于允许的最小值",
    "less_than_equal": "数值大于允许的最大值",
  };

  const dict = {
    RISK_LEVELS,
    RULE_HITS,
    REDLINES,
    FUSION_CODES,
    SEV_PREFIX,
    ALERT_LEVELS,
    ALERT_ORDER,
    LEVEL_FLOORS,
    INDICATORS,
    SEVERITY_LABEL,
    DISEASE_GROUPS,
    TRACE_KINDS,
    EVIDENCE_SOURCES,
    TOOL_ERRORS,
    VALIDATION_HINTS,
  };

  /** 把任意 code 转成「中文标签 + 严重度」，未知 code 原样返回（绝不显示 undefined） */
  dict.ruleLabel = function (code) {
    const r = RULE_HITS[code];
    return r ? r.label : (code || "—");
  };
  dict.ruleSeverity = function (code) {
    const r = RULE_HITS[code];
    return r ? r.severity : 0;
  };
  dict.redlineLabel = function (code) {
    const r = REDLINES[code];
    if (r) return r.value ? `${r.label}（${r.value}）` : r.label;
    // 后端将来加了红线码、这边还没登记时，至少借规则表的中文名，
    // 不要把英文码直接甩到界面上。
    if (RULE_HITS[code]) return RULE_HITS[code].label;
    return code || "—";
  };
  dict.fusionCodeLabel = function (code) {
    if (!code) return "—";
    const c = FUSION_CODES[code];
    if (c) return c.label;
    if (code.startsWith("SEV_")) {
      const key = code.slice(4);
      const ind = SEV_PREFIX[key];
      return ind ? `${ind} 处于警戒/危险区间` : code;
    }
    const r = REDLINES[code];
    if (r) return r.label;
    return code;
  };
  dict.indicatorLabel = function (key) {
    const i = INDICATORS[key];
    return i ? i.label : (key || "—");
  };
  dict.indicatorUnit = function (key) {
    const i = INDICATORS[key];
    return i ? i.unit : "";
  };
  dict.diseaseGroup = function (t) { return DISEASE_GROUPS[t] || t || "—"; };
  dict.severityLabel = function (s) {
    return SEVERITY_LABEL[s] !== undefined ? SEVERITY_LABEL[s] : (s == null ? "—" : String(s));
  };
  dict.errorLabel = function (code) { return TOOL_ERRORS[code] || code || "未知错误"; };

  global.Fisheries = global.Fisheries || {};
  global.Fisheries.dict = dict;
})(window);
