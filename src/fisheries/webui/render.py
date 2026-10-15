"""把决策结果排成页面能直接显示的文本。

这里只做「排版」，不做任何判断：等级的取色、告警的排序、数字的进位
都按 Agent 给的结果原样呈现。**一个数字都不许在这里算**——
一旦页面自己算分，演示时它和报告就会各说各话。

两类输出分工明确：

* ``render_*_markdown()`` 给 ``gr.Markdown``，写结构化文字；
* :func:`render_alerts` 给 ``gr.HTML``——因为 ``gr.Markdown`` 默认会
  ``sanitize_html``，带内联样式的告警卡片会被它清掉色。
"""

from __future__ import annotations

from html import escape
from typing import Any

#: 等级 -> (颜色, 图标)。配色跟报告里的用词一致：红=马上处理、橙=今天盯紧。
LEVEL_STYLE: dict[str, tuple[str, str]] = {
    "高危": ("#c0392b", "🔴"),
    "预警": ("#d9822b", "🟠"),
    "正常": ("#1e8449", "🟢"),
}

_ALERT_STYLE: dict[str, tuple[str, str]] = {
    "critical": ("#fdecea", "#c0392b", "严重"),
    "warning": ("#fdf3e3", "#d9822b", "注意"),
    "info": ("#eef4fb", "#2c6fb0", "提示"),
}

#: 融合层指标的中文名，给「算法依据」表用。
FUSION_LABELS: dict[str, str] = {
    "water_score": "水质模态分",
    "vision_score": "图像模态分",
    "weighted_score": "加权公式值",
    "fused_score": "最终融合分",
    "score_floor": "等级下限分数线",
    "synergy_score": "协同加成",
}

#: 等级下限的 ID -> 人话。与 ``fusion.py`` 的 ``level_floors`` 取值一一对应。
FLOOR_LABELS: dict[str, str] = {
    "REDLINE": "水质红线（溶氧泛塘线 / 氨氮或亚硝酸盐红线）",
    "SEVERE_DISEASE": "高致病性病害（高严重度 + 高置信度）",
    "WATER_LEVEL:高危": "水质工具自身判决为高危",
    "WATER_LEVEL:预警": "水质工具自身判决为预警",
    "DISEASE_DETECTED": "检出病害（需启动确诊流程）",
}

#: 工具层 ``severity_map`` 的指标名 -> 中文。
INDICATOR_LABELS: dict[str, str] = {
    "water_temp": "水温",
    "ph": "pH",
    "dissolved_oxygen": "溶氧",
    "ammonia_n": "氨氮",
    "nitrite_n": "亚硝酸盐",
}


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    """宽容地格式化一个可能是 None / 字符串 / 数字的值。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _severity_zh(value: Any) -> str:
    """把 ``severity_map`` 的 0 / 0.5 / 1.0 翻成中文档位。"""
    if value is None:
        return "未评估"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number >= 1.0:
        return "危险"
    if number >= 0.5:
        return "警戒"
    return "适宜"


# --------------------------------------------------------------------------- #
# 顶部结论
# --------------------------------------------------------------------------- #

def render_summary_markdown(result: Any) -> str:
    """顶部大结论：等级、分数、一句话、理由。"""
    if not getattr(result, "ok", False):
        return ""

    level = result.risk_level or "未知"
    color, icon = LEVEL_STYLE.get(level, ("#5d6d7e", "⚪"))
    lines = [
        f"## {icon} 风险等级：<span style='color:{color}'>{escape(level)}</span>",
        f"**融合分 {_fmt(result.fused_score, 1)} / 100**　"
        f"（{escape(str(result.pond_id or '未命名塘口'))}　"
        f"{escape(str(result.species or '鲤科'))}）",
    ]
    if result.headline:
        lines.append(f"### {escape(result.headline)}")
    if result.reason:
        lines.append(f"> {escape(result.reason)}")
    return "\n\n".join(lines)


def render_alerts(result: Any) -> str:
    """告警卡片（HTML）。没有告警时返回一句绿色说明，而不是空白。"""
    if not getattr(result, "ok", False):
        return ""

    alerts = list(result.alerts or [])
    if not alerts:
        return (
            "<div style='padding:10px 12px;border-radius:8px;background:#eaf7ee;"
            "border-left:4px solid #1e8449;font-size:14px'>"
            "本次没有产生告警。规则引擎只在水质越限、命中红线或检出病害时告警。</div>"
        )

    # critical 排前面：现场演示时最该先看到的就得在最上面。
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: order.get(str(a.get("level", "info")), 3))

    cards: list[str] = []
    for alert in alerts:
        level = str(alert.get("level", "info"))
        bg, border, tag = _ALERT_STYLE.get(level, _ALERT_STYLE["info"])
        title = escape(str(alert.get("title", "")).strip() or "告警")
        detail = escape(str(alert.get("detail", "")).strip())
        code = escape(str(alert.get("code", "")))
        cards.append(
            f"<div style='padding:10px 12px;margin-bottom:8px;border-radius:8px;"
            f"background:{bg};border-left:4px solid {border}'>"
            f"<div style='font-weight:600;font-size:14px;color:#1c2833'>"
            f"[{tag}] {title}</div>"
            f"<div style='font-size:13px;color:#3c4a58;margin-top:4px'>{detail}</div>"
            f"<div style='font-size:11px;color:#93a1ad;margin-top:4px;"
            f"font-family:ui-monospace,Menlo,Consolas,monospace'>{code}</div>"
            "</div>"
        )
    heading = (
        f"<div style='font-weight:600;font-size:15px;margin-bottom:6px'>"
        f"告警 {len(alerts)} 条</div>"
    )
    return heading + "".join(cards)


# --------------------------------------------------------------------------- #
# 养殖建议
# --------------------------------------------------------------------------- #

def _actions_table(actions: list[dict[str, Any]]) -> str:
    """把一组建议排成 markdown 表。"""
    if not actions:
        return "_（本次没有这类建议）_\n"
    rows = [
        "| 优先级 | 对象 | 怎么做 | 什么时候 |",
        "| --- | --- | --- | --- |",
    ]
    for item in sorted(actions, key=lambda a: int(a.get("priority", 9))):
        rows.append(
            "| P{prio} | {target} | {action} | {timing} |".format(
                prio=escape(str(item.get("priority", ""))),
                target=escape(str(item.get("target", ""))),
                action=escape(str(item.get("action", ""))),
                timing=escape(str(item.get("timing", "")) or "—"),
            )
        )
    return "\n".join(rows) + "\n"


def render_report(result: Any) -> str:
    """养殖建议主体：水质动作与病害动作**分开两块**。"""
    if not getattr(result, "ok", False):
        return ""

    parts: list[str] = ["## 处置建议"]
    parts.append("### 水质调控（调水、增氧、投料）\n")
    parts.append(_actions_table(list(result.water_actions or [])))
    parts.append("\n### 病害处置（捞鱼、用药、隔离）\n")
    parts.append(_actions_table(list(result.disease_actions or [])))

    if result.observation_window:
        parts.append(
            f"\n### 观察窗口\n\n{escape(result.observation_window)}\n"
        )
    if result.confidence_note:
        parts.append(f"\n> 可信度说明：{escape(result.confidence_note)}\n")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 输入回显：水质表摄入质量 + 图片识别结果
# --------------------------------------------------------------------------- #

def render_data_quality(result: Any) -> str:
    """水质侧「数据接入质量」面板。"""
    if not getattr(result, "ok", False):
        return ""
    ingest = getattr(result, "ingest", None)
    if ingest is None:
        return "_本次没有上传水质表格。_\n"

    parts: list[str] = ["## 水质数据接入\n", ingest.summary_markdown()]
    water = getattr(result, "water", None)
    if getattr(result, "water_error", ""):
        parts.append(f"\n> ⚠️ **水质工具没跑成功**：{escape(result.water_error)}")
    elif water:
        parts.append("\n### 水质指标现状\n")
        parts.append(_water_table(water))
        if water.get("degradation_summary"):
            parts.append(f"\n> {escape(str(water['degradation_summary']))}\n")
    return "\n".join(parts)


def _water_table(water: dict[str, Any]) -> str:
    """水质工具返回的指标现状表。"""
    rows = ["| 指标 | 当前值 | 档位 | 工具判定 |", "| --- | --- | --- | --- |"]
    severity = water.get("severity_map") or {}
    trends = water.get("trends") or {}
    for key, label in INDICATOR_LABELS.items():
        trend = trends.get(key) or {}
        current = trend.get("current")
        unit = trend.get("unit") or ""
        rows.append(
            "| {label} | {current} | {level} | {note} |".format(
                label=label,
                current=escape(_fmt(current, 2, f" {unit}".rstrip())),
                level=_severity_zh(severity.get(key)),
                note=escape(str(trend.get("direction_label") or trend.get("direction") or "—")),
            )
        )
    return "\n".join(rows) + "\n"


def render_vision(result: Any) -> str:
    """图像识别结果面板。"""
    if not getattr(result, "ok", False):
        return ""
    vision = getattr(result, "vision", None)
    if vision is None:
        return "_本次没有上传鱼体图片，结论只基于水质单模态。_\n"
    if getattr(result, "vision_error", ""):
        return (
            "> ⚠️ **鱼体图片分析失败**："
            f"{escape(result.vision_error)}\n\n"
            "结论**没有**包含该模态证据，请把它当作水质单模态结果看。\n"
        )

    healthy = bool(vision.get("is_healthy"))
    parts = ["## 鱼体图片识别\n"]
    parts.append(
        "| 检出 | 归属 | 置信度 | 严重度权重 | 是否需复核 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| {d} | {g} | {c} | {s} | {low} |\n".format(
            d=escape(str(vision.get("disease_type", "—"))),
            g=escape(str(vision.get("disease_group", "—"))),
            c=_fmt(vision.get("confidence"), 3),
            s=_fmt(vision.get("severity_weight"), 2),
            low="需要" if vision.get("is_low_confidence") else "不需要",
        )
    )
    scores = vision.get("all_scores") or {}
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: float(kv[1] or 0), reverse=True)
        parts.append("\n**各标签概率**：" + "　".join(
            f"`{escape(str(k))}` {_fmt(v, 3)}" for k, v in ranked
        ))
    if healthy:
        parts.append("\n> 本次未检出病害。注意：**照片上看着正常不等于水质没问题**。\n")
    if vision.get("degraded_mode"):
        parts.append(
            f"\n> ⚠️ 识别处于**降级模式**：{escape(str(vision.get('note') or '未加载真实权重'))}"
            "——标签来自可复现的模拟，不能用于精度评价。\n"
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 算法依据与推理链路
# --------------------------------------------------------------------------- #

def render_fusion(result: Any) -> str:
    """融合层依据表。**这是演示时最该讲清楚的一页**。"""
    if not getattr(result, "ok", False):
        return ""
    fusion = result.fusion or {}
    if not fusion:
        return ""

    weights = fusion.get("effective_weights") or {}
    parts = ["## 融合依据\n"]

    rows = ["| 项目 | 数值 |", "| --- | --- |"]
    for key in ("weighted_score", "score_floor", "fused_score", "synergy_score"):
        if key in fusion:
            rows.append(f"| {FUSION_LABELS.get(key, key)} | {_fmt(fusion.get(key), 2)} |")
    for name, weight in weights.items():
        rows.append(f"| 有效权重（{escape(str(name))}） | {_fmt(weight, 2)} |")
    parts.append("\n".join(rows) + "\n")

    floors = list(fusion.get("level_floors") or [])
    if floors:
        parts.append("\n**触发的等级下限**（融合等级不得低于任一单模态自身等级）：\n")
        for floor in floors:
            parts.append(f"- `{escape(str(floor))}` — {FLOOR_LABELS.get(str(floor), '—')}")
        parts.append("")
    else:
        parts.append("\n本次没有触发等级下限，融合等级完全由加权分决定。\n")

    if fusion.get("score_floor") is not None:
        parts.append(
            f"\n> 分数被下限抬升过：加权公式算出 {_fmt(fusion.get('weighted_score'), 2)}，"
            f"抬升到 {_fmt(fusion.get('score_floor'), 2)}。"
            "**这是一种保护**——否则「照片上鱼看着还行」会把水质高危稀释成预警。\n"
        )

    synergy_hits = fusion.get("synergy_hits") or []
    if synergy_hits:
        parts.append("\n**命中的协同规则**：" + "、".join(
            f"`{escape(str(h))}`" for h in synergy_hits
        ) + "\n")

    modalities = fusion.get("modalities") or []
    failed = fusion.get("failed_modalities") or []
    parts.append(f"\n**参与定级的模态**：{'、'.join(escape(str(m)) for m in modalities) or '无'}")
    if failed:
        parts.append(
            f"　**失败模态**：{'、'.join(escape(str(m)) for m in failed)}"
            "（结论未包含其证据）"
        )
    return "\n".join(parts) + "\n"


def render_trace(result: Any) -> str:
    """五步推理链路，折叠在最后给评委看「Agent 到底做了什么」。"""
    if not getattr(result, "ok", False):
        return ""
    trace = list(result.trace or [])
    if not trace:
        return "_没有推理记录。_\n"

    kind_icon = {
        "llm": "🧠",
        "tool_call": "🔧",
        "tool_result": "📥",
        "guardrail": "🛡️",
        "fusion": "⚖️",
        "advisor": "📝",
        "error": "❗",
    }
    lines: list[str] = []
    for step in trace:
        icon = kind_icon.get(str(step.get("kind", "")), "•")
        lines.append(
            f"**{step.get('step', '?')}. {icon} {escape(str(step.get('title', '')))}**"
        )
        detail = str(step.get("detail", "") or "").strip()
        if detail:
            lines.append(f"　{escape(detail)}")
        lines.append("")
    return "\n".join(lines)


def render_knowledge(result: Any) -> str:
    """RAG 引用与知识库状态。"""
    if not getattr(result, "ok", False):
        return ""

    summary = result.knowledge_summary or {}
    parts: list[str] = ["## 知识库引用\n"]
    if not summary:
        parts.append(
            "_本次没有走知识库_（未开启、或向量库不可用）。"
            "建议内容来自规则模板，口径仍与知识库一致。\n"
        )
        return "\n".join(parts)

    parts.append(
        f"- 检索引擎：`{escape(str(summary.get('backend', '—')))}`\n"
        f"- 引用片段数：{summary.get('n_chunks', 0)}\n"
    )
    purposes = summary.get("purposes") or {}
    if purposes:
        parts.append(
            "- 各用途命中：" + "、".join(
                f"{escape(str(k))} {v} 块" for k, v in purposes.items()
            ) + "\n"
        )
    if summary.get("degraded"):
        parts.append(
            "- ⚠️ 检索处于降级状态：" + escape("；".join(summary.get("warnings") or []) or "未知原因")
            + "\n"
        )

    references = list(result.references or [])
    if references:
        parts.append("\n**引用来源**\n")
        for ref in references:
            parts.append(f"- {escape(str(ref))}")
        parts.append("")
    return "\n".join(parts)


def render_status(result: Any) -> str:
    """页脚：耗时、后端、降级原因。让「不可信」这件事一眼可见。"""
    if not getattr(result, "ok", False):
        return ""

    backends = result.llm_backends or {}
    bits = [
        f"耗时 {_fmt(result.elapsed_seconds, 2)} 秒",
        f"调度模型 `{escape(str(backends.get('scheduler', '—')))}`",
        f"建议模型 `{escape(str(backends.get('advisor', '—')))}`",
    ]
    if result.degraded_mode:
        reasons = "；".join(str(r) for r in (result.degraded_reasons or [])) or "未说明"
        bits.append(f"⚠️ **降级运行**：{escape(reasons)}")
    return "　|　".join(bits)


def render_error(message: str) -> str:
    """统一的红色错误块。"""
    return (
        "<div style='padding:12px 14px;border-radius:8px;background:#fdecea;"
        "border-left:4px solid #c0392b'>"
        "<div style='font-weight:600;color:#c0392b;margin-bottom:6px'>运行失败</div>"
        f"<pre style='margin:0;white-space:pre-wrap;font-size:12px;color:#3c4a58'>"
        f"{escape(message)}</pre></div>"
    )
