"""Agent 工具声明（Function Calling Schema）。

本文件是"给大模型看的说明书"，不包含任何业务逻辑。
格式遵循 OpenAI / DeepSeek 通用的 tools 参数规范：

    [{"type": "function",
      "function": {"name": ..., "description": ..., "parameters": <JSON Schema>}}]

设计要点
--------
1. description 必须写清"什么时候该调用我、什么时候不该调用"，否则模型会乱调；
2. parameters 用 JSON Schema 描述，字段名统一 snake_case，与 Python handler 形参一致；
3. 数组类入参一律用「对象数组」而非「数组的数组」——模型生成 [[28,7.2,...]] 的出错率远高于
   [{"water_temp":28,...}]；
4. 所有阈值/等级枚举值直接写进 schema 的 enum，减少模型自由发挥空间。
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# 工具 1：water_risk_predict
# --------------------------------------------------------------------------- #
WATER_RISK_PREDICT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "water_risk_predict",
        "description": (
            "【水质时序风险预测】输入一段按时间升序排列的池塘水质监测序列"
            "（水温、pH、溶氧、氨氮、亚硝酸盐），输出综合风险等级、各项指标变化趋势，"
            "以及对未来一段时间的恶化预测描述。\n"
            "适用场景：用户提供传感器数据或水质 CSV；需要判断当前水质是否安全、"
            "是否需要提前增氧/换水/停料。\n"
            "不适用场景：只问养殖知识而不涉及具体数值（此时应查知识库）；"
            "只看图片（此时应调用 fish_disease_detect）。\n"
            "注意：readings 必须按时间升序，至少 3 条；建议 ≥24 条以获得可靠趋势。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pond_id": {
                    "type": "string",
                    "description": "池塘/塘口编号，用于在多塘场景下区分告警来源，例如 'POND-03'。未知时填 'unknown'。",
                },
                "readings": {
                    "type": "array",
                    "minItems": 3,
                    "description": (
                        "按时间升序排列的水质记录数组，至少 3 条，建议 ≥24 条（约 1 天，1 小时/条）。"
                        "每条至少包含 water_temp / ph / dissolved_oxygen / ammonia_n 四项。"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {
                                "type": "string",
                                "description": "ISO 8601 时间戳，如 '2026-10-14T08:00:00+08:00'。可选；缺失时按 interval_minutes 等间隔处理。",
                            },
                            "water_temp": {
                                "type": "number",
                                "description": "水温，单位 ℃。",
                            },
                            "ph": {
                                "type": "number",
                                "description": "pH 值，无量纲，淡水养殖正常范围 6.5–8.5。",
                            },
                            "dissolved_oxygen": {
                                "type": "number",
                                "description": "溶解氧，单位 mg/L。",
                            },
                            "ammonia_n": {
                                "type": "number",
                                "description": "氨氮（以 N 计），单位 mg/L。",
                            },
                            "nitrite_n": {
                                "type": ["number", "null"],
                                "description": (
                                    "亚硝酸盐氮（以 N 计），单位 mg/L。"
                                    "若该塘口未安装亚硝酸盐探头，可传 null 或省略，"
                                    "此时该指标不参与打分，其余指标权重自动归一化。"
                                ),
                            },
                        },
                        "required": [
                            "water_temp",
                            "ph",
                            "dissolved_oxygen",
                            "ammonia_n",
                        ],
                    },
                },
                "interval_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 60,
                    "description": "相邻两条记录的采样间隔（分钟）。若每条记录都带 timestamp，本参数会被自动推断值覆盖。",
                },
                "horizon_hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 168,
                    "default": 72,
                    "description": "预测窗口长度（小时），表示'未来多久内会不会出问题'。默认 72 小时（3 天）。",
                },
                "species": {
                    "type": "string",
                    "enum": ["鲤科", "罗非鱼", "虾蟹"],
                    "default": "鲤科",
                    "description": "养殖品种，决定阈值分组。鲤科（草鱼/鲫鱼/鲤鱼等）、罗非鱼、虾蟹（南美白对虾/河蟹等）。不确定时用默认值 '鲤科'。",
                },
            },
            "required": ["readings"],
        },
    },
}


# --------------------------------------------------------------------------- #
# 工具 2：fish_disease_detect
# --------------------------------------------------------------------------- #
FISH_DISEASE_DETECT_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fish_disease_detect",
        "description": (
            "【鱼体病害图像识别】输入一张鱼体（或鳃丝、体表病灶）图片的路径，"
            "调用 YOLO 分类模型识别病害类型，输出病害名称、置信度及概率分布。\n"
            "可识别类别：烂鳃、出血（细菌性败血症）、水霉、健康；"
            "此外底层 7 分类模型中不属于上述三类的病灶（赤皮/寄生虫/病毒性白尾）"
            "会归入 '其他'，并在 raw_class 字段保留原始判定，供人工复核。\n"
            "适用场景：用户上传鱼体照片并希望判断病症。\n"
            "不适用场景：只有传感器数值没有图片（此时应调用 water_risk_predict）。\n"
            "注意：单张图片只调用一次；若置信度低于阈值，返回值中 is_low_confidence 为 true，"
            "应向用户说明'图片不够清晰/需要人工复核'，不要强行下结论。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "鱼体图片的本地绝对路径（如 'D:\\\\data\\\\fish\\\\gill_01.jpg'）或 http(s) 图片 URL。",
                },
                "conf_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.25,
                    "description": "最低置信度阈值。低于该值仍会返回结果，但会将 is_low_confidence 置为 true。",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 4,
                    "description": "返回概率最高的前 K 个类别，用于展示模型的犹豫程度。",
                },
            },
            "required": ["image_path"],
        },
    },
}


#: 工具名 → schema。Agent 调度器把本列表直接塞进 LLM 请求的 tools 字段。
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "water_risk_predict": WATER_RISK_PREDICT_SCHEMA,
    "fish_disease_detect": FISH_DISEASE_DETECT_SCHEMA,
}


def get_tool_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """获取工具声明列表。

    Args:
        names: 需要返回的工具名列表；传 None 表示返回全部已注册工具。

    Returns:
        可直接作为 LLM 请求 `tools=` 参数的列表。

    Raises:
        KeyError: 请求了未注册的工具名。

    Examples:
        >>> schemas = get_tool_schemas(["water_risk_predict"])
        >>> schemas[0]["function"]["name"]
        'water_risk_predict'
    """
    if names is None:
        return list(TOOL_SCHEMAS.values())
    unknown = [n for n in names if n not in TOOL_SCHEMAS]
    if unknown:
        raise KeyError(f"未注册的工具：{unknown}；已注册：{sorted(TOOL_SCHEMAS)}")
    return [TOOL_SCHEMAS[n] for n in names]
