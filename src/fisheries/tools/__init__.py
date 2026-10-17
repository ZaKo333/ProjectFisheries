"""Agent 工具层。

本包对外提供两样东西：
    1. TOOL_SCHEMAS  —— 给大模型看的函数声明（JSON Schema，function calling 用）
    2. TOOL_HANDLERS —— 真正执行的 Python 函数（名称与 schema 一一对应）

分开放置的原因：schema 是"契约"，handler 是"实现"。Agent 调度器只依赖契约，
因此替换实现（如规则基线 → LSTM 模型）不需要改动调度逻辑。
"""

from .definitions import TOOL_SCHEMAS, get_tool_schemas
from .fish_disease import fish_disease_detect
from .water_risk import water_risk_predict

#: 工具名 → 可执行函数 的派发表。Agent 调度器据此把模型的 tool_call 落到真实代码。
TOOL_HANDLERS = {
    "water_risk_predict": water_risk_predict,
    "fish_disease_detect": fish_disease_detect,
}

__all__ = [
    "TOOL_HANDLERS",
    "TOOL_SCHEMAS",
    "fish_disease_detect",
    "get_tool_schemas",
    "water_risk_predict",
]
