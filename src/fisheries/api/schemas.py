"""后端接口层的数据契约（请求 / 响应模型）。

设计原则
--------
1. **入参尽量复用工具层已有的模型**，而不是另起一套。
   `/assess` 的 `readings` 直接用 `water_risk.WaterQualityPoint`，
   于是「API 接受的字段」与「工具接受的字段」永远一致——
   不会出现「接口收得进、工具吃不下」这类只有联调时才暴露的问题。

2. **出参只对接口自己拥有的结构建类型**（告警 / 建议 / 证据 / 轨迹），
   对 `water` / `vision` / `fusion` 三个嵌套结果保留 `dict[str, Any]`。
   原因：这三个结构分别有 20+ / 20+ / 12 个字段，手抄一份等价模型出来，
   工具层一改字段就会静默漂移；而它们的权威定义在
   `src/fisheries/tools/definitions.py` 的 `TOOL_SCHEMAS`，
   通过 ``GET /api/v1/config/tools`` 可以取到完整契约。

3. 所有模型 ``extra="forbid"``：客户端把 ``dissolved_oxygen`` 拼成
   ``dissolved_oxyen`` 时应当**报错**，而不是静默当成没传这个指标、
   再算出一个「没有溶氧的风险分」。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..agent.orchestrator import AdvisorReport
from ..tools.water_risk import WaterQualityPoint

__all__ = [
    "SensorReadingIn",
    "AssessRequest",
    "AssessResponse",
    "AlertOut",
    "AdvisoryOut",
    "EvidenceOut",
    "TraceStepOut",
    "IngestSensorRequest",
    "IngestSensorResponse",
    "IngestImageResponse",
    "ChatRequest",
    "ChatResponse",
    "ChatChunkOut",
    "HistoryResponse",
    "HealthResponse",
    "ToolCallRequest",
    "ToolResultResponse",
    "ThresholdResponse",
    "ToolSchemasResponse",
    "TemplateInfo",
    "TemplateCatalogResponse",
]


class _Strict(BaseModel):
    """所有接口模型的公共基类：禁止未知字段。"""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #

#: `readings` 直接用工具层的模型，保证接口与工具字段完全一致。
SensorReadingIn = WaterQualityPoint


class AssessRequest(_Strict):
    """核心评估请求：融合水质时序 + 鱼体图片，产出风险等级与养殖建议。"""

    pond_id: str = Field(default="unknown", max_length=64, description="池塘编号")
    species: Literal["鲤科", "罗非鱼", "虾蟹"] = Field(
        default="鲤科", description="养殖品种，决定阈值档位"
    )
    readings: list[SensorReadingIn] | None = Field(
        default=None,
        min_length=3,
        max_length=20_000,
        description=(
            "水质时序读数，按时间升序。至少 3 条（趋势回归的硬下限）；"
            "不传则只做图像模态。上限 20000 条 ≈ 14 天的一分钟采样。"
        ),
    )
    use_buffer: bool = Field(
        default=False,
        description=(
            "为 True 且未传 readings 时，改用 /ingest/sensor 累积的会话缓冲。"
            "（内存态，进程重启即失效，仅用于演示与联调）"
        ),
    )
    image_path: str | None = Field(
        default=None,
        max_length=512,
        description="服务器本地图片路径。生产环境应关闭，见 allow_local_path。",
    )
    upload_id: str | None = Field(
        default=None,
        max_length=64,
        description="先调 /ingest/image 拿到 upload_id，再在这里引用（推荐做法）",
    )
    mock_source: Literal["hash", "folder_hint"] = Field(
        default="hash",
        description=(
            "仅图像模型未训练（mock 兜底）时生效。hash=按文件内容确定性随机；"
            "folder_hint=从文件夹名取标签以稳定触发协同规则，仅用于链路联调。"
        ),
    )
    use_knowledge: bool | None = Field(
        default=None, description="是否做 RAG 检索；None=按服务默认配置"
    )
    question: str | None = Field(
        default=None, max_length=500, description="可选的自然语言补充问题"
    )
    include_trace: bool = Field(default=True, description="是否返回完整推理轨迹")
    include_knowledge_block: bool = Field(
        default=True,
        description=(
            "是否返回实际注入建议提示词的知识块原文。"
            "保留它是为了让前端/评委能直接看到建议的依据，而不只是相信「接了 RAG」。"
        ),
    )

    @field_validator("readings")
    @classmethod
    def _readings_not_empty(
        cls, v: list[SensorReadingIn] | None
    ) -> list[SensorReadingIn] | None:
        # 传了空列表和没传是两回事：前者是调用方搞错了，应当报错。
        if v is not None and len(v) == 0:
            raise ValueError("readings 传了空列表；若不使用水质数据请传 null 或省略该字段")
        return v


class IngestSensorRequest(_Strict):
    """追加传感器读数到会话缓冲。"""

    pond_id: str = Field(default="unknown", max_length=64)
    readings: list[SensorReadingIn] = Field(min_length=1, description="至少 1 条")
    #: 缓冲是内存态，必须设上限，否则长时间运行的演示进程会一直涨内存。
    replace: bool = Field(default=False, description="True=覆盖原有缓冲而非追加")


class ChatRequest(_Strict):
    """知识库问答。"""

    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=5, ge=1, le=20)
    category: Literal["水质", "病害", "投料", "应急", "标准"] | None = None
    disease: Literal["烂鳃", "出血", "水霉", "其他"] | None = None
    answer_with_llm: bool = Field(
        default=True,
        description=(
            "是否让建议角色大模型基于检索结果组织回答。"
            "未配置大模型时本参数无效，会直接返回检索原文。"
        ),
    )


class ToolCallRequest(_Strict):
    """直接调用某个工具（跳过 LLM 决策），用于调试与前端联调。"""

    tool: Literal["water_risk_predict", "fish_disease_detect"]
    arguments: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


class AlertOut(_Strict):
    code: str
    level: str
    title: str
    detail: str


class AdvisoryOut(_Strict):
    priority: int
    target: str
    action: str
    timing: str


class EvidenceOut(_Strict):
    source: str
    code: str
    label: str
    value: float | None = None
    unit: str | None = None


class TraceStepOut(_Strict):
    step: int
    kind: str
    title: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class AssessResponse(_Strict):
    """融合评估结果。字段名与 ``AgentResult`` 保持一致，便于前端与 CLI 共用一套理解。"""

    pond_id: str
    risk_level: Literal["正常", "预警", "高危"]
    fused_score: float
    reason: str

    headline: str = ""
    advisories: list[AdvisoryOut] = Field(default_factory=list)
    observation_window: str = ""
    confidence_note: str = ""

    #: 面向养殖户的结构化报告：综合风险等级 / 告警信息 / 水质调控建议 / 病害处置方案。
    #: 上面的 headline/advisories 等字段是从它展平出来的，保留是为了兼容既有前端；
    #: 新接入方建议直接用 ``report``。它的 risk_level 与 alerts 由融合层强制回填，
    #: 与顶层同名字段恒等——大模型只能改措辞，改不了定级。
    report: AdvisorReport | None = None

    alerts: list[AlertOut] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)

    #: 工具返回原文（契约见 GET /api/v1/config/tools）
    water: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    #: 融合层结构化结果（权重 / 协同命中 / 失败模态等）
    fusion: dict[str, Any] | None = None

    references: list[str] = Field(default_factory=list)
    knowledge_summary: dict[str, Any] = Field(default_factory=dict)
    #: 实际注入建议角色提示词的知识块原文（可按 include_knowledge_block 关闭）
    knowledge_block: str = ""

    trace: list[TraceStepOut] = Field(default_factory=list)

    llm_backends: dict[str, str] = Field(default_factory=dict)
    degraded_mode: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class IngestSensorResponse(_Strict):
    pond_id: str
    accepted: int
    buffered_total: int
    first_timestamp: str | None = None
    last_timestamp: str | None = None


class IngestImageResponse(_Strict):
    upload_id: str
    filename: str
    size_bytes: int
    content_type: str | None = None
    #: 落盘后的绝对路径，可直接用于 /assess 的 image_path
    stored_path: str
    vision: dict[str, Any]


class ChatChunkOut(_Strict):
    chunk_id: str
    title: str
    heading_text: str
    text: str
    category: str
    #: 只有病害类语料才有病种；水质 / 投料 / 应急 / 标准类语料这里是 None。
    #: 早期写成 `str` 会让这类语料在序列化时抛 ValidationError → 整个 /chat 500。
    disease: str | None = None
    verified: bool
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    matched_by: list[str] = Field(default_factory=list)
    citation_keys: list[str] = Field(default_factory=list)


class ChatResponse(_Strict):
    query: str
    answer: str
    #: True 表示 answer 是检索原文的拼装结果，不是大模型组织过的回答
    answer_is_raw_retrieval: bool = False
    chunks: list[ChatChunkOut] = Field(default_factory=list)
    backend: str = ""
    warnings: list[str] = Field(default_factory=list)


class HistoryResponse(_Strict):
    pond_id: str
    total: int
    items: list[AssessResponse] = Field(default_factory=list)


class HealthResponse(_Strict):
    status: Literal["ok", "degraded"]
    version: str
    server_time: str
    uptime_seconds: float
    tools: dict[str, str] = Field(default_factory=dict)
    llm_backends: dict[str, str] = Field(default_factory=dict)
    knowledge: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ToolResultResponse(_Strict):
    tool: str
    elapsed_seconds: float
    result: dict[str, Any]


class ThresholdResponse(_Strict):
    """阈值与权重的只读视图——让前端能直接展示「判定依据」。"""

    species: list[str]
    weights: dict[str, float]
    effective_note: str
    profiles: dict[str, Any]
    redlines: dict[str, float]
    action_thresholds: dict[str, list[dict[str, Any]]]
    extrapolation_horizon_hours: float
    min_reliable_points: int
    persistence_cap: float
    disclaimer: str


class TemplateInfo(_Strict):
    """单个内置水质模板的目录信息（不含数据本体）。"""

    key: str
    name: str
    scenario: str
    species: str
    expected_risk: str
    focus: str
    tags: list[str]
    header_style: str
    csv_filename: str
    csv_url: str
    n_points: int
    interval_minutes: int
    start: str
    end: str


class TemplateCatalogResponse(_Strict):
    """内置水质模板目录——让前端给出「选一个模板直接试」的入口。"""

    count: int
    interval_minutes: int
    columns: dict[str, str]
    usage: str
    templates: list[TemplateInfo]
    disclaimer: str


class ToolSchemasResponse(_Strict):
    tools: list[dict[str, Any]]
