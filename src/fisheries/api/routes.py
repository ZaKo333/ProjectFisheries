"""接口层的路由实现。

一个贯穿全文件的原则：**接口层不重新实现业务逻辑**。
它只做四件事——校验入参、调用已实现的工具/Agent、把结果转成稳定的响应契约、
把异常转成明确的 HTTP 状态码。任何"顺手在这里算一下"的冲动都会让
同一个规则在两个地方各有一份，日后必然漂移。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from .. import __version__
from ..agent.orchestrator import AdvisorReport, AgentInput, AgentResult
from ..tools import TOOL_HANDLERS, get_tool_schemas
from ..tools.fish_disease import fish_disease_detect
from ..tools.water_risk import (
    ACTION_THRESHOLDS,
    BASE_WEIGHTS,
    MAX_EXTRAPOLATION_HOURS,
    MIN_RELIABLE_POINTS,
    PERSISTENCE_CAP,
    REDLINE,
    SPECIES_PROFILES,
)
from .deps import AppState, get_state
from .schemas import (
    AdvisoryOut,
    AlertOut,
    AssessRequest,
    AssessResponse,
    ChatChunkOut,
    ChatRequest,
    ChatResponse,
    EvidenceOut,
    HealthResponse,
    HistoryResponse,
    IngestImageResponse,
    IngestSensorRequest,
    IngestSensorResponse,
    ThresholdResponse,
    ToolCallRequest,
    ToolResultResponse,
    ToolSchemasResponse,
    TraceStepOut,
)

__all__ = ["router"]

router = APIRouter(prefix="/api/v1")

#: 允许上传的图片扩展名。**只是第一道粗筛**——真正的格式判定由
#: `fish_disease_detect` 的魔数校验负责，这里不重复实现一份魔数表。
_ALLOWED_UPLOAD_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# --------------------------------------------------------------------------- #
# 转换：AgentResult -> 响应契约
# --------------------------------------------------------------------------- #

def _to_response(
    result: AgentResult,
    *,
    include_trace: bool = True,
    include_knowledge_block: bool = True,
) -> AssessResponse:
    """把 Agent 的返回值转成 HTTP 响应模型。

    这里刻意不像 `water` / `vision` / `fusion` 那样做字段级重建：
    它们的权威定义在工具层，抄一份进来只会在工具改字段时静默漂移。
    """
    return AssessResponse(
        pond_id=result.pond_id,
        risk_level=result.risk_level,  # type: ignore[arg-type]
        fused_score=result.fused_score,
        reason=result.reason,
        headline=result.headline,
        advisories=[
            AdvisoryOut(
                priority=a.priority, target=a.target, action=a.action, timing=a.timing
            )
            for a in result.advisories
        ],
        observation_window=result.observation_window,
        confidence_note=result.confidence_note,
        report=AdvisorReport.model_validate(result.report) if result.report else None,
        alerts=[
            AlertOut(
                code=str(a.get("code", "")),
                level=str(a.get("level", "")),
                title=str(a.get("title", "")),
                detail=str(a.get("detail", "")),
            )
            for a in result.alerts
        ],
        evidence=[
            EvidenceOut(
                source=str(e.get("source", "")),
                code=str(e.get("code", "")),
                label=str(e.get("label", "")),
                value=e.get("value"),
                unit=e.get("unit"),
            )
            for e in result.evidence
        ],
        water=result.water,
        vision=result.vision,
        fusion=result.fusion or None,
        references=list(result.references),
        knowledge_summary=dict(result.knowledge_summary),
        knowledge_block=result.knowledge_block if include_knowledge_block else "",
        trace=(
            [
                TraceStepOut(
                    step=t.step,
                    kind=t.kind,
                    title=t.title,
                    detail=t.detail,
                    data=t.data or {},
                )
                for t in result.trace
            ]
            if include_trace
            else []
        ),
        llm_backends=dict(result.llm_backends),
        degraded_mode=result.degraded_mode,
        degraded_reasons=list(result.degraded_reasons),
        elapsed_seconds=result.elapsed_seconds,
    )


def _readings_to_dicts(readings: list[Any]) -> list[dict[str, Any]]:
    """把接口入参模型转成工具层吃的普通 dict。

    ``mode="json"`` 不是可有可无的：``WaterQualityPoint.timestamp`` 是
    ``datetime``，用默认的 python 模式 dump 出来是 datetime 对象。工具层能
    接受它，但调度提示词要用 ``json.dumps`` 把它塞进 ``##CTX##`` 段——
    于是整个 ``/assess`` 接口会以
    ``TypeError: Object of type datetime is not JSON serializable`` 报 500。
    工具契约里 timestamp 本来就是 ISO 8601 字符串，在这里转成字符串才是一致的。

    ``exclude_none=True`` 也是刻意的：工具把「字段不存在」与「字段为 null」
    都当作缺失指标处理，但传 None 会让日志与 trace 里多出一串无意义的 null。
    """
    return [
        r.model_dump(mode="json", exclude_none=True) if hasattr(r, "model_dump") else dict(r)
        for r in readings
    ]


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #

@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(request: Request) -> HealthResponse:
    state: AppState = get_state(request)
    kb = state.knowledge_status
    degraded = kb.get("status") != "ready" or "mock" in state.describe_tools()["fish_disease_detect"]
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        server_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        uptime_seconds=round(time.time() - state.started_at, 3),
        tools=state.describe_tools(),
        llm_backends=state.agent.describe_backends(),
        knowledge=kb,
        notes=list(state.notes),
    )


# --------------------------------------------------------------------------- #
# 核心：融合评估
# --------------------------------------------------------------------------- #

@router.post(
    "/assess",
    response_model=AssessResponse,
    summary="融合评估（水质 + 鱼体图片 → 风险等级 / 告警 / 养殖建议）",
)
def assess(payload: AssessRequest, request: Request) -> AssessResponse:
    state: AppState = get_state(request)

    readings = _readings_to_dicts(payload.readings) if payload.readings else None
    if readings is None and payload.use_buffer:
        readings = state.get_readings(payload.pond_id)
        if not readings:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"pond_id={payload.pond_id!r} 的会话缓冲里没有读数；"
                    "请先 POST /api/v1/ingest/sensor，或直接在 readings 里传数据。"
                ),
            )

    image_path = _resolve_image_path(state, payload.image_path, payload.upload_id)
    if not readings and not image_path:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="水质时序数据与鱼体图片至少要提供一项（readings / image_path / upload_id）",
        )

    inp = AgentInput(
        pond_id=payload.pond_id,
        species=payload.species,
        water_readings=readings,
        image_path=image_path,
        question=payload.question,
        mock_source=payload.mock_source,
        use_knowledge=payload.use_knowledge,
    )

    # 用同一个 Agent 实例必须串行：它会重置并复用 self.trace / self.degraded_reasons。
    with state.run_lock:
        try:
            result = state.agent.run(inp)
        except Exception as exc:  # noqa: BLE001 —— 兜底，别把 traceback 直接甩给调用方
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent 执行失败：{type(exc).__name__}: {exc}",
            ) from exc

    response = _to_response(
        result,
        include_trace=payload.include_trace,
        include_knowledge_block=payload.include_knowledge_block,
    )
    # 历史里存精简版：轨迹和知识块原文又大又只对当次请求有意义。
    state.record(payload.pond_id, _to_response(result, include_trace=False, include_knowledge_block=False))
    return response


def _resolve_image_path(
    state: AppState, image_path: str | None, upload_id: str | None
) -> str | None:
    """确定本次要分析的图片路径。

    优先 ``upload_id``（上传得到，路径由服务端生成，不存在路径穿越问题）；
    其次才是客户端直传的服务器路径，且受 ``allow_local_path`` 开关约束。
    """
    if upload_id:
        path = state.lookup_upload(upload_id)
        if path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"未知的 upload_id={upload_id!r}；请先 POST /api/v1/ingest/image",
            )
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"upload_id={upload_id!r} 对应的文件已被清理：{path}",
            )
        return str(path)

    if image_path is None:
        return None

    if not state.settings.allow_local_path:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "服务已关闭「按服务器路径读图」（FISHERIES_API_ALLOW_LOCAL_PATH=0）；"
                "请改用 POST /api/v1/ingest/image 上传图片并传 upload_id。"
            ),
        )
    if not Path(image_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"image_path 指向的文件不存在：{image_path}",
        )
    return image_path


# --------------------------------------------------------------------------- #
# 工具直调（跳过 LLM 决策）
# --------------------------------------------------------------------------- #

@router.post(
    "/tools/call",
    response_model=ToolResultResponse,
    summary="直接调用工具（跳过 LLM 决策，便于调试与前端联调）",
)
def call_tool(payload: ToolCallRequest, request: Request) -> ToolResultResponse:
    get_state(request)  # 确保应用已初始化
    handler = TOOL_HANDLERS.get(payload.tool)
    if handler is None:  # pragma: no cover —— Literal 已限定取值
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"未注册的工具：{payload.tool}"
        )
    started = time.perf_counter()
    try:
        result = handler(**payload.arguments)
    except TypeError as exc:
        # 参数名写错 / 少传必填项时，把真实的签名错误回给调用方，别吞掉。
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"工具参数不合法：{exc}",
        ) from exc
    return ToolResultResponse(
        tool=payload.tool,
        elapsed_seconds=round(time.perf_counter() - started, 4),
        result=result,
    )


# --------------------------------------------------------------------------- #
# 数据写入
# --------------------------------------------------------------------------- #

@router.post(
    "/ingest/sensor",
    response_model=IngestSensorResponse,
    summary="追加传感器读数到会话缓冲（内存态）",
)
def ingest_sensor(payload: IngestSensorRequest, request: Request) -> IngestSensorResponse:
    state: AppState = get_state(request)
    readings = _readings_to_dicts(payload.readings)
    total, first, last = state.append_readings(
        payload.pond_id, readings, replace=payload.replace
    )
    return IngestSensorResponse(
        pond_id=payload.pond_id,
        accepted=len(readings),
        buffered_total=total,
        first_timestamp=first,
        last_timestamp=last,
    )


@router.get(
    "/ponds/{pond_id}/readings",
    summary="读取会话缓冲里的读数",
)
def get_readings(pond_id: str, request: Request, limit: int = 200) -> dict[str, Any]:
    state: AppState = get_state(request)
    items = state.get_readings(pond_id)
    limit = max(1, min(limit, state.settings.buffer_max))
    return {
        "pond_id": pond_id,
        "total": len(items),
        "returned": min(limit, len(items)),
        "readings": items[-limit:],
    }


@router.post(
    "/ingest/image",
    response_model=IngestImageResponse,
    summary="上传鱼体图片并立即识别",
)
async def ingest_image(
    request: Request, file: UploadFile = File(..., description="鱼体图片")
) -> IngestImageResponse:
    state: AppState = get_state(request)
    original = Path(file.filename or "upload")
    ext = original.suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"不支持的扩展名 {ext or '(无)'}；仅接受 "
                f"{'、'.join(sorted(_ALLOWED_UPLOAD_EXT))}"
            ),
        )

    limit = state.settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件超过 {state.settings.max_upload_mb} MB 上限",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="上传内容为空"
        )

    # 文件名由服务端生成，**不使用客户端文件名**——这样路径穿越在源头就不成立。
    upload_id = uuid.uuid4().hex
    dest_dir = state.settings.upload_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{upload_id}{ext}"
    dest.write_bytes(data)

    result = await run_in_threadpool(fish_disease_detect, str(dest))
    if result.get("status") != "ok":
        # 格式不合法（魔数对不上）等情况下不要把垃圾文件留在磁盘上。
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": result.get("error_code", "UNKNOWN"),
                "error_message": result.get("error_message", "图像识别失败"),
            },
        )

    state.register_upload(upload_id, dest)
    return IngestImageResponse(
        upload_id=upload_id,
        filename=original.name,
        size_bytes=len(data),
        content_type=file.content_type,
        stored_path=str(dest),
        vision=result,
    )


# --------------------------------------------------------------------------- #
# RAG 问答
# --------------------------------------------------------------------------- #

@router.post("/chat", response_model=ChatResponse, summary="知识库问答（RAG）")
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    state: AppState = get_state(request)
    if state.knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "知识库不可用：" + str(state.knowledge_status.get("detail", "未初始化"))
                + "；可先运行 scripts/build_knowledge_base.py 建库。"
            ),
        )

    from ..rag.schemas import RetrievalQuery

    query = RetrievalQuery(
        query=payload.query,
        k=payload.k,
        category=payload.category,
        disease=payload.disease,
    )
    try:
        result = state.knowledge_base.search(query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"检索失败：{type(exc).__name__}: {exc}",
        ) from exc

    chunks = [
        ChatChunkOut(
            chunk_id=c.chunk_id,
            title=c.title,
            heading_text=c.heading_text,
            text=c.text,
            category=c.category,
            disease=c.disease,
            verified=c.verified,
            score=c.score,
            dense_score=c.dense_score,
            sparse_score=c.sparse_score,
            matched_by=list(c.matched_by),
            citation_keys=list(c.citation_keys),
        )
        for c in result.chunks
    ]

    answer, is_raw = _answer_from_chunks(state, payload, result)
    return ChatResponse(
        query=payload.query,
        answer=answer,
        answer_is_raw_retrieval=is_raw,
        chunks=chunks,
        backend=str(getattr(state.knowledge_base, "backend", "")),
        warnings=_warnings_for_chat(result, chunks),
    )


def _warnings_for_chat(result: Any, chunks: list[ChatChunkOut]) -> list[str]:
    """检索告警 + 「语料未核对」告警。

    检索器自己只报检索机制层面的问题（召回不足、过滤过严），
    而「这批条目没和标准原文核对过」是**语料的可信度**问题，属于检索器的
    职责之外。但它是调用方最需要知道的一条——只看 ``verified`` 字段很容易
    漏掉，所以在接口层显式提升为顶层 warning。Agent 那条路径上
    ``KnowledgeContext.render()`` 已经在做同样的事，这里保持一致。
    """
    warnings = list(result.warnings)
    unverified = [c for c in chunks if not c.verified]
    if unverified:
        warnings.append(
            f"返回的 {len(unverified)}/{len(chunks)} 条知识条目标注为『未与标准原文核对』，"
            "引用其数值时需人工复核现行标准。"
        )
    return warnings


def _provenance_block(chunk: Any) -> str:
    """把一条检索结果渲染成可直接返回给调用方的文本块。

    标记口径与报告层的 ``references`` 共用 ``RetrievedChunk.provenance_mark()``，
    避免两处各写一套「核对过没有」的措辞。
    """
    return (
        f"【{chunk.title} > {chunk.heading_text}】{chunk.provenance_mark()}\n{chunk.text}"
    )


def _answer_from_chunks(
    state: AppState, payload: ChatRequest, result: Any
) -> tuple[str, bool]:
    """生成回答。

    没配大模型时**不编一段像模像样的回答**，而是把检索到的原文拼出来，
    并把 ``answer_is_raw_retrieval`` 置 True 告诉调用方"这不是模型写的"。
    理由：接口层最重要的品质是让调用方分得清「模型说的」和「系统查到的」。
    """
    from ..agent.llm import MockLLMClient

    if not payload.answer_with_llm or isinstance(state.agent.advisor, MockLLMClient):
        if not result.chunks:
            return "知识库中未检索到相关条目。", True
        blocks = [_provenance_block(c) for c in result.chunks]
        return "\n\n".join(blocks), True

    context = "\n\n".join(
        f"【{i + 1}】{c.title} > {c.heading_text}\n{c.text}"
        + ("" if c.verified else "（⚠ 未与标准原文核对）")
        for i, c in enumerate(result.chunks)
    )
    system = (
        "你是淡水池塘养殖技术助理。只能依据【参考资料】回答，"
        "资料里没有的内容必须明确说『知识库中没有相关依据』，不得凭常识补充。"
        "资料标注『依据 GB 11607-1989 等』的数值可直接引用并给出标准编号与条款；"
        "标注『未与标准原文核对』的必须一并说明其数值需人工复核。用中文回答，分点、可执行。"
    )
    user = f"【问题】{payload.query}\n\n【参考资料】\n{context or '（无）'}"
    try:
        resp = state.agent.advisor.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=None,
        )
    except Exception as exc:  # noqa: BLE001 —— LLM 挂了就退回检索原文，别让接口 500
        return (
            f"（大模型调用失败，以下是检索到的原文）\n\n"
            + "\n\n".join(f"【{c.title} > {c.heading_text}】\n{c.text}" for c in result.chunks),
            True,
        )
    return (resp.content or "").strip(), False


# --------------------------------------------------------------------------- #
# 历史与配置
# --------------------------------------------------------------------------- #

@router.get(
    "/ponds/{pond_id}/history",
    response_model=HistoryResponse,
    summary="评估历史（内存态，进程重启即失效）",
)
def get_history(pond_id: str, request: Request, limit: int = 20) -> HistoryResponse:
    state: AppState = get_state(request)
    items = state.get_history(pond_id)
    limit = max(1, min(limit, state.settings.history_max))
    return HistoryResponse(pond_id=pond_id, total=len(items), items=items[-limit:][::-1])


@router.get(
    "/config/thresholds",
    response_model=ThresholdResponse,
    summary="阈值、权重与红线规则（供前端展示判定依据）",
)
def get_thresholds(request: Request) -> ThresholdResponse:
    get_state(request)
    return ThresholdResponse(
        species=list(SPECIES_PROFILES.keys()),
        weights=dict(BASE_WEIGHTS),
        effective_note=(
            "计算风险分时权重会在『本次实际可用的指标』上重新归一化："
            "亚硝酸盐缺失（实测数据集只有硝酸盐）时该列剔除，其余权重按比例放大。"
        ),
        profiles=SPECIES_PROFILES,
        redlines=dict(REDLINE),
        action_thresholds={
            k: [{"value": v, "label": lbl} for v, lbl in pairs]
            for k, pairs in ACTION_THRESHOLDS.items()
        },
        extrapolation_horizon_hours=MAX_EXTRAPOLATION_HOURS,
        min_reliable_points=MIN_RELIABLE_POINTS,
        persistence_cap=PERSISTENCE_CAP,
        disclaimer=(
            "标准限值部分已对到 GB 11607-1989《渔业水质标准》（现行）原文"
            "（2026-10-15 核验，见 knowledge/corpus/08_标准与依据.md）；"
            "适宜 / 警戒 / 危险 / 红线四档中的后三档是本项目的工程扩展，不是标准原文；"
            "正式使用前仍须由有资质的水产技术人员复核。本系统输出不构成用药指令。"
        ),
    )


@router.get(
    "/config/tools",
    response_model=ToolSchemasResponse,
    summary="工具契约（OpenAI function calling 原生格式）",
)
def get_tools(request: Request) -> ToolSchemasResponse:
    get_state(request)
    return ToolSchemasResponse(tools=get_tool_schemas())
