"""把一次「网页点击」翻译成一次 Agent 决策。

这一层要说清三件事，因为它们都直接影响演示能不能跑稳：

1. **串行执行**。``AquacultureAgent.run()`` 会在开头重置 ``self.trace`` /
   ``self.degraded_reasons``，实例本身不是线程安全的。Gradio 默认并发处理
   请求，所以这里照抄接口层 ``AppState`` 的做法，用一把 ``RLock`` 把
   ``run()`` 串起来。现场演示时两个人同时点「运行」不会把 trace 搅在一起。
2. **知识库是可选件**。Gradio 页面要在**断网、没装向量库、甚至没建过索引**
   的机器上也能跑。所以知识库初始化包在 try/except 里，失败只记状态，
   不阻断主流程——这和接口层 ``/assess`` 仍然返回 200 是同一个口径。
3. **不重新实现业务逻辑**。风险等级、告警、建议全部来自
   :class:`~fisheries.agent.orchestrator.AquacultureAgent`，
   这一层只负责组织入参、串行化、以及把 ``AgentResult`` 摊平成好用的形状。
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.orchestrator import AgentInput, AquacultureAgent

#: 界面上能选的养殖品种。取值必须与 ``water_risk.SPECIES_PROFILES`` 的键一致，
#: 否则工具层会静默回落到鲤科档位，演示时就变成「选了虾蟹却是鲤科阈值」。
SPECIES_CHOICES: tuple[str, ...] = ("鲤科", "罗非鱼", "虾蟹")

#: 病害识别在拿不到权重文件时的模拟模式。``hash`` 是按图片内容做 SHA-256
#: 推导标签，同一张图永远给同一个结果——演示要的就是可复现。
MOCK_SOURCE_CHOICES: tuple[str, ...] = ("hash", "folder_hint")

KNOWLEDGE_DISABLE_ENV = "FISHERIES_DISABLE_KNOWLEDGE"


@dataclass
class WebRequest:
    """一次「点击运行」的完整意图。"""

    readings: list[dict[str, Any]] = field(default_factory=list)
    image_path: str | None = None
    pond_id: str = "demo-pond-1"
    species: str = "鲤科"
    question: str = ""
    mock_source: str = "hash"
    use_knowledge: bool = True

    @property
    def has_water(self) -> bool:
        return len(self.readings) >= 3

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)


@dataclass
class WebResult:
    """一次决策的结果，已经摊平成界面好取用的样子。"""

    ok: bool = False
    #: 面向用户的一句话失败原因（``ok=False`` 时才有意义）
    error: str = ""

    pond_id: str = ""
    species: str = ""
    risk_level: str = "未知"
    fused_score: float = 0.0
    headline: str = ""
    reason: str = ""
    observation_window: str = ""
    confidence_note: str = ""

    #: ``Alert.model_dump()`` 列表，字段是 ``code/level/title/detail``
    alerts: list[dict[str, Any]] = field(default_factory=list)
    #: ``Advisory.model_dump()`` 列表，字段是 ``priority/target/action/timing``
    advisories: list[dict[str, Any]] = field(default_factory=list)
    water_actions: list[dict[str, Any]] = field(default_factory=list)
    disease_actions: list[dict[str, Any]] = field(default_factory=list)

    water: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    fusion: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    trace: list[dict[str, Any]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    knowledge_block: str = ""
    knowledge_summary: dict[str, Any] = field(default_factory=dict)

    llm_backends: dict[str, str] = field(default_factory=dict)
    degraded_mode: bool = False
    degraded_reasons: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    #: 水质模态失败时的原因（工具返回 ``status != "ok"``）
    water_error: str = ""
    vision_error: str = ""

    #: 本次的水质表格摄入结果（:class:`~fisheries.webui.table.TableIngest`）。
    #: 由调用方在 ``run()`` 之后挂上——跑 Agent 本身不需要它，
    #: 但页面要把「数据接入质量」那一栏画出来。
    ingest: Any | None = None
    #: 本次实际喂给工具的水质读数行数
    n_readings: int = 0


class WebSession:
    """持有 Agent 与知识库，供 Gradio 回调长期复用。

    之所以自己建一个类而不是复用 ``fisheries.api.deps.AppState``：那个类还带着
    上传登记表、``allow_local_path``、历史缓冲这些给 HTTP 用的东西，
    页面这边一个都不需要。少一层耦合，演示脚本也就少一处会坏的地方。
    """

    def __init__(self, *, use_knowledge: bool = True) -> None:
        #: ``run()`` 会重置实例内部状态，必须串行
        self.run_lock = threading.RLock()
        self.agent = AquacultureAgent()
        self.knowledge_status: dict[str, Any] = {
            "status": "disabled",
            "detail": "",
            "backend": "",
            "n_chunks": 0,
        }
        self.knowledge_error = ""
        if use_knowledge:
            self._prepare_knowledge()
        else:
            self.knowledge_status["detail"] = "已按请求关闭知识库"

    # ------------------------------------------------------------------ #
    # 知识库
    # ------------------------------------------------------------------ #
    def _prepare_knowledge(self) -> None:
        """尽力打开现成的向量库；打不开就降级，不抛异常。

        注意用 ``auto_build=False``：现场演示点一下按钮不该触发一次
        全量向量化（那要几十秒）。索引没建过时应提示用户先去跑
        ``scripts/build_knowledge_base.py``，而不是让页面卡住。
        """
        try:
            from ..rag.retriever import open_knowledge_base

            kb = open_knowledge_base(auto_build=False)
            # 直接赋值 ``knowledge_base`` 还不够：``_ensure_knowledge_base()``
            # 靠 ``_knowledge_tried`` 判断「要不要再开一次」。不置位的话
            # 第一次 run() 会把向量库重新打开一遍（白等几百毫秒）。
            self.agent.knowledge_base = kb
            self.agent._knowledge_tried = True  # noqa: SLF001 - 同包内显式同步状态
            info = kb.describe() if hasattr(kb, "describe") else {}
            self.knowledge_status = {
                "status": "ready",
                "detail": "知识库已加载",
                "backend": str(info.get("backend", "")),
                "n_chunks": int(info.get("n_chunks", 0) or 0),
            }
        except Exception as exc:  # noqa: BLE001 - 演示环境什么都可能缺
            self.knowledge_error = f"{type(exc).__name__}: {exc}"
            self.knowledge_status = {
                "status": "unavailable",
                "detail": (
                    "知识库不可用，本次不引用语料，"
                    "建议仅为规则模板口径。可先执行 "
                    "`python scripts/build_knowledge_base.py` 重建索引。"
                ),
                "backend": "",
                "n_chunks": 0,
            }

    @property
    def knowledge_ready(self) -> bool:
        return self.knowledge_status.get("status") == "ready"

    # ------------------------------------------------------------------ #
    # 运行
    # ------------------------------------------------------------------ #
    def run(self, req: WebRequest) -> WebResult:
        """跑一次完整决策。**不抛异常**——失败也返回 ``WebResult(ok=False)``。"""
        if not req.has_water and not req.has_image:
            return WebResult(
                ok=False,
                error=(
                    "还没有可用的输入：请上传水质表格（至少 3 行）"
                    "或鱼体图片，两者给一个就能跑。"
                ),
            )

        knowledge = req.use_knowledge and self.knowledge_ready
        try:
            inp = AgentInput(
                pond_id=req.pond_id or "unknown",
                species=req.species,
                water_readings=req.readings if req.has_water else None,
                image_path=req.image_path if req.has_image else None,
                question=req.question or None,
                mock_source=req.mock_source,
                force_mock_llm=True,
                use_knowledge=knowledge,
            )
        except Exception as exc:  # noqa: BLE001 - pydantic 校验失败也要变成人话
            return WebResult(ok=False, error=f"输入校验没通过：{exc}")

        try:
            with self.run_lock:
                result = self.agent.run(inp)
        except Exception as exc:  # noqa: BLE001
            # 演示页最忌讳「点了没反应」。把异常类型、消息、最后几帧调用栈
            # 一起摆出来，现场才能判断是数据问题还是环境问题。
            tail = "".join(traceback.format_exc().splitlines(keepends=True)[-4:])
            return WebResult(
                ok=False,
                error=f"Agent 运行出错：{type(exc).__name__}: {exc}\n\n{tail}",
            )

        return _to_web_result(result, req)


def _to_web_result(result: Any, req: WebRequest) -> WebResult:
    """把 ``AgentResult`` 摊平成 :class:`WebResult`。"""
    water = result.water or None
    vision = result.vision or None

    water_error = ""
    if water and water.get("status") != "ok":
        water_error = str(water.get("error_message") or water.get("status") or "水质工具未返回结果")
    vision_error = ""
    if vision and vision.get("status") != "ok":
        vision_error = str(vision.get("error_message") or vision.get("status") or "图片工具未返回结果")

    report = result.report or {}
    return WebResult(
        ok=True,
        pond_id=result.pond_id or "",
        species=req.species,
        risk_level=result.risk_level,
        fused_score=float(result.fused_score or 0.0),
        headline=result.headline or "",
        reason=result.reason or "",
        observation_window=result.observation_window or "",
        confidence_note=result.confidence_note or "",
        alerts=list(result.alerts or []),
        advisories=[a.model_dump(mode="json") for a in (result.advisories or [])],
        # ``report`` 已经是 ``AdvisorReport.model_dump(mode="json")`` 的结果，
        # 里面的元素是 dict 不是模型对象——这里不能再调 ``model_dump()``。
        water_actions=[dict(a) for a in (report.get("water_actions") or [])],
        disease_actions=[dict(a) for a in (report.get("disease_actions") or [])],
        water=water,
        vision=vision,
        fusion=dict(result.fusion or {}),
        report=dict(report),
        trace=[
            {
                "step": t.step,
                "kind": t.kind,
                "title": t.title,
                "detail": t.detail,
                "data": t.data,
            }
            for t in (result.trace or [])
        ],
        references=list(result.references or []),
        knowledge_block=result.knowledge_block or "",
        knowledge_summary=dict(result.knowledge_summary or {}),
        llm_backends=dict(result.llm_backends or {}),
        degraded_mode=bool(result.degraded_mode),
        degraded_reasons=list(result.degraded_reasons or []),
        elapsed_seconds=float(result.elapsed_seconds or 0.0),
        water_error=water_error,
        vision_error=vision_error,
        n_readings=len(req.readings) if req.has_water else 0,
    )


def default_session() -> WebSession:
    """按环境变量决定要不要加载知识库，返回一个可直接用的会话。"""
    import os

    disabled = str(os.environ.get(KNOWLEDGE_DISABLE_ENV, "")).strip().lower()
    return WebSession(use_knowledge=disabled not in {"1", "true", "yes", "on"})


def session_info(session: WebSession) -> str:
    """给界面顶部用的一行后端说明。"""
    try:
        backends = session.agent.describe_backends()
    except Exception:  # noqa: BLE001
        backends = {}
    scheduler = backends.get("scheduler", "未知") if isinstance(backends, dict) else "未知"
    advisor = backends.get("advisor", "未知") if isinstance(backends, dict) else "未知"
    kb = session.knowledge_status
    kb_text = {
        "ready": f"已就绪（{kb.get('n_chunks', 0)} 块）",
        "unavailable": "不可用（已降级）",
        "disabled": "已关闭",
    }.get(str(kb.get("status")), "未知")
    return f"调度模型 `{scheduler}`　建议模型 `{advisor}`　知识库 {kb_text}"


def example_table_path() -> Path:
    """示例水质表的路径（供界面做「一键填入」）。"""
    from .table import EXAMPLE_TABLE

    return EXAMPLE_TABLE
