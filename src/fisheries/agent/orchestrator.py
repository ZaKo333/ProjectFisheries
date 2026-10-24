"""M6 Agent 主调度 —— 接收数据、决策调工具、融合、出建议。

整条链路
==========
    用户输入（水质时序 + 鱼病图片）
        ↓
    ① 调度角色 LLM：看数据 → 决定调哪些工具 → 发起 function call
        ↓
    ② 工具执行：water_risk_predict / fish_disease_detect
        ↓
    ③ 结果回灌 LLM → 它再决定"还要调别的吗"
        ↓
    ④ 护栏校验：该调的漏没漏？（见下）
        ↓
    ⑤ 融合层（确定性规则）：算风险等级 + 告警 + 证据链
        ↓
    ⑥ 建议角色 LLM：把结构化结论翻译成可落地养殖建议
        ↓
    输出：风险等级、告警、建议、证据链、完整推理轨迹

为什么要加"护栏"（guardrail）
================================
纯 LLM 驱动的工具选择有一个致命失败模式：**模型该调工具却没调**。
7B 级模型尤其容易漏调第二个工具，或者把两类数据只处理一类。
一旦漏调，下游融合就拿不到那一模态，结论会静默地变得片面——
而调用方（后端/前端）完全看不出来，还以为一切正常。

所以本层在 LLM 循环结束后做一次**必调工具覆盖检查**：
凡是输入里提供了数据、但 LLM 没调的工具，由调度层直接补调，
并在 trace 里记为 `guardrail`，**不隐瞒这件事**。
这相当于给 LLM 决策加了一层安全带：模型靠谱时它是冗余的，模型抽风时它救命。

另一条护栏是**防重复调用**：同一工具 + 同样参数只允许调用一次，
避免小模型陷入"调工具 → 看结果 → 再调同一个工具"的死循环。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from fisheries.agent.fusion import Alert, FusionResult, fuse_assessment
from fisheries.agent.llm import (
    ChatClient,
    LLMConfig,
    LLMUnavailableError,
    MockLLMClient,
    OpenAICompatibleClient,
    build_client,
    describe_backend,
    load_llm_config,
)
from fisheries.agent.prompts import (
    advisor_system_prompt,
    build_advisor_user_message,
    build_confidence_facts,
    build_scheduler_context,
    render_confidence_note,
    scheduler_system_prompt,
)
from fisheries.agent.translate import (
    LANG_EN,
    LANG_ZH,
    back_translate_report,
    build_translator,
    describe_translator,
    load_translator_client,
    resolve_prompt_lang,
)
from fisheries.tools import TOOL_HANDLERS, get_tool_schemas

#: LLM 循环最大轮数。超过即中断并交给护栏兜底，防止死循环烧 token。
MAX_ROUNDS: int = 6

#: 关闭 RAG 的环境变量开关。设为 ``1``/``true``/``yes``/``on`` 即关闭。
#:
#: 存在两个理由：
#:     1. 单测里不应该去加载 90 MB 的向量模型——那会把毫秒级的用例变成秒级，
#:        而且让"Agent 调度逻辑"的测试耦合到"向量库是否已构建"。
#:     2. 部署到没有知识库的环境（或显存/内存吃紧的边缘设备）时，
#:        Agent 应能退化成"无依据建议"而不是启动失败。
KNOWLEDGE_DISABLE_ENV = "FISHERIES_DISABLE_KNOWLEDGE"


def knowledge_enabled_by_default() -> bool:
    """读环境变量决定是否默认启用 RAG。"""
    raw = os.environ.get(KNOWLEDGE_DISABLE_ENV, "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


#: 跳过「调度 LLM」直接跑确定性工具的环境变量。
#:
#: 为什么要有这条快路径：一次 ``/assess`` 原本至少两段串行 LLM —— 先让调度角色
#: 决定调哪些工具（还要多一轮把工具结果回灌、再问一次"还要不要调"），再让建议角色
#: 写建议。而输入里已经有读数/图片时，**该调哪个工具是确定的**（见
#: :meth:`AquacultureAgent._required_tools`），问模型等于花 10–20 秒换一个和
#: 护栏一样的结果。实测 qwen-plus 单轮约 8–15 秒，砍掉调度这一两轮，
#: ``/assess`` 从 40–60 秒降到 20 秒上下。
#:
#: 取值：``auto``（默认，真实 LLM 才走快路径，Mock 保持原样以便单测覆盖调度循环）、
#: ``1``/``on``（强制）、``0``/``off``（关闭）。
DIRECT_TOOLS_ENV = "FISHERIES_AGENT_DIRECT_TOOLS"

#: 单次 ``run()`` 的总时间预算（秒，``0`` = 不限）。
#:
#: 前端对 ``/assess`` 的等待是有限的，与其让用户看到一个"请求超时"，不如在
#: 预算用完时**主动降级**——跳过建议 LLM、改用规则模板（等级与告警本来就由
#: 融合层确定性生成，模型只影响措辞）。
BUDGET_ENV = "FISHERIES_AGENT_BUDGET_SECONDS"

#: 给建议 LLM 预留的时间（秒）。剩余时间少于它就不再发起调用。
ADVISOR_RESERVE_ENV = "FISHERIES_AGENT_ADVISOR_RESERVE_SECONDS"


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else 0.0


def direct_tools_enabled(scheduler: Any = None) -> bool:
    """是否跳过调度 LLM，直接执行由输入决定的确定性工具。

    ``auto``（默认）只在**接了真实模型**（:class:`OpenAICompatibleClient`）时启用：
    单测里的 :class:`MockLLMClient` 与各种脚本化替身都要走完整调度循环，
    才能覆盖"模型选工具 / 护栏补调 / 参数坏掉"这些路径。
    """
    raw = os.environ.get(DIRECT_TOOLS_ENV, "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return isinstance(scheduler, OpenAICompatibleClient)



# --------------------------------------------------------------------------- #
# 一、输入输出契约
# --------------------------------------------------------------------------- #

class AgentInput(BaseModel):
    """Agent 的一次决策请求。

    两个模态都可选，但**至少要有其一**；只有图片或只有水质数据都能正常出结论，
    融合层会自动把权重归一化到单模态上。
    """

    pond_id: str = Field(default="unknown", description="塘口编号")
    species: str = Field(default="鲤科", description="养殖品种，决定阈值档位：鲤科/罗非鱼/虾蟹")
    water_readings: list[dict[str, Any]] | None = Field(
        default=None, description="水质时序读数列表，每条含 timestamp/water_temp/ph/dissolved_oxygen/ammonia_n/nitrite_n")
    image_path: str | None = Field(default=None, description="鱼体图片路径")
    question: str | None = Field(default=None, description="用户额外提问，可选")
    mock_source: str = Field(default="hash",
                             description="仅开发期联调用：图像 mock 的取值来源 hash/folder_hint，不暴露给大模型")
    force_mock_llm: bool = Field(default=False, description="强制使用离线 MockLLM")
    use_knowledge: bool | None = Field(
        default=None,
        description=(
            "本次请求是否做 RAG 检索。None=用 Agent 实例上的默认值；"
            "False=本次跳过（例如只要一个快速的告警，不需要引证）。"
            "只影响检索，不影响风险等级与告警。"
        ),
    )

    @field_validator("water_readings")
    @classmethod
    def _not_empty(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v is not None and len(v) == 0:
            raise ValueError("water_readings 传了空列表；没有数据请传 null")
        return v

    def model_post_init(self, __context: Any, /) -> None:
        if not self.water_readings and not self.image_path:
            raise ValueError("water_readings 与 image_path 至少要提供一个，否则 Agent 无事可做")

    def summary(self) -> str:
        bits = [f"塘口 {self.pond_id}", f"品种 {self.species}"]
        bits.append(f"水质 {len(self.water_readings)} 条读数" if self.water_readings else "无水质数据")
        bits.append("含鱼体图片" if self.image_path else "无图片")
        return "，".join(bits)


class TraceStep(BaseModel):
    """一次推理轨迹。答辩时用于证明"不是黑箱"。"""

    step: int
    kind: Literal["llm", "tool_call", "tool_result", "guardrail",
                  "fusion", "advisor", "error"]
    title: str
    detail: str = ""
    data: dict[str, Any] | None = None


class Advisory(BaseModel):
    """一条可落地的养殖建议。"""

    priority: int = Field(..., ge=1)
    target: str = Field(..., description="针对的指标或问题")
    action: str = Field(..., description="具体措施")
    timing: str = Field(default="", description="时间窗或频次")


class AdvisorReport(BaseModel):
    """面向养殖户的最终建议报告 —— 建议角色提示词要求大模型输出的结构。

    四个业务块与用户约定一致：**综合风险等级 / 告警信息 / 病害处置方案 / 水质调控建议**。

    ``risk_level`` 与 ``alerts`` 虽然由大模型**复述**（这样输出的 JSON 自身完整、
    可以直接给前端或塞进演示），但 :meth:`AquacultureAgent._report_from_parsed`
    会用融合层的确定性结果**强制回填**，并把不一致记进推理轨迹。
    让模型复述不是为了信它，而是为了能**检出**它试图改动等级——
    这是"融合决策不交给 LLM"这条原则的可观测化。
    """

    risk_level: str
    headline: str = ""
    alerts: list[Alert] = Field(default_factory=list, description="告警信息")
    water_actions: list[Advisory] = Field(default_factory=list, description="水质调控建议")
    disease_actions: list[Advisory] = Field(default_factory=list, description="病害处置方案")
    observation_window: str = ""
    confidence_note: str = ""

    def all_actions(self) -> list[Advisory]:
        """水质与病害两块合并成一条按优先级排序的清单（前端直接展示用）。"""
        merged = [*self.water_actions, *self.disease_actions]
        merged.sort(key=lambda a: (a.priority, a.target))
        return merged


class AgentResult(BaseModel):
    """Agent 决策的完整输出。"""

    pond_id: str
    risk_level: str
    fused_score: float
    reason: str

    alerts: list[dict[str, Any]] = Field(default_factory=list)
    advisories: list[Advisory] = Field(default_factory=list)
    headline: str = ""
    observation_window: str = ""
    confidence_note: str = ""

    evidence: list[dict[str, Any]] = Field(default_factory=list)
    water: dict[str, Any] | None = None
    vision: dict[str, Any] | None = None
    fusion: dict[str, Any] = Field(default_factory=dict)

    # RAG 知识库：注入建议角色的原文 + 来源清单。
    # 把注入的原文一并返回，是为了让演示界面能**证明**建议是有依据的，
    # 而不是让评委只能相信我们"接了 RAG"。
    knowledge_block: str = Field(
        default="", description="实际注入建议角色 prompt 的知识块原文"
    )
    references: list[str] = Field(
        default_factory=list,
        description="引用到的知识条目（'《标题》> 小节'），供前端展示出处",
    )
    knowledge_summary: dict[str, Any] = Field(
        default_factory=dict, description="知识库后端、块数、是否降级等元信息"
    )

    # 面向养殖户的最终报告（综合风险等级 / 告警信息 / 病害处置方案 / 水质调控建议）。
    # 以 dict 形式随结果一起返回，便于直接喂给前端、写入历史、或作为 JSON 落盘。
    report: dict[str, Any] = Field(
        default_factory=dict, description="结构化建议报告 AdvisorReport 的 JSON 形式"
    )

    trace: list[TraceStep] = Field(default_factory=list)
    llm_backends: dict[str, str] = Field(default_factory=dict)
    degraded_mode: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0

    # D12：提示词语言与回译留痕。
    # 把这两个字段放进结果里，是为了让"这次到底发生了回译"可被界面与测试观察到——
    # 否则一个英文模型接了中文语料、回译又整段退模板，从最终中文文本上根本看不出来。
    prompt_lang: str = Field(default=LANG_ZH, description="本次提示词使用的语言：zh / en")
    translation_backend: str = Field(default="", description="回译后端：glossary / llm / bypass")
    translation_notes: list[str] = Field(
        default_factory=list, description="回译过程中降级或改写的明细"
    )


# --------------------------------------------------------------------------- #
# 二、主类
# --------------------------------------------------------------------------- #

class AquacultureAgent:
    """水产养殖智能监测与决策 Agent。"""

    def __init__(self,
                 scheduler_client: ChatClient | None = None,
                 advisor_client: ChatClient | None = None,
                 tool_handlers: dict[str, Any] | None = None,
                 max_rounds: int = MAX_ROUNDS,
                 knowledge_base: Any | None = None,
                 use_knowledge: bool | None = None,
                 prompt_lang: str | None = None,
                 translator: Any | None = None,
                 translator_mode: str | None = None) -> None:
        self.tool_handlers = tool_handlers or TOOL_HANDLERS
        self.tool_schemas = get_tool_schemas()
        self.max_rounds = max_rounds

        self.scheduler_cfg: LLMConfig = load_llm_config("scheduler")
        self.advisor_cfg: LLMConfig = load_llm_config("advisor")
        self.scheduler: ChatClient = scheduler_client or build_client(self.scheduler_cfg)
        # 建议角色默认复用调度客户端，避免没配置时静默跑到 Mock 上
        self.advisor: ChatClient = advisor_client or self.scheduler

        # 输入已带读数/图片时，该调哪个工具是确定的：直接执行，省掉调度 LLM 的一两轮。
        self.direct_tools: bool = direct_tools_enabled(self.scheduler)
        #: 单次 run() 的总时间预算（秒，0 = 不限）与留给建议 LLM 的余量。
        self.budget_seconds: float = _env_seconds(BUDGET_ENV, 0.0)
        self.advisor_reserve_seconds: float = _env_seconds(ADVISOR_RESERVE_ENV, 25.0)

        self.use_knowledge = (
            knowledge_enabled_by_default() if use_knowledge is None else use_knowledge
        )
        self.knowledge_base = knowledge_base
        # 只尝试**打开**已有的向量库，绝不在这里构建：
        # 构建要加载 90 MB 模型并全量重算，放在 Agent 构造里会让
        # 首次调用莫名其妙卡十几秒。构建是显式的离线步骤（见 scripts/build_knowledge_base.py）。
        self._knowledge_tried = knowledge_base is not None

        self.trace: list[TraceStep] = []
        self.degraded_reasons: list[str] = []
        #: 最近一次 run() 里因为「参数不是合法 JSON」而没被采纳的工具名。
        #: 这些工具不算调用过，护栏要补调，日志也要说清是参数坏了而不是漏调。
        self._bad_json_calls: set[str] = set()

        # ---- D12：提示词语言 + 回译后端 ----
        # 语言默认跟着建议角色后端走：aqua-7b / mistral 这类英文底座用英文提示词，
        # DeepSeek / Qwen 这类中文后端用中文提示词。显式入参 > 环境变量 > 自动判断，
        # 判断逻辑在 translate.resolve_prompt_lang 里，这里只负责把 provider 传进去。
        self.prompt_lang: str = resolve_prompt_lang(
            prompt_lang, provider=getattr(self.advisor_cfg, "provider", "") or ""
        )
        # 中文路径下 translator 根本不会被调用，但仍然构造一个——这样
        # "这个 Agent 的回译后端是什么"始终有确定答案，不必在两条路径上分别判断。
        self.translator: Any = (
            translator if translator is not None
            else build_translator(translator_mode, client=load_translator_client())
        )
        #: 最近一次 run() 的回译明细，供 AgentResult 透出。
        self.translation_notes: list[str] = []
        self.translation_backend: str = getattr(self.translator, "backend", "")
        #: 最近一次 run() 的可信度事实与由此渲染的模板句。
        #: 在 __init__ 里给出默认值，是为了让 _report_from_parsed 这类
        #: 可能被单独调用的方法在没有先跑 _generate_advisories 时也不炸。
        self._confidence_facts: dict[str, Any] = {}
        self._confidence_note_default: str = ""

    def describe_backends(self) -> dict[str, str]:
        """返回两个角色的后端描述，供启动日志与 demo 打印。"""
        return {
            "scheduler": describe_backend(self.scheduler_cfg),
            "advisor": describe_backend(self.advisor_cfg),
            "prompt_lang": self.prompt_lang,
            "translator": describe_translator(self.translator),
        }

    # ---------------- 内部工具 ----------------

    def _log(self, kind: str, title: str, detail: str = "",
             data: dict[str, Any] | None = None) -> None:
        self.trace.append(TraceStep(
            step=len(self.trace) + 1, kind=kind, title=title, detail=detail, data=data,
        ))

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """执行一次工具调用。任何异常都转成结构化错误，绝不让 Agent 断链。"""
        if "__parse_error__" in arguments:
            # 模型把 arguments 写成了非法 JSON（llm.py 只做记号，不做猜测）。
            # 绝不能把记号当关键字参数塞进工具函数——那会变成
            # "unexpected keyword argument '__parse_error__'" 这种对外看不懂的报错。
            raw = str(arguments.get("__parse_error__") or "")
            return {
                "status": "error",
                "error_code": "BAD_JSON_ARGUMENTS",
                "error_message": (
                    f"工具参数不是合法 JSON（原文：{raw[:120]}），本轮调用没有执行；"
                    f"请重新调用 {name}，并把 arguments 写成 JSON 对象。"
                ),
            }
        if name not in self.tool_handlers:
            return {
                "status": "error",
                "error_code": "UNKNOWN_TOOL",
                "error_message": f"未注册的工具 {name}；可用工具：{', '.join(self.tool_handlers)}",
            }
        try:
            return self.tool_handlers[name](**arguments)
        except TypeError as exc:
            # 参数名错/多余参数——这是模型最常见的出错方式，要把可用参数名告诉它
            return {
                "status": "error",
                "error_code": "BAD_ARGUMENTS",
                "error_message": f"参数不匹配：{exc}",
            }
        except Exception as exc:                                   # noqa: BLE001
            return {
                "status": "error",
                "error_code": "TOOL_CRASHED",
                "error_message": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _args_key(name: str, arguments: dict[str, Any]) -> str:
        return f"{name}::{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"

    def _required_tools(self, inp: AgentInput) -> set[str]:
        """输入里提供了数据 → 这些工具就必须被调用。"""
        required: set[str] = set()
        if inp.water_readings:
            required.add("water_risk_predict")
        if inp.image_path:
            required.add("fish_disease_detect")
        return required

    # ---------------- ① 调度循环 ----------------

    def _direct_dispatch(self, inp: AgentInput) -> dict[str, dict[str, Any]]:
        """不问模型，直接按输入跑确定性工具。

        参数与 :meth:`_apply_guardrail` 的补调完全一致——同一条路径，两种触发方式：
        一种是"模型漏调了"，一种是"根本不用问模型"。前者会被记成降级，
        后者是正常快路径，不该出现在 ``degraded_reasons`` 里。
        """
        results: dict[str, dict[str, Any]] = {}
        names = sorted(self._required_tools(inp))
        # kind 用 tool_call：TraceStep.kind 是封闭 Literal，前端按这几个值渲染；
        # 快路径本身就是"确定性地调工具"，不需要新增一种步骤类型。
        self._log("tool_call", f"输入已带齐数据，跳过调度 LLM，直接执行 {len(names)} 个工具",
                  f"工具：{'、'.join(names)}（省掉 1–2 轮模型决策）")
        if "water_risk_predict" in names:
            results["water_risk_predict"] = self._execute_tool("water_risk_predict", {
                "readings": inp.water_readings,
                "pond_id": inp.pond_id,
                "species": inp.species,
            })
        if "fish_disease_detect" in names:
            results["fish_disease_detect"] = self._execute_tool("fish_disease_detect", {
                "image_path": inp.image_path,
                "mock_source": inp.mock_source,
            })
        for name in names:
            payload = results.get(name, {})
            self._log("tool_result", f"{name}（直连）返回 status={payload.get('status')}",
                      str(payload.get("error_message") or payload.get("degradation_summary")
                          or payload.get("disease_type") or "")[:200])
        return results

    def _scheduling_loop(self, inp: AgentInput) -> dict[str, dict[str, Any]]:
        """跑 LLM 工具调用循环，返回 {工具名: 工具结果}。"""
        ctx = {
            "pond_id": inp.pond_id,
            "species": inp.species,
            "water_readings": inp.water_readings,
            "image_path": inp.image_path,
            "mock_source": inp.mock_source,
        }
        messages: list[dict[str, Any]] = [
            # 上下文以 ##CTX## 标记附在 system 尾部：
            # 真实模型看不到也不受影响，离线 MockLLM 靠它复现同样的工具决策。
            {"role": "system",
             "content": scheduler_system_prompt(self.prompt_lang)
             + build_scheduler_context(ctx)},
            {"role": "user", "content": self._render_user_message(inp)},
        ]

        results: dict[str, dict[str, Any]] = {}
        seen_calls: set[str] = set()
        self._bad_json_calls = set()

        for round_idx in range(1, self.max_rounds + 1):
            try:
                resp = self.scheduler.chat(messages, tools=self.tool_schemas)
            except LLMUnavailableError as exc:
                self._log("error", f"调度 LLM 不可用（第 {round_idx} 轮）", str(exc))
                self.degraded_reasons.append(f"调度 LLM 不可用：{exc}")
                break

            self._log("llm", f"调度 LLM 第 {round_idx} 轮决策",
                      f"finish_reason={resp.finish_reason}，"
                      f"tool_calls={len(resp.tool_calls)}",
                      {"content": (resp.content or "")[:200]})

            if not resp.tool_calls:
                break

            messages.append(self._assistant_msg(resp))

            for call in resp.tool_calls:
                key = self._args_key(call.name, call.arguments)
                if key in seen_calls:
                    # 防死循环：拒绝重复调用，但要把"拒绝原因"回灌，否则模型会一直重试
                    self._log("guardrail", f"拒绝重复调用 {call.name}", "参数与上一次完全相同")
                    payload = {
                        "status": "error",
                        "error_code": "DUPLICATE_CALL",
                        "error_message": "该工具已用完全相同的参数调用过，请直接使用已有结果。",
                    }
                else:
                    seen_calls.add(key)
                    self._log("tool_call", f"调用工具 {call.name}",
                              json.dumps(call.arguments, ensure_ascii=False)[:300])
                    payload = self._execute_tool(call.name, call.arguments)
                    if payload.get("error_code") == "BAD_JSON_ARGUMENTS":
                        # 参数是坏的 ⇒ 这一轮等于没调成，不算 results，
                        # 留给护栏用确定性输入补调（否则水质分析会静默缺席）。
                        self._bad_json_calls.add(call.name)
                    else:
                        results[call.name] = payload
                    self._log("tool_result", f"{call.name} 返回 status={payload.get('status')}",
                              str(payload.get("error_message") or payload.get("degradation_summary")
                                  or payload.get("disease_type") or "")[:200])

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                })
        else:
            self._log("error", f"调度循环达到上限 {self.max_rounds} 轮",
                      "疑似模型陷入循环，已强制中断")
            self.degraded_reasons.append(f"调度循环超过 {self.max_rounds} 轮上限")

        return results

    def _render_user_message(self, inp: AgentInput) -> str:
        parts = [f"塘口编号：{inp.pond_id}", f"养殖品种：{inp.species}"]
        if inp.water_readings:
            parts.append(f"水质时序数据共 {len(inp.water_readings)} 条读数，"
                         f"首条：{json.dumps(inp.water_readings[0], ensure_ascii=False)}"
                         f"，末条：{json.dumps(inp.water_readings[-1], ensure_ascii=False)}")
        else:
            parts.append("本次没有水质数据。")
        parts.append(f"鱼体图片路径：{inp.image_path}" if inp.image_path else "本次没有鱼体图片。")
        if inp.question:
            parts.append(f"用户补充提问：{inp.question}")
        parts.append("请调用必要的工具完成任务。")
        return "\n".join(parts)

    @staticmethod
    def _json_arguments(call: Any) -> str:
        """回灌给后端的 ``function.arguments`` 必须是**合法 JSON 字符串**。

        DashScope 见到坏 JSON 会把整轮请求打成 400（"The function.arguments
        parameter of the code model must be in JSON format"），也就是说模型自己
        写坏一条参数，后面的轮次会全部连坐。这里兜底成 ``{}``：坏原文已经通过
        tool 消息回灌给模型了，不需要再让后端因它整轮失败。
        """
        raw = getattr(call, "raw_arguments", "") or ""
        if raw.strip():
            try:
                if isinstance(json.loads(raw), dict):
                    return raw
            except json.JSONDecodeError:
                pass
        data = {k: v for k, v in (getattr(call, "arguments", None) or {}).items()
                if k != "__parse_error__"}
        return json.dumps(data, ensure_ascii=False)

    def _assistant_msg(self, resp: Any) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": resp.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name,
                                 "arguments": self._json_arguments(c)},
                }
                for c in resp.tool_calls
            ],
        }

    # ---------------- ② 护栏补调 ----------------

    def _apply_guardrail(self, inp: AgentInput, results: dict[str, dict[str, Any]]) -> None:
        missing = self._required_tools(inp) - set(results)
        if not missing:
            return

        bad = sorted(missing & self._bad_json_calls)
        note = f"其中 {'、'.join(bad)} 的调用参数不是合法 JSON，未被采纳" if bad else ""
        self._log("guardrail", f"LLM 漏调 {len(missing)} 个必需工具，调度层补调",
                  f"补调：{'、'.join(sorted(missing))}（说明模型本轮工具选择不完整）"
                  + (f"；{note}" if note else ""))
        degraded = f"LLM 漏调工具，已由护栏补调：{'、'.join(sorted(missing))}"
        if bad:
            degraded += f"（{'、'.join(bad)} 的调用参数不是合法 JSON，已改用原始输入重算）"
        self.degraded_reasons.append(degraded)

        if "water_risk_predict" in missing:
            results["water_risk_predict"] = self._execute_tool("water_risk_predict", {
                "readings": inp.water_readings,
                "pond_id": inp.pond_id,
                "species": inp.species,
            })
        if "fish_disease_detect" in missing:
            results["fish_disease_detect"] = self._execute_tool("fish_disease_detect", {
                "image_path": inp.image_path,
                "mock_source": inp.mock_source,
            })
        for name in missing:
            payload = results.get(name, {})
            self._log("tool_result", f"{name}（护栏）返回 status={payload.get('status')}",
                      str(payload.get("error_message") or "")[:200])

    # ---------------- ③.5 RAG 检索 ----------------

    def _ensure_knowledge_base(self) -> Any | None:
        """惰性加载知识库。**只打开，不构建**。"""
        if self._knowledge_tried:
            return self.knowledge_base
        self._knowledge_tried = True
        try:
            from ..rag.retriever import open_knowledge_base

            self.knowledge_base = open_knowledge_base(auto_build=False)
        except Exception as exc:  # noqa: BLE001 - 知识库缺失/依赖未装/维度不匹配……
            # 知识库不可用**不该让整个 Agent 挂掉**：
            # 水质与图像工具仍然能给出正确结论，只是措施不够具体。
            self.knowledge_base = None
            self.degraded_reasons.append(f"知识库不可用，建议未引用规范依据：{exc}")
        return self.knowledge_base

    def _retrieve_knowledge(
        self,
        water: dict[str, Any] | None,
        vision: dict[str, Any] | None,
        fusion: FusionResult,
        use: bool | None = None,
    ) -> Any | None:
        """按融合结论定向检索知识库。任何异常都降级为空，不打断主链路。

        ``use`` 是**本次请求**的开关（来自 ``AgentInput.use_knowledge``），
        None 表示沿用实例默认值。做成参数而不是临时改 ``self.use_knowledge``，
        是因为 ``AquacultureAgent`` 会被多个请求共享：改实例属性会让并发的
        另一个请求跟着变，那种 bug 只在并发下出现，极难复现。
        """
        effective = self.use_knowledge if use is None else use
        if not effective:
            return None
        kb = self._ensure_knowledge_base()
        if kb is None:
            self._log("guardrail", "跳过 RAG 检索（知识库不可用）",
                      "建议将仅依据工具返回值生成")
            return None
        try:
            ctx = kb.retrieve_for_agent(
                water=water, vision=vision, fusion=fusion.model_dump(mode="json")
            )
        except Exception as exc:  # noqa: BLE001 - 检索失败降级为空，不打断主链路
            self.knowledge_base = None
            self.degraded_reasons.append(f"知识库检索失败，建议未引用规范依据：{exc}")
            self._log("error", "RAG 检索失败", str(exc))
            return None

        self._log(
            "tool_result",
            f"知识库检索：{ctx.n_chunks} 条（{'、'.join(ctx.by_purpose) or '无'}）",
            "；".join(f"{k}×{len(v)}" for k, v in ctx.by_purpose.items()),
            {"chunk_ids": [c.chunk_id for c in ctx.chunks]},
        )
        return ctx

    # ---------------- ③ 建议生成 ----------------

    def _generate_advisories(self, fusion: FusionResult,
                             water: dict[str, Any] | None,
                             vision: dict[str, Any] | None,
                             knowledge: Any | None = None,
                             elapsed: float = 0.0) -> AdvisorReport:
        """让建议角色写养殖建议；LLM 不可用时退回规则模板。

        无论走哪条路径，返回值都经过 :meth:`_report_from_parsed` 或
        :meth:`_fallback_advisories` 归一化为 :class:`AdvisorReport`，
        因此 ``risk_level`` 与 ``alerts`` **一定是融合层的确定性结果**，
        大模型的文本只能影响措辞，不能影响定级。
        """
        knowledge_block = "" if knowledge is None else knowledge.render()

        # 可信度说明的事实基础在这里算一次，供三条路径共用：规则模板、
        # 提示词里给模型看的素材、以及模型没写时的兜底。三条路径同一个来源，
        # 因此页面上那句话与提示词里那张素材表不可能互相矛盾。
        confidence_facts = build_confidence_facts(water, vision, fusion, knowledge)
        self._confidence_facts = confidence_facts
        self._confidence_note_default = render_confidence_note(confidence_facts, LANG_ZH)

        # 时间预算不够了就先降级：宁可给一份规则模板写成的建议（等级与告警本来就是
        # 融合层确定性算出来的，模型只影响措辞），也不要让前端等到超时。
        if self.budget_seconds and (elapsed + self.advisor_reserve_seconds
                                    > self.budget_seconds):
            self._log("advisor", "时间预算不足，跳过建议 LLM 改用规则模板",
                      f"已用 {elapsed:.1f} 秒 / 预算 {self.budget_seconds:.0f} 秒，"
                      f"给建议留 {self.advisor_reserve_seconds:.0f} 秒")
            self.degraded_reasons.append(
                f"响应时间预算不足（已用 {elapsed:.0f} 秒），养殖建议由规则模板生成"
            )
            return self._fallback_advisories(fusion, water, vision,
                                            confidence_note=self._confidence_note_default)

        # 没有配置真实建议 LLM 时，直接走规则模板。
        # 不要先去"调用" Mock 再判定它输出非法——那会把"本就没配 LLM"
        # 误报成"模型输出格式错误"，是两件完全不同的事。
        if isinstance(self.advisor, MockLLMClient):
            self._log("advisor", "未配置建议 LLM，使用规则模板生成养殖建议",
                      "规则模板的结论与结构化字段完全一致，不会引入模型幻觉")
            self.degraded_reasons.append("未配置建议 LLM，养殖建议由规则模板生成")
            return self._fallback_advisories(fusion, water, vision,
                                            confidence_note=self._confidence_note_default)

        try:
            resp = self.advisor.chat(
                [
                    {"role": "system", "content": advisor_system_prompt(self.prompt_lang)},
                    {"role": "user",
                     "content": build_advisor_user_message(
                         fusion.model_dump(mode="json"), water, vision,
                         knowledge_block=knowledge_block,
                         lang=self.prompt_lang,
                         confidence_facts=confidence_facts)},
                ],
                tools=None,
            )
        except LLMUnavailableError as exc:
            self._log("error", "建议 LLM 不可用，退回规则模板", str(exc))
            self.degraded_reasons.append(f"建议 LLM 不可用：{exc}")
            return self._fallback_advisories(fusion, water, vision,
                                            confidence_note=self._confidence_note_default)

        parsed = self._parse_advisor_output(resp.content or "")
        if parsed is None:
            self._log("error", "建议输出不是合法 JSON，退回规则模板",
                      (resp.content or "")[:200])
            self.degraded_reasons.append("建议 LLM 输出格式非法，已改用规则模板")
            return self._fallback_advisories(fusion, water, vision,
                                            confidence_note=self._confidence_note_default)

        parsed = self._back_translate(parsed, fusion, water, vision)
        report = self._report_from_parsed(parsed, fusion, water, vision)
        self._log("advisor", "建议角色生成养殖建议",
                  f"水质调控 {len(report.water_actions)} 条 / "
                  f"病害处置 {len(report.disease_actions)} 条",
                  parsed)
        return report

    def _back_translate(self, parsed: dict[str, Any],
                        fusion: FusionResult,
                        water: dict[str, Any] | None,
                        vision: dict[str, Any] | None) -> dict[str, Any]:
        """英文路径下把建议文本回译成中文；中文路径原样返回。

        三件事必须说清楚：

        1. **回译只碰自由文本**（``headline`` / ``observation_window`` /
           ``confidence_note`` / 各条 ``target``/``action``/``timing``）。
           ``risk_level`` 与 ``alerts`` 一个字都不动——它们要么是护栏要比对的
           标识符、要么是融合层的确定性结果，过一遍翻译只会引入风险。
        2. **译不出来就退回规则模板的同名字段，而不是保留英文。**
           留英文等于交付一份半英半中的报告，比少一句建议更糟。模板本来
           就与融合层同口径，退回它不会降低结论的正确性，只会降低措辞的丰富度。
        3. **每一次退回都记进 ``degraded_reasons`` 与 trace。**
           否则"这次报告其实是模板兜的"从文本上完全看不出来，等于静默降级。
        """
        self.translation_notes = []
        self.translation_backend = getattr(self.translator, "backend", "")
        if self.prompt_lang != LANG_EN:
            return parsed

        fallback = self._fallback_advisories(
            fusion, water, vision,
            confidence_note=self._confidence_note_default,
        ).model_dump(mode="json")
        outcome = back_translate_report(
            parsed, translator=self.translator, fallback=fallback, lang=self.prompt_lang,
        )
        self.translation_notes = list(outcome.notes)
        self.translation_backend = outcome.backend

        self._log(
            "advisor", f"英文建议回译（{outcome.backend}）",
            f"直译 {outcome.n_translated} 段 / 退回中文模板 {outcome.n_replaced} 段",
            {"backend": outcome.backend,
             "n_translated": outcome.n_translated,
             "n_replaced": outcome.n_replaced,
             "notes": list(outcome.notes)},
        )
        for note in outcome.notes:
            self.degraded_reasons.append(f"回译：{note}")
        return outcome.parsed

    def _report_from_parsed(self, parsed: dict[str, Any],
                            fusion: FusionResult,
                            water: dict[str, Any] | None = None,
                            vision: dict[str, Any] | None = None) -> AdvisorReport:
        """把大模型输出归一化成报告，并**用融合层强制回填**定级与告警。

        这里做三件护栏工作：

        1. ``risk_level`` 与 ``alerts`` 一律取融合层结果。提示词已要求模型原样
           复述，但仍然强制回填——模型把"预警"写成"高危"会让养殖户半夜白跑一趟，
           把"高危"写成"正常"则可能直接翻塘，这种字段不能有一丝信任。
        2. 模型复述的等级若与融合层不一致，**整份输出作废**，改用规则模板，
           并**记一条 guardrail 轨迹**而不是静默覆盖。理由见下方注释。
        3. 两条动作清单逐条过 :class:`Advisory` 校验，字段缺失或 priority 非法
           的条目被丢弃——宁可少一条建议，也不要抛异常炸掉整个 Agent 循环。
        """
        notes: list[str] = []

        echoed = str(parsed.get("risk_level", "")).strip()
        level_mismatch = bool(echoed) and echoed != fusion.risk_level
        if level_mismatch:
            notes.append(
                f"建议角色把风险等级写成「{echoed}」，与融合层的「{fusion.risk_level}」"
                f"不一致，其整份建议已作废，改用规则模板"
            )
        else:
            raw_alerts = parsed.get("alerts")
            if isinstance(raw_alerts, list) and raw_alerts:
                echoed_titles = {
                    str(a.get("title", "")).strip()
                    for a in raw_alerts if isinstance(a, dict)
                }
                expected_titles = {a.title for a in fusion.alerts}
                if echoed_titles != expected_titles:
                    notes.append(
                        "建议角色复述的告警与融合层不一致，已按融合层结果回填"
                    )
            elif fusion.alerts:
                notes.append("建议角色漏掉了告警信息，已按融合层结果补齐")

        for note in notes:
            self._log("guardrail", "建议输出与融合层不一致，已回填", note)
            self.degraded_reasons.append(note)

        if level_mismatch:
            # 等级写错说明模型**误读了输入**（把高危看成正常），那么它写的主结论、
            # 复查窗口、可靠性说明同样建立在错误前提上。
            # 只回填 risk_level 会得到一个自相矛盾的报告——顶部写"高危"、
            # 正文写"水质正常，一切良好"，这种报告比没有报告更危险。
            # 所以整份作废，退回规则模板。
            return self._fallback_advisories(fusion, water, vision,
                                            confidence_note=self._confidence_note_default)

        water_raw = parsed.get("water_actions")
        disease_raw = parsed.get("disease_actions")
        legacy = parsed.get("advisories")
        if not water_raw and not disease_raw and legacy:
            # 旧版提示词只有一个 advisories 数组，水质与病害混在一起、无法再拆分。
            # 整份建议因为字段改名而丢掉，代价远比归错类别大，所以归入水质块。
            water_raw = legacy

        return AdvisorReport(
            risk_level=fusion.risk_level,
            headline=str(parsed.get("headline", "")).strip(),
            alerts=list(fusion.alerts),
            water_actions=self._coerce_actions(water_raw),
            disease_actions=self._coerce_actions(disease_raw),
            observation_window=str(parsed.get("observation_window", "")).strip(),
            # 模型漏写可信度说明时，用同一套事实渲染的模板句补上，而不是留空。
            # 留空等于报告少一栏；由模型"大概写一句"又会与事实脱钩——
            # 这里宁可退回模板，因为模板说的每一条都能在 trace 里找到出处。
            confidence_note=(
                str(parsed.get("confidence_note", "")).strip()
                or self._confidence_note_default
            ),
        )

    @staticmethod
    def _coerce_actions(raw: Any) -> list[Advisory]:
        """把模型给的动作数组转成 Advisory 列表，丢弃不合法条目。

        同时接受旧字段名 ``advisories``：模型换版本、或有人手工构造 JSON 时，
        整份建议因为字段改名而全部丢失，代价太大。
        """
        if not isinstance(raw, list):
            return []
        out: list[Advisory] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            try:
                out.append(Advisory(
                    priority=int(item.get("priority", idx + 1)),
                    target=str(item.get("target", "")),
                    action=str(item.get("action", "")),
                    timing=str(item.get("timing", "")),
                ))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda a: a.priority)
        return out

    @staticmethod
    def _parse_advisor_output(text: str) -> dict[str, Any] | None:
        """解析建议角色的 JSON 输出，容忍 ```json 代码围栏。"""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
            stripped = "\n".join(lines).strip()
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    #: 红线规则编号 → 养殖户听得懂的说法。
    #: 规则编号（DO_REDLINE 之类）是给工程师看的，直接写进建议会让人一脸茫然；
    #: 编号本身仍然保留在 alerts[] 的 code 字段与 trace 里，需要追溯时查得到。
    _REDLINE_LABELS: ClassVar[dict[str, str]] = {
        "DO_REDLINE": "溶氧已跌破泛塘线",
        "DO_REDLINE_PROJECTED": "溶氧快跌破泛塘线",
        "DO_CRASH_TREND": "溶氧掉得太快",
        "NH3_REDLINE": "氨氮超过红线",
        "NO2_REDLINE": "亚硝酸盐超过红线",
        "PH_REDLINE": "pH 严重异常",
    }

    @classmethod
    def _fallback_advisories(cls, fusion: FusionResult,
                             water: dict[str, Any] | None,
                             vision: dict[str, Any] | None,
                             confidence_note: str = "") -> AdvisorReport:
        """规则模板建议器。保证没有 LLM 时 demo 依然有可读输出。

        措辞刻意与 :data:`ADVISOR_SYSTEM_PROMPT` 要求**同一套口径**：
        句子短、不说术语、每句都带动作。
        否则会出现一个很难解释的现象——没接大模型时建议很长很专业，
        接上大模型反而变短变通俗，评委会以为是模型变差了。
        风险等级与告警取自融合层，与 LLM 路径完全同源，因此两条路径的结构化字段
        **永远一致**，只有措辞不同。

        ``confidence_note`` 由调用方传入（通常是 :func:`render_confidence_note`
        按同一套事实渲染的结果）。不传就地按 ``water``/``vision``/``fusion`` 渲染，
        因此单独调用本方法也是自洽的——只是少了知识库那一路的不确定项，
        因为类方法拿不到实例上的检索结果。
        """
        water_actions: list[Advisory] = []
        disease_actions: list[Advisory] = []
        sev = (water or {}).get("severity_map") or {}

        # 「定级由哪一路证据抬起来」在整份模板里要用三次（水质动作兜底、标题、观察窗口），
        # 所以统一在这里算一次。退化成「预警就说水质」会产出自相矛盾的文案：
        # 水质满分、只查出烂鳃，标题却写「水质开始不对」，
        # 养殖户照着去调水，而真正该做的是捞病鱼、用药。
        wsev = sev
        water_abnormal = bool(
            (water or {}).get("status") == "ok"
            and ((water or {}).get("risk_level") in ("预警", "高危")
                 or (water or {}).get("redline_hits")
                 or any((v or 0) > 0 for v in wsev.values())
                 or (water or {}).get("projected_do_alert_eta_hours") is not None)
        )
        disease_visible = bool(
            vision
            and vision.get("status") == "ok"
            # is_fish=False 表示「图里没检测到鱼」：此时 disease_type 为 None，
            # 不能当成「有病害」（旧代码只看 is_healthy，None 恰好是 falsy）。
            and vision.get("is_fish") is not False
            and not vision.get("is_healthy")
        )

        def add(store: list[Advisory], target: str, action: str, timing: str) -> None:
            store.append(Advisory(
                priority=len(store) + 1, target=target, action=action, timing=timing,
            ))

        hits = (water or {}).get("redline_hits") or []
        if hits:
            names = "、".join(cls._REDLINE_LABELS.get(h, h) for h in hits)
            add(water_actions, names,
                "全部增氧机马上开，停料，换掉 1/3 水，撒增氧剂。",
                "立刻做，至少守 6 小时")

        if (sev.get("dissolved_oxygen") or 0) > 0 or (water or {}).get("projected_do_alert_eta_hours") is not None:
            eta = (water or {}).get("projected_do_alert_eta_hours")
            threshold = (water or {}).get("projected_do_alert_threshold")
            timing = (f"{eta:.1f} 小时后会掉到 {threshold}，之前必须开机"
                      if eta is not None else "晚上 10 点开到早上 8 点")
            add(water_actions, "溶氧（水里的氧气）",
                "增氧机一直开着别停，后半夜最容易缺氧。",
                timing)

        if (sev.get("ammonia_n") or 0) > 0 or (sev.get("nitrite_n") or 0) > 0:
            offenders = [n for n, k in (("氨氮", "ammonia_n"), ("亚硝酸盐", "nitrite_n"))
                         if (sev.get(k) or 0) > 0]
            add(water_actions, "、".join(offenders),
                "停料 2–3 天，换掉两三成水，撒芽孢杆菌。",
                "今天开始，两天后复测")

        if not water_actions:
            if disease_visible:
                # 项目自己的知识库把"发病期停料"列为硬规则
                # （knowledge/corpus/05_投料管理规范.md:71「发病期应先停料 1–2 天再改投药饵」），
                # 所以"水质没有问题、只查出病害"时绝不能回一句"按常规量喂料"——
                # 那等于让养殖户给一塘病鱼正常投饵，加重氮负荷与死亡。
                add(water_actions, "投料",
                    "马上停料 1–2 天，之后改投药饵；剩料烂在水里会更缺氧。",
                    "今天开始")
            else:
                add(water_actions, "当前无异常",
                    "按常规量喂料，夜里开增氧机，早晚各巡塘一次。",
                    "照常做")

        if (
            vision
            and vision.get("status") == "ok"
            and vision.get("is_fish") is not False
            and not vision.get("is_healthy")
        ):
            add(disease_actions,
                f"{vision.get('disease_type')}（把握 {vision.get('confidence')}）",
                "捞走病鱼单独养，按说明书撒药，同时送检。",
                "今天内做完")
            if vision.get("is_low_confidence"):
                add(disease_actions, "病害识别把握不大",
                    "重拍三张清楚的照片：侧面、鳃盖、病灶特写。",
                    "今天内")

        # 标题必须点名**真正把等级抬起来的那一路证据**（water_abnormal /
        # disease_visible 已在开头算好）。
        limiting = (water or {}).get("limiting_factor_label") or ""

        if fusion.risk_level == "正常":
            headline = "【正常】水好鱼也好，照常管理就行。"
        elif water_abnormal and fusion.risk_level == "高危":
            headline = f"【高危】{limiting or '塘口'}出问题了，马上处理。"
        elif water_abnormal:
            headline = f"【预警】{limiting or '水质'}开始不对，今天盯紧点。"
        elif disease_visible and fusion.risk_level == "高危":
            headline = f"【高危】{vision.get('disease_type')}很重，马上处理。"
        elif disease_visible:
            headline = f"【预警】查出{vision.get('disease_type')}，今天盯紧点。"
        elif fusion.risk_level == "高危":
            headline = "【高危】塘口出问题了，马上处理。"
        else:
            headline = "【预警】水质开始不对，今天盯紧点。"

        # 观察窗口同样要跟着证据走：只查出病害时让养殖户"复测溶氧和氨氮"是答非所问，
        # 该盯的是死鱼数量和病灶变化。
        if water_abnormal:
            observation_window = "6 小时后复测溶氧和氨氮，明天看鱼有没有好转。"
        elif disease_visible:
            observation_window = "每天早晚各捞一次死鱼并记数，3 天后复看病灶有没有扩大。"
        else:
            observation_window = "明天早晚各巡塘一次，看吃食和浮头情况。"

        return AdvisorReport(
            risk_level=fusion.risk_level,
            headline=headline,
            alerts=list(fusion.alerts),
            water_actions=water_actions,
            disease_actions=disease_actions,
            observation_window=observation_window,
            # 可信度说明照模板生成：事实由规则算，措辞由模板给。
            # 接了大模型时这句话由模型照同一张素材表重写（见 _generate_advisories），
            # 这里是没有模型可用时的同结构降级版本。
            confidence_note=confidence_note or render_confidence_note(
                build_confidence_facts(water, vision, fusion),
                LANG_ZH,
            ),
        )

    # ---------------- ④ 主入口 ----------------

    def run(self, inp: AgentInput) -> AgentResult:
        """执行一次完整决策。"""
        started = time.perf_counter()
        self.trace = []
        self.degraded_reasons = []
        # 回译明细也必须每次清空：否则上一次的"回译失败"会挂在下一次结果上，
        # 而这一次可能压根没走英文路径。
        self.translation_notes = []
        self.translation_backend = getattr(self.translator, "backend", "")
        # 可信度素材同理：它由本次的模态与检索结果算出，留着上次的会写错证据。
        self._confidence_facts = {}
        self._confidence_note_default = ""

        self._log("llm", "启动 Agent 调度", inp.summary())

        # ① 工具执行：输入已带齐数据就走确定性直连，否则让模型决定调什么
        if self.direct_tools and self._required_tools(inp):
            results = self._direct_dispatch(inp)
        else:
            results = self._scheduling_loop(inp)
        # ② 护栏补调
        self._apply_guardrail(inp, results)

        water = results.get("water_risk_predict")
        vision = results.get("fish_disease_detect")

        # ③ 确定性融合
        fusion = fuse_assessment(water, vision)
        self._log("fusion", f"融合决策：{fusion.risk_level}",
                  fusion.reason,
                  fusion.model_dump(mode="json"))

        # ④ RAG 检索：把"该怎么做"的依据取出来
        ctx = self._retrieve_knowledge(water, vision, fusion, inp.use_knowledge)

        # ⑤ 建议生成
        report: AdvisorReport = self._generate_advisories(
            fusion, water, vision, ctx,
            elapsed=time.perf_counter() - started,
        )

        # 报告里的等级与告警一定等于融合层的结论（见 _report_from_parsed 的护栏），
        # 下面这行是断言而非赋值，若哪天真不一致，我们宁可当场炸掉也不要静默发错告警。
        assert report.risk_level == fusion.risk_level, "报告等级被改写，护栏失效"
        advisories = report.all_actions()

        degraded = bool(fusion.degraded_mode or self.degraded_reasons
                        or (water or {}).get("degraded_mode")
                        or (vision or {}).get("degraded_mode"))

        references: list[str] = (
            [c.reference_line() for c in ctx.chunks] if ctx is not None else []
        )

        return AgentResult(
            pond_id=inp.pond_id,
            risk_level=fusion.risk_level,
            fused_score=fusion.fused_score,
            reason=fusion.reason,
            alerts=[a.model_dump(mode="json") for a in fusion.alerts],
            advisories=advisories,
            headline=report.headline,
            observation_window=report.observation_window,
            confidence_note=report.confidence_note,
            evidence=[e.model_dump(mode="json") for e in fusion.evidence],
            water=water,
            vision=vision,
            fusion=fusion.model_dump(mode="json"),
            knowledge_block="" if ctx is None else ctx.render(),
            references=references,
            knowledge_summary={} if ctx is None else {
                "backend": ctx.backend,
                "n_chunks": ctx.n_chunks,
                "degraded": ctx.degraded,
                "warnings": list(ctx.warnings),
                "purposes": {k: len(v) for k, v in ctx.by_purpose.items()},
            },
            report=report.model_dump(mode="json"),
            trace=self.trace,
            llm_backends={
                "scheduler": describe_backend(self.scheduler_cfg),
                "advisor": describe_backend(self.advisor_cfg),
            },
            degraded_mode=degraded,
            degraded_reasons=list(self.degraded_reasons),
            elapsed_seconds=round(time.perf_counter() - started, 3),
            prompt_lang=self.prompt_lang,
            translation_backend=self.translation_backend,
            translation_notes=list(self.translation_notes),
        )


def run_agent(water_readings: list[dict[str, Any]] | None = None,
              image_path: str | None = None,
              pond_id: str = "unknown",
              species: str = "鲤科",
              question: str | None = None,
              mock_source: str = "hash",
              use_knowledge: bool | None = None) -> AgentResult:
    """一次性调用的便捷封装。

    Examples:
        >>> r = run_agent(water_readings=[{"timestamp": "2026-05-11T08:00:00",
        ...     "water_temp": 28.0, "ph": 7.8, "dissolved_oxygen": 4.5,
        ...     "ammonia_n": 0.2} for _ in range(24)], pond_id="P01")
        >>> r.risk_level in ("正常", "预警", "高危")
        True
    """
    agent = AquacultureAgent(use_knowledge=use_knowledge)
    return agent.run(AgentInput(
        pond_id=pond_id, species=species, water_readings=water_readings,
        image_path=image_path, question=question, mock_source=mock_source,
    ))


__all__ = [
    "KNOWLEDGE_DISABLE_ENV",
    "MAX_ROUNDS",
    "AdvisorReport",
    "Advisory",
    "AgentInput",
    "AgentResult",
    "AquacultureAgent",
    "TraceStep",
    "run_agent",
]
