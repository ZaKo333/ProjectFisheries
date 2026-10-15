"""LLM 后端层 —— OpenAI 兼容协议的 function calling 客户端。

为什么自己写而不引入 LangGraph / LlamaIndex
=============================================
1. **协议即接口**：Ollama、vLLM、DeepSeek、OpenAI、DashScope 全部提供
   OpenAI 兼容的 `/chat/completions`。自己写一层薄客户端，换模型只是改一个环境变量；
   引入框架反而把"换模型"这件事绑死在框架的适配层上。
2. **零依赖可跑**：本模块只用标准库 `urllib`，不依赖 `openai` / `httpx`。
   评委机器上没网、没装包也能跑通 demo —— 这对比赛是刚需。
3. **链路是线性的**：本项目是「收数据 → 调工具 → 融合 → 出建议」四步，
   LangGraph 的价值在复杂多智能体状态机上，这里引入纯属依赖风险。

双角色架构（本项目对 AQUA-7B 短板的核心应对）
================================================
实测 `KurmaAI/AQUA-7B` 的两个短板：
  - 训练数据 `language: en`，纯英文，中文输出质量无保证；
  - 微调数据是水产问答，**没有工具调用训练**，能否稳定走 function calling 未经验证。

而"决定调哪个工具"恰恰是整条链路最不能出错的一步。所以本模块把 LLM 拆成两个角色：

  - **scheduler（调度角色）**：必须可靠地输出 function call。
    推荐 DeepSeek / Qwen 等中文工具调用成熟的模型。
  - **advisor（建议角色）**：把已算好的结构化结论翻译成专业养殖建议。
    这里才是 AQUA-7B 的用武之地 —— 它的水产领域知识是真的。

两个角色可用不同模型，通过 `FISHERIES_LLM_SCHEDULER_*` / `FISHERIES_LLM_ADVISOR_*`
环境变量分别配置，未配置时回落到通用的 `FISHERIES_LLM_*`。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 一、配置
# --------------------------------------------------------------------------- #

Role = Literal["scheduler", "advisor"]


class LLMConfig(BaseModel):
    """LLM 后端配置。"""

    provider: str = "mock"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    role: str = "generic"

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock" or not self.base_url


#: 预置后端。key 为 provider 名。
#: 注意 `aqua-7b-ollama` / `aqua-7b-vllm` 用的是社区量化版本
#: （`DevQuasar/KurmaAI.AQUA-7B-GGUF`），因为 7B fp16 需要约 15GB 显存，
#: 而本机 RTX 5070 Laptop 只有 8GB，必须用量化权重。
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "mock": {"base_url": "", "model": "mock-deterministic", "api_key": ""},
    "aqua-7b-ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "aqua-7b",
        "api_key": "ollama",           # Ollama 不校验 key，但字段不能为空
    },
    "aqua-7b-vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "KurmaAI/AQUA-7B",
        "api_key": "EMPTY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
}


def load_llm_config(role: Role = "scheduler", env: dict[str, str] | None = None) -> LLMConfig:
    """从环境变量装载 LLM 配置。

    优先级：角色专属变量 > 通用变量 > provider 预置默认值。

    Args:
        role: `scheduler` 或 `advisor`。
        env: 注入的环境字典（测试用），默认读 `os.environ`。

    Returns:
        LLMConfig。

    Examples:
        >>> cfg = load_llm_config("advisor", env={"FISHERIES_LLM_PROVIDER": "mock"})
        >>> cfg.is_mock
        True
    """
    e = dict(os.environ if env is None else env)
    prefix = f"FISHERIES_LLM_{role.upper()}_"

    def pick(suffix: str, default: str = "") -> str:
        return e.get(prefix + suffix) or e.get("FISHERIES_LLM_" + suffix) or default

    provider = pick("PROVIDER", "mock")
    preset = PROVIDER_PRESETS.get(provider, {})

    api_key = pick("API_KEY") or e.get(preset.get("api_key_env", ""), "") or preset.get("api_key", "")
    base_url = pick("BASE_URL") or preset.get("base_url", "")
    model = pick("MODEL") or preset.get("model", "")

    def pick_num(suffix: str, default: float) -> float:
        raw = pick(suffix)
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=pick_num("TEMPERATURE", 0.2),
        max_tokens=int(pick_num("MAX_TOKENS", 1024)),
        timeout=pick_num("TIMEOUT", 60.0),
        role=role,
    )


# --------------------------------------------------------------------------- #
# 二、消息与响应契约
# --------------------------------------------------------------------------- #

class ToolCall(BaseModel):
    """一次工具调用请求。`arguments` 已从 JSON 字符串解析为 dict。"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str = ""


class ChatResponse(BaseModel):
    """归一化后的模型响应。"""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict, description="原始响应，排障用")


class LLMUnavailableError(RuntimeError):
    """网络不可达 / 鉴权失败 / 后端未启动。调用方可据此降级到 Mock。"""


class ChatClient(Protocol):
    """聊天客户端协议。"""

    def chat(self,
             messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        ...


# --------------------------------------------------------------------------- #
# 三、OpenAI 兼容客户端（仅标准库）
# --------------------------------------------------------------------------- #

class OpenAICompatibleClient:
    """通过 `/chat/completions` 调用任何 OpenAI 兼容后端。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        if not config.base_url:
            raise LLMUnavailableError("base_url 为空，无法发起请求")

    def chat(self,
             messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:                      # 4xx/5xx
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMUnavailableError(f"HTTP {exc.code}：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMUnavailableError(f"无法连接 {url}：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMUnavailableError(f"响应不是合法 JSON：{exc}") from exc

        return self._parse(body)

    @staticmethod
    def _parse(body: dict[str, Any]) -> ChatResponse:
        choices = body.get("choices") or []
        if not choices:
            raise LLMUnavailableError(f"响应缺少 choices 字段：{str(body)[:300]}")

        msg = choices[0].get("message") or {}
        calls: list[ToolCall] = []
        for idx, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                # 工具调用参数不是合法 JSON：保留原文，交由调度层报错，
                # 不要在这里静默当成空参数，否则会掩盖模型的格式问题。
                args = {"__parse_error__": raw_args}
            calls.append(ToolCall(
                id=tc.get("id") or f"call_{idx}",
                name=fn.get("name") or "",
                arguments=args,
                raw_arguments=raw_args if isinstance(raw_args, str) else "",
            ))

        return ChatResponse(
            content=msg.get("content"),
            tool_calls=calls,
            finish_reason=choices[0].get("finish_reason") or "",
            model=body.get("model") or "",
            usage=body.get("usage") or {},
            raw=body,
        )


# --------------------------------------------------------------------------- #
# 四、确定性 Mock 客户端（离线 demo 的主路径）
# --------------------------------------------------------------------------- #

class MockLLMClient:
    """不联网的确定性客户端，用于离线演示与单元测试。

    它**真实地执行 function calling 协议**：先根据用户输入发出工具调用，
    拿到 `role="tool"` 结果后再产出最终答复。因此用它跑通的调度链路，
    换成真实模型时链路本身是已验证的。

    行为完全由消息历史决定，不含任何随机成分：
      第一次调用 → 有什么数据就调什么工具（可一次发多个）
      工具结果齐了 → 返回最终答复
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig(provider="mock", model="mock-deterministic")
        self._call_seq = 0

    def chat(self,
             messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> ChatResponse:
        available = {t["function"]["name"] for t in (tools or [])}
        called = {
            m.get("name") for m in messages
            if m.get("role") == "tool" and m.get("name")
        }
        user_ctx = self._user_context(messages)

        # ---- 阶段一：补齐尚未调用且数据允许调用的工具 ----
        pending: list[ToolCall] = []
        if "water_risk_predict" in available and "water_risk_predict" not in called:
            readings = user_ctx.get("water_readings")
            if readings:
                self._call_seq += 1
                pending.append(ToolCall(
                    id=f"mock_call_{self._call_seq}",
                    name="water_risk_predict",
                    arguments={
                        "readings": readings,
                        "pond_id": user_ctx.get("pond_id", "unknown"),
                        "species": user_ctx.get("species", "鲤科"),
                    },
                ))
        if "fish_disease_detect" in available and "fish_disease_detect" not in called:
            image = user_ctx.get("image_path")
            if image:
                self._call_seq += 1
                pending.append(ToolCall(
                    id=f"mock_call_{self._call_seq}",
                    name="fish_disease_detect",
                    arguments={"image_path": image,
                               "mock_source": user_ctx.get("mock_source", "hash")},
                ))

        if pending:
            return ChatResponse(
                content=None, tool_calls=pending, finish_reason="tool_calls",
                model=self.config.model,
            )

        # ---- 阶段二：给出最终答复 ----
        return ChatResponse(
            content="[MockLLM] 已完成工具调用与融合分析，详见结构化输出。",
            tool_calls=[], finish_reason="stop", model=self.config.model,
        )

    @staticmethod
    def _user_context(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """从消息里回收调度层注入的上下文（system 消息中的 JSON 块）。"""
        for msg in messages:
            if msg.get("role") != "system":
                continue
            content = msg.get("content") or ""
            marker = "##CTX##"
            if marker in content:
                try:
                    return json.loads(content.split(marker, 1)[1])
                except json.JSONDecodeError:
                    return {}
        return {}


# --------------------------------------------------------------------------- #
# 五、工厂
# --------------------------------------------------------------------------- #

def build_client(config: LLMConfig) -> ChatClient:
    """按配置构造客户端。mock 或 base_url 为空时返回 MockLLMClient。"""
    if config.is_mock:
        return MockLLMClient(config)
    return OpenAICompatibleClient(config)


def describe_backend(config: LLMConfig) -> str:
    """返回人类可读的后端描述，用于启动日志与 demo 输出。"""
    if config.is_mock:
        return f"{config.role}: MockLLM（确定性离线模式）"
    return f"{config.role}: {config.provider} / {config.model} @ {config.base_url}"


__all__ = [
    "ChatClient",
    "ChatResponse",
    "LLMConfig",
    "LLMUnavailableError",
    "MockLLMClient",
    "OpenAICompatibleClient",
    "PROVIDER_PRESETS",
    "Role",
    "ToolCall",
    "build_client",
    "describe_backend",
    "load_llm_config",
]
