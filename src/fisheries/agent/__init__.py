"""Agent 层 —— 调度、提示词、融合决策。

模块职责
==========
    fusion.py         M4 多源融合与风险决策（确定性规则，不交给大模型）
    llm.py            OpenAI 兼容协议的 function calling 客户端 + 离线 Mock
    prompts.py        调度角色与建议角色的提示词
    orchestrator.py   M6 Agent 主调度循环

快速上手
==========
    >>> from fisheries.agent import AquacultureAgent, AgentInput
    >>> agent = AquacultureAgent()                       # 未配置 LLM 时自动用离线 Mock
    >>> result = agent.run(AgentInput(pond_id="P01", image_path="fish.jpg"))
    >>> result.risk_level in ("正常", "预警", "高危")
    True
"""

from .fusion import Alert, Evidence, FusionResult, fuse_assessment
from .llm import (
    ChatResponse,
    LLMConfig,
    LLMUnavailableError,
    MockLLMClient,
    OpenAICompatibleClient,
    ToolCall,
    build_client,
    describe_backend,
    load_llm_config,
)
from .orchestrator import (
    AdvisorReport,
    Advisory,
    AgentInput,
    AgentResult,
    AquacultureAgent,
    TraceStep,
    run_agent,
)

__all__ = [
    "AdvisorReport",
    "Advisory",
    "AgentInput",
    "AgentResult",
    "Alert",
    "AquacultureAgent",
    "ChatResponse",
    "Evidence",
    "FusionResult",
    "LLMConfig",
    "LLMUnavailableError",
    "MockLLMClient",
    "OpenAICompatibleClient",
    "ToolCall",
    "TraceStep",
    "build_client",
    "describe_backend",
    "fuse_assessment",
    "load_llm_config",
    "run_agent",
]
