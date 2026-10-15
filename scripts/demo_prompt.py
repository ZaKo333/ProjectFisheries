"""建议角色提示词演示 —— 直接看提示词长什么样、模型该吐什么。

本脚本回答三个问题，全部用真实代码跑出来、不是贴文档：

    场景一  系统提示词到底写了什么？多少个字？约束落在哪几句？
    场景二  真正送给模型的用户消息长什么样（含 RAG 知识块）？
    场景三  一个"听话的"建议角色输出 → 结构化报告
            一个"不听话的"建议角色输出（把高危改成正常）→ 护栏把它按回去

关于"样例模型"：本机没有 API key，也不该为了演示去联网。
所以场景三用两个**手写**的 JSON 冒充模型输出，分别代表
"按提示词要求写的"与"违反提示词的"两种情况。
它们在代码里叫 :class:`ScriptedAdvisor`，只实现 ``chat()``，
与真实的 OpenAI 兼容客户端走**完全相同**的解析、校验、护栏路径。

用法：
    python scripts/demo_prompt.py
    python scripts/demo_prompt.py --only system      # 只看系统提示词
    python scripts/demo_prompt.py --only message     # 只看发给模型的消息
    python scripts/demo_prompt.py --only report      # 只看两条路径的结果对比
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.agent.llm import ChatResponse  # noqa: E402
from fisheries.agent.orchestrator import (  # noqa: E402
    AgentInput,
    AquacultureAgent,
)
from fisheries.agent.prompts import (  # noqa: E402
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_USER_TEMPLATE,
)

BAR = "=" * 78

# --------------------------------------------------------------------------- #
# 演示数据：24 小时持续恶化的水质 + 一张真实烂鳃图片
# --------------------------------------------------------------------------- #

def crashing_water(n: int = 24) -> list[dict[str, Any]]:
    """溶氧 6.2 → 2.6、氨氮 0.15 → 0.62、亚硝酸盐 0.05 → 0.15。"""
    out: list[dict[str, Any]] = []
    for i in range(n):
        r = i / (n - 1)
        out.append({
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": round(28.0 + 1.5 * r, 2),
            "ph": round(8.05 + 0.20 * r, 3),
            "dissolved_oxygen": round(6.2 - 3.6 * r, 3),
            "ammonia_n": round(0.15 + 0.47 * r, 3),
            "nitrite_n": round(0.05 + 0.10 * r, 3),
        })
    return out

GILL_IMAGE = _datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
)

class ScriptedAdvisor:
    """用固定 JSON 冒充建议角色，用来演示提示词与护栏。"""

    def __init__(self, payload: dict[str, Any], label: str) -> None:
        self.payload = payload
        self.label = label
        self.last_messages: list[dict[str, Any]] = []

    def chat(self, messages, tools=None):  # noqa: ANN001
        self.last_messages = messages
        return ChatResponse(
            content=json.dumps(self.payload, ensure_ascii=False, indent=2),
            finish_reason="stop",
        )

#: 一份"照着提示词写"的模型输出。注意全部采用养殖户口语、每条动作都很短。
GOOD_OUTPUT: dict[str, Any] = {
    "risk_level": "高危",
    "headline": "溶氧快见底了，先开增氧机。",
    "alerts": [
        {"level": "critical", "title": "水质红线告警",
         "detail": "预测 12 小时内溶氧跌破泛塘临界值"},
        {"level": "critical", "title": "高致病性病害告警",
         "detail": "检出烂鳃，置信度 0.86，属高致病性类别"},
    ],
    "water_actions": [
        {"priority": 1, "target": "溶氧", "action": "全部增氧机马上开，天亮前不要停。",
         "timing": "现在就开"},
        {"priority": 2, "target": "氨氮、亚硝酸盐", "action": "停料两天，换掉三成水。",
         "timing": "今天开始"},
    ],
    "disease_actions": [
        {"priority": 1, "target": "烂鳃", "action": "捞出病鱼单独养，按说明书撒药。",
         "timing": "今天内"},
    ],
    "observation_window": "6 小时后复测溶氧，明天看鱼还浮不浮头。",
    "confidence_note": "数值来自实测，可直接照着做。",
}

#: 一份"违反提示词"的模型输出：把高危悄悄改成正常，还自己编了一条告警。
BAD_OUTPUT: dict[str, Any] = {
    "risk_level": "正常",
    "headline": "水质正常，一切良好。",
    "alerts": [{"level": "info", "title": "模型自己编的告警", "detail": "无"}],
    "water_actions": [
        {"priority": 0, "target": "溶氧", "action": "建议加强管理。", "timing": "随时"},
        {"target": "氨氮", "action": "注意观察。", "timing": "随时"},
    ],
    "disease_actions": [],
    "observation_window": "无",
    "confidence_note": "正常",
}

# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #

def case_system() -> None:
    print(BAR)
    print("场景一  建议角色的系统提示词")
    print(BAR)
    lines = ADVISOR_SYSTEM_PROMPT.splitlines()
    print(f"（共 {len(ADVISOR_SYSTEM_PROMPT)} 字 / {len(lines)} 行，逐段打印）\n")
    for line in lines:
        print(f"  {line}")

    print(f"\n  -- 硬约束自检 --")
    checks = {
        "面向养殖户": "养殖户" in ADVISOR_SYSTEM_PROMPT,
        "禁止改数值": "一个字都不能改" in ADVISOR_SYSTEM_PROMPT,
        "禁止编造未测指标": "未测量" in ADVISOR_SYSTEM_PROMPT,
        "等级/告警只复述": "只能复述" in ADVISOR_SYSTEM_PROMPT,
        "动作≤40字": "40 个字" in ADVISOR_SYSTEM_PROMPT,
        "有正反例": "✓" in ADVISOR_SYSTEM_PROMPT and "✗" in ADVISOR_SYSTEM_PROMPT,
        "限定条数": "0–3 条" in ADVISOR_SYSTEM_PROMPT,
    }
    for name, ok in checks.items():
        print(f"     {'[OK]' if ok else '[MISSING]'} {name}")

def _run(payload: dict[str, Any], label: str):
    advisor = ScriptedAdvisor(payload, label)
    agent = AquacultureAgent(advisor_client=advisor, use_knowledge=True)
    result = agent.run(AgentInput(
        pond_id="P01",
        species="鲤科",
        water_readings=crashing_water(),
        image_path=str(GILL_IMAGE),
        mock_source="folder_hint",
    ))
    return advisor, result

def good_output_echoing_fusion() -> dict[str, Any]:
    """把 ``GOOD_OUTPUT`` 的告警改成**融合层真实给出的那几条**。

    ``GOOD_OUTPUT`` 里手写的两条告警是固定文案，而融合层实际会给出 5 条
    （红线 + 重病 + 3 条协同）。不补齐的话，一个"完全照着提示词写"的模型
    也会被判成"复述的告警与融合层不一致"，演示时反而看不清护栏的边界。

    做法是两趟：先跑一次拿到融合层告警，再填进 payload 重跑。
    """
    _, probe = _run(GOOD_OUTPUT, "probe")
    alerts = [
        {"level": str(a.get("level", "warning")),
         "title": str(a.get("title", "")),
         "detail": str(a.get("detail", ""))}
        for a in (probe.fusion.get("alerts") or [])
    ]
    payload = dict(GOOD_OUTPUT)
    payload["alerts"] = alerts
    return payload

def case_message() -> None:
    print(BAR)
    print("场景二  真正发给建议角色的用户消息（含 RAG 知识块）")
    print(BAR)
    advisor, _ = _run(GOOD_OUTPUT, "good")
    system, user = advisor.last_messages[0]["content"], advisor.last_messages[1]["content"]
    print(f"  messages[0].role = {advisor.last_messages[0]['role']}"
          f"   （{len(system)} 字）")
    print(f"  messages[1].role = {advisor.last_messages[1]['role']}"
          f"   （{len(user)} 字）")
    print(f"\n  -- 用户消息全文 --")
    for line in user.splitlines():
        print(f"  {line}")

    print(f"\n  -- 模板占位符自检（不该残留 {{{{'}}' 字样）--")
    leftovers = [p for p in ("{risk_level}", "{fused_score}", "{reason}",
                             "{water_block}", "{vision_block}", "{evidence_block}",
                             "{alert_block}", "{knowledge_block}")
                 if p in user]
    print(f"     残留占位符：{leftovers or '无'}")

def case_report() -> None:
    print(BAR)
    print("场景三  两条路径：听话的模型 vs 不听话的模型")
    print(BAR)

    for payload, label, expect in (
        (good_output_echoing_fusion(), "按提示词写的模型输出",
         "等级与告警应与融合层一致，无护栏告警"),
        (BAD_OUTPUT, "违反提示词的模型输出", "护栏应把等级/告警按回融合层并留痕"),
    ):
        print(f"\n{BAR}")
        print(f"  【{label}】预期：{expect}")
        print(BAR)
        advisor, result = _run(payload, label)

        print(f"\n  模型原始输出：")
        for line in json.dumps(payload, ensure_ascii=False, indent=2).splitlines():
            print(f"     {line}")

        print(f"\n  → 融合层判定：{result.risk_level}（融合分 {result.fused_score}）")
        print(f"  → 最终报告：")
        rep = result.report
        print(f"       risk_level      = {rep['risk_level']}")
        print(f"       headline        = {rep['headline']}")
        print(f"       alerts          = "
              f"{[a['title'] for a in rep['alerts']] or '（无）'}")
        print(f"       water_actions   = {len(rep['water_actions'])} 条")
        for a in rep["water_actions"]:
            print(f"           {a['priority']}. {a['target']} → {a['action']}"
                  f"（{a['timing']}）")
        print(f"       disease_actions = {len(rep['disease_actions'])} 条")
        for a in rep["disease_actions"]:
            print(f"           {a['priority']}. {a['target']} → {a['action']}"
                  f"（{a['timing']}）")
        print(f"       observation     = {rep['observation_window']}")
        print(f"       confidence_note = {rep['confidence_note']}")

        guards = [t for t in result.trace if t.kind == "guardrail"]
        print(f"\n  → 护栏触发 {len(guards)} 次：")
        for g in guards:
            print(f"       · {g.title}：{g.detail}")
        print(f"  → degraded_reasons：{result.degraded_reasons or '（空）'}")

        # 这是本场景最该被记住的一句话
        same = rep["risk_level"] == result.risk_level
        print(f"\n  → 报告等级 == 融合等级 ：{'[OK]' if same else '[FAIL]'}")

    print(f"\n{BAR}")
    print("  结论：大模型可以决定'话怎么说'，决定不了'风险几级'。")
    print("        它写错等级时，系统不会采信，也不会静默——会在 trace 里留一条护栏记录。")
    print(BAR)

def main() -> int:
    parser = argparse.ArgumentParser(description="建议角色提示词演示")
    parser.add_argument("--only", choices=["system", "message", "report"],
                        help="只跑其中一个场景")
    args = parser.parse_args()

    print(BAR)
    print("ProjectFisheries —— Agent 建议角色提示词演示")
    print(BAR)
    print("  说明：本机未配置 API key，场景三用**手写 JSON** 冒充模型输出，")
    print("        走的是与真实模型完全相同的解析 / 校验 / 护栏路径。")

    if args.only in (None, "system"):
        print()
        case_system()
    if args.only in (None, "message"):
        print()
        case_message()
    if args.only in (None, "report"):
        print()
        case_report()

    print(f"\n{BAR}")
    print("  提示词源码：src/fisheries/agent/prompts.py")
    print("  解析与护栏：src/fisheries/agent/orchestrator.py（_report_from_parsed）")
    print(BAR)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
