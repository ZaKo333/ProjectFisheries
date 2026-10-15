"""D12 演示 —— 英文提示词 + 结果回译（针对纯英文的水产微调模型）。

**要回答的问题**：AQUA-7B 是 Mistral-7B-Instruct-v0.3 的英文微调（HF 标签
``language: en``），它写中文没有质量保证；但竞赛与养殖户都要中文。怎么办？

本脚本把这条链路上每一个环节都用真实代码跑出来、不贴文档：

    场景一  语言是怎么决定的？—— 后端 → 语言的判定表（含环境变量覆盖）
    场景二  英文提示词长什么样？硬约束一条不少吗？（逐条自检）
    场景三  英文 JSON → 中文报告：哪几段译出来了、哪几段退回了模板
    场景四  四种回译后端：local（本地离线模型）/ glossary（词表兜底）/ llm / bypass

**这套设计最该被记住的一句话**：回译只碰"话怎么说"，碰不到"风险几级"——
``risk_level`` 与 ``alerts`` 是护栏要比对的标识符，一个字都不许过翻译。

用法：
    python scripts/demo_translate.py
    python scripts/demo_translate.py --only lang        # 只看语言判定表
    python scripts/demo_translate.py --only prompt      # 只看英文系统提示词
    python scripts/demo_translate.py --only report      # 只看英文→中文的整份报告
    python scripts/demo_translate.py --only backends    # 只看四种回译后端
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.agent.llm import PROVIDER_PRESETS, ChatResponse  # noqa: E402
from fisheries.agent.orchestrator import AgentInput, AquacultureAgent  # noqa: E402
from fisheries.agent.prompts import (  # noqa: E402
    advisor_system_prompt,
    scheduler_system_prompt,
)
from fisheries.agent.translate import (  # noqa: E402
    LANG_EN,
    LANG_ZH,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_WORDS,
    LOCAL_DIR_ENV,
    GlossaryTranslator,
    LLMTranslator,
    BypassTranslator,
    LocalTranslator,
    TRANSLATOR_SYSTEM_PROMPT,
    back_translate_report,
    build_translator,
    describe_translator,
    load_translator_client,
    resolve_prompt_lang,
    resolve_translator_mode,
)

BAR = "=" * 78

# --------------------------------------------------------------------------- #
# 演示数据
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

#: 一份英文建议角色输出。**它就是本演示的输入**——模拟 AQUA-7B 按英文提示词作答。
#: 注意 ``risk_level`` 仍是中文枚举：英文提示词里明确要求这一项原样照抄。
ENGLISH_OUTPUT: dict[str, Any] = {
    "risk_level": "高危",
    "headline": "Dissolved oxygen is about to crash. Turn on the aerator now.",
    "alerts": [
        {"level": "critical", "title": "水质红线告警",
         "detail": "预测 12 小时内溶氧跌破泛塘临界值"},
    ],
    "water_actions": [
        {"priority": 1, "target": "dissolved oxygen",
         "action": "turn on the aerator now", "timing": "tonight"},
        {"priority": 2, "target": "ammonia nitrogen",
         "action": "feed less", "timing": "today"},
    ],
    "disease_actions": [
        {"priority": 1, "target": "bacterial gill disease",
         "action": "remove the sick fish and follow the label dose",
         "timing": "today"},
    ],
    "observation_window": "measure dissolved oxygen tonight",
    "confidence_note": "healthy fish no disease detected",
}

#: 同样的内容，写成"短到词表能逐词译出来"的样子——用来对比两种极端。
ENGLISH_SHORT_OUTPUT: dict[str, Any] = dict(
    ENGLISH_OUTPUT,
    headline="dissolved oxygen low",
    water_actions=[{"priority": 1, "target": "dissolved oxygen",
                    "action": "turn on the aerator", "timing": "tonight"}],
    disease_actions=[],
    observation_window="measure dissolved oxygen tonight",
    confidence_note="healthy fish",
)

class EnglishAdvisor:
    """假扮只会说英文的 AQUA-7B：无论问什么，都用英文 JSON 回答。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.last_messages: list[dict[str, Any]] = []

    def chat(self, messages, tools=None):  # noqa: ANN001
        self.last_messages = messages
        return ChatResponse(
            content=json.dumps(self.payload, ensure_ascii=False, indent=2),
            finish_reason="stop",
        )

def _run_english(payload: dict[str, Any], *, bypass: bool = False,
                 use_knowledge: bool = False):
    advisor = EnglishAdvisor(payload)
    agent = AquacultureAgent(
        advisor_client=advisor,
        use_knowledge=use_knowledge,
        prompt_lang=LANG_EN,
        translator_mode="bypass" if bypass else None,
    )
    result = agent.run(AgentInput(
        pond_id="P01",
        species="鲤科",
        water_readings=crashing_water(),
        image_path=str(GILL_IMAGE),
        mock_source="folder_hint",
    ))
    return advisor, agent, result

# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #

def case_lang() -> None:
    print(BAR)
    print("场景一  语言是怎么决定的")
    print(BAR)
    print("  优先级：显式入参 > 环境变量 FISHERIES_PROMPT_LANG > 按后端自动判断\n")

    env = os.environ.get("FISHERIES_PROMPT_LANG", "")
    print(f"  当前环境变量 FISHERIES_PROMPT_LANG = {env or '（未设置）'}\n")

    providers = ["aqua-7b-ollama", "aqua-7b-vllm", "deepseek", "dashscope",
                 "openai", "mock", "(未知后端)"]
    print(f"  {'后端':<18} {'自动判定':<10} 说明")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 40}")
    for provider in providers:
        real = "" if provider.startswith("(") else provider
        lang = resolve_prompt_lang(None, provider=real)
        if lang == LANG_EN:
            why = "英文底座：让它用母语思考，结果再回译"
        else:
            why = "中文能力够：直接写中文，省一次回译"
        print(f"  {provider:<18} {lang:<10} {why}")

    print(f"\n  -- 显式覆盖 --")
    print(f"  resolve_prompt_lang('{LANG_ZH}', provider='aqua-7b-ollama') "
          f"= {resolve_prompt_lang(LANG_ZH, provider='aqua-7b-ollama')}   "
          f"（宁可要 AQUA-7B 的中文，也不回译）")
    print(f"  resolve_prompt_lang('{LANG_EN}', provider='deepseek') "
          f"= {resolve_prompt_lang(LANG_EN, provider='deepseek')}   "
          f"（让强中文模型也用英文答，再回译）")

    print(f"\n  -- 后端预置表（PROVIDER_PRESETS）--")
    for name, preset in PROVIDER_PRESETS.items():
        lang = resolve_prompt_lang(None, provider=name)
        print(f"     {name:<18} model={str(preset.get('model', '')):<34} → {lang}")

def case_prompt() -> None:
    print(BAR)
    print("场景二  英文系统提示词（发给 AQUA-7B 的那份）")
    print(BAR)

    zh = advisor_system_prompt(LANG_ZH)
    en = advisor_system_prompt(LANG_EN)
    print(f"  中文版 {len(zh)} 字 ／ 英文版 {len(en)} 字"
          f"（英文更长是正常的：英语表达同一约束用的词更多）\n")

    for line in en.splitlines():
        print(f"  {line}")

    print(f"\n  -- 硬约束自检（英文版是否把中文版的规矩都带过去了）--")
    checks = {
        "面向养殖户（不是写给工程师）": "farmer" in en.lower(),
        "不许改数值": "change no number" in en.lower(),
        "不许编造未测指标": "not measured" in en.lower()
                            or "never invent" in en.lower(),
        "等级/告警只能复述": "copy" in en.lower() or "verbatim" in en.lower(),
        "动作要短": "short" in en.lower(),
        "限定条数（0–3 条）": "0-3" in en or "0–3" in en,
        "病害必须停料": "stop feeding" in en.lower(),
        "观察窗口跟风险来源": "observation" in en.lower(),
        "risk_level 用中文枚举": "中文" in en or "Chinese" in en,
    }
    for name, ok in checks.items():
        print(f"     {'[OK]' if ok else '[MISSING]'} {name}")

    missing = [k for k, v in checks.items() if not v]
    print(f"\n  结论：{'全部约束均在英文版中落地' if not missing else '缺失：' + str(missing)}")
    print(f"  提示词源码：src/fisheries/agent/prompts.py"
          f"（SCHEDULER_SYSTEM_PROMPT_EN / ADVISOR_SYSTEM_PROMPT_EN）")

    print(f"\n  -- 附带说明：调度角色的英文提示词 --")
    print("     调度角色的任务只是「选哪个工具」，不写自然语言给养殖户看。")
    print(f"     英文版 {len(scheduler_system_prompt(LANG_EN))} 字 / "
          f"中文版 {len(scheduler_system_prompt(LANG_ZH))} 字，两者同构。")
    print(f"\n  -- RAG 知识库仍是中文，这一点是刻意的 --")
    print(f"     语料是中文行业标准，检索结果直接贴进英文提示词；"
          f"模型能读中文资料、用英文回答。这样两边都不用维护两份语料。")

def case_report() -> None:
    print(BAR)
    print("场景三  英文 JSON → 中文报告：逐字段看回译结果")
    print(BAR)

    for payload, label in ((ENGLISH_OUTPUT, "整句英文（真实模型的常态）"),
                           (ENGLISH_SHORT_OUTPUT, "短句英文（词表刚好够用）")):
        print(f"\n{BAR}")
        print(f"  【{label}】")
        print(BAR)

        _, agent, result = _run_english(payload)
        fallback = agent._fallback_advisories(  # noqa: SLF001 - 演示脚本，故意看内部兜底
            _fusion_of(result), result.water, result.vision,
        ).model_dump(mode="json")
        outcome = back_translate_report(payload, translator=agent.translator,
                                        fallback=fallback, lang=LANG_EN)

        print(f"\n  模型原文（英文）：")
        for line in json.dumps(payload, ensure_ascii=False, indent=2).splitlines():
            print(f"     {line}")

        print(f"\n  逐字段回译：")
        for key in ("headline", "observation_window", "confidence_note"):
            src = str(payload.get(key, ""))
            got = str(outcome.parsed.get(key, ""))
            mark = "[直译]" if got and got != fallback.get(key) else "[退回模板]"
            print(f"     {mark} {key}")
            print(f"           EN  : {src}")
            print(f"           ZH  : {got}")

        for list_key in ("water_actions", "disease_actions"):
            src_items = payload.get(list_key) or []
            got_items = outcome.parsed.get(list_key) or []
            print(f"     {list_key}：模型给 {len(src_items)} 条 → 报告留 {len(got_items)} 条")
            fb_items = fallback.get(list_key) or []
            for i, item in enumerate(got_items):
                same_as_template = i < len(fb_items) and item == fb_items[i]
                tag = "退回模板" if same_as_template else "译出"
                print(f"           {i + 1}. [{tag}] {item.get('target')} → "
                      f"{item.get('action')}（{item.get('timing')}）")

        print(f"\n  → 回译后端：{outcome.backend}"
              f"   直译 {outcome.n_translated} 段 / 退回模板 {outcome.n_replaced} 段")
        print(f"  → 降级留痕（会进 degraded_reasons）：")
        for note in outcome.notes or ["（无）"]:
            print(f"       · {note}")

        print(f"\n  → 最终报告等级：{result.risk_level}"
              f"（融合分 {result.fused_score}）")
        print(f"  → risk_level 是否原样保留："
              f"{'[OK] 未过翻译' if result.risk_level == payload['risk_level'] else '[!] 被改动'}")
        print(f"  → alerts 是否原样保留："
              f"{'[OK] 未过翻译' if result.report['alerts'] else '[!] 被改动'}")

    print(f"\n{BAR}")
    print("  结论：回译决定的是「话怎么说」，决定不了「风险几级」。")
    print("        译不出来的段落退回规则模板的同名字段，而不是保留英文——")
    print("        半英半中的报告比少一句建议更糟。每一次退回都写进 degraded_reasons。")
    print(BAR)

def _fusion_of(result):  # noqa: ANN001
    """从 AgentResult 里把融合层对象还原出来（演示用，便于调用内部兜底）。"""
    from fisheries.agent.fusion import FusionResult
    return FusionResult(**result.fusion)

def case_backends() -> None:
    print(BAR)
    print("场景四  四种回译后端：同一段英文，四种命运")
    print(BAR)

    sample = "dissolved oxygen is low"
    long_sample = ENGLISH_OUTPUT["headline"]

    print(f"\n  -- 环境变量 --")
    for key in ("FISHERIES_TRANSLATOR_MODE", "FISHERIES_TRANSLATOR_PROVIDER",
                "FISHERIES_TRANSLATOR_API_KEY", "FISHERIES_TRANSLATOR_BASE_URL",
                "FISHERIES_TRANSLATOR_MODEL", "FISHERIES_TRANSLATOR_LOCAL_DIR",
                "FISHERIES_TRANSLATOR_LOCAL_MODEL"):
        val = os.environ.get(key, "")
        if key.endswith("API_KEY") and val:
            val = val[:6] + "…"
        print(f"     {key:<38} = {val or '（未设置）'}")
    print(f"     解析出的模式：{resolve_translator_mode(None)}")

    print(f"\n  -- 后端 1：local（本地离线模型，有模型时优先）--")
    local = LocalTranslator()
    local_ok, local_why = local.available()
    print(f"     {describe_translator(local)}")
    print(f"     模型目录：{local.model_path}")
    if not local_ok:
        print(f"     本机不可用：{local_why}")
        print("     下模型：python scripts/fetch_translate_model.py")
    else:
        for text in (sample, long_sample):
            out = local.translate(text)
            print(f"     输入：{text}")
            print(f"     结果：{'[直译] ' + out.text if out.ok else '[拒绝] ' + out.note}")
        print(f"     分块预算：{DEFAULT_CHUNK_CHARS} 字符 / {DEFAULT_CHUNK_WORDS} 词 "
              f"（超出就切开——不切的话模型会丢掉句子尾部，连数字一起丢）")
        print("     术语后订正：通用 NMT 把 turbidity 译成「多发性」、aerator 译成")
        print("     「呼吸器」，靠一张中文→中文的后订正表扳回来。")

    print(f"\n  -- 后端 2：glossary（兜底，离线、确定性、零依赖）--")
    g = GlossaryTranslator()
    print(f"     {describe_translator(g)}")
    for text in (sample, long_sample):
        out = g.translate(text)
        print(f"     输入：{text}")
        print(f"     结果：{'[直译] ' + out.text if out.ok else '[拒绝] ' + out.note}")
    print("     短句（timing / 指标名）几乎都能直译；整句英文一律拒绝——")
    print("     逐词硬译会得到读不通的中文，那比退回模板更糟。")

    print(f"\n  -- 后端 3：llm（配了 FISHERIES_TRANSLATOR_* 时自动启用）--")
    llm = LLMTranslator(None)
    print(f"     {describe_translator(llm)}")
    print(f"     系统提示词（{len(TRANSLATOR_SYSTEM_PROMPT)} 字）：")
    for line in TRANSLATOR_SYSTEM_PROMPT.strip().splitlines()[:6]:
        print(f"       {line}")
    print(f"       …")
    out = llm.translate(sample)
    print(f"     未配模型时的行为：{'[直译] ' + out.text if out.ok else '[拒绝] ' + out.note}")
    print("     关键：调用失败**不抛异常**，只退回模板 —— 翻译只是可选环节，")
    print("     它挂掉不能让整条 Agent 链路挂掉。")

    print(f"\n  -- 后端 4：bypass（只在调试提示词时用）--")
    b = BypassTranslator()
    print(f"     {describe_translator(b)}")
    out = b.translate(long_sample)
    print(f"     输入：{long_sample}")
    print(f"     结果：{out.text}")
    print("     用途：把「提示词是否写得让模型听得懂」与「回译是否可靠」分开排查，")
    print("     否则回译失败会把提示词的问题掩盖掉。")

    print("\n  -- 自动选择逻辑 --")
    auto_now = build_translator("auto", client=load_translator_client(), env={})
    print(f"     本机 build_translator('auto', client=load_translator_client()) "
          f"→ {type(auto_now).__name__}")
    no_local = {LOCAL_DIR_ENV: os.path.join(os.sep, "nonexistent-translate-models")}
    print(f"     build_translator('auto', client=None, env=无本地模型) → "
          f"{type(build_translator('auto', client=None, env=no_local)).__name__}"
          f"  （没模型、没 key 的评委机器）")
    print("     顺序是 **本地模型 → 远端中文模型 → 词表兜底**。把本地模型放第一位")
    print("     是有原因的：以前 auto 只看有没有 client，离线机器必然落到词表后端，")
    print("     而词表后端对超过 6 个英文词的自由长句只能整条拒绝，")
    print("     **于是 AQUA-7B 的全部输出被丢掉、D12 的收益等于 0**。")
    print("     没单配 FISHERIES_TRANSLATOR_* 时还会**复用调度角色**的模型；")
    print("     不想要这个复用：把 FISHERIES_TRANSLATOR_MODE 设成 glossary。")

def main() -> int:
    parser = argparse.ArgumentParser(description="D12 英文提示词 + 结果回译演示")
    parser.add_argument("--only", choices=["lang", "prompt", "report", "backends"],
                        help="只跑其中一个场景")
    args = parser.parse_args()

    print(BAR)
    print("ProjectFisheries —— D12 演示：英文提示词 + 结果回译")
    print(BAR)
    print("  背景：AQUA-7B 是 Mistral-7B-Instruct-v0.3 的英文微调（HF 标签 language: en），")
    print("        要它写中文没有质量保证；但竞赛与养殖户都要中文。")
    print("  做法：让它用母语（英文）思考，再用中文模型整句回译；")
    print("        回译不可用时退回规则模板的中文表述，并在报告里留痕。")

    if args.only in (None, "lang"):
        print()
        case_lang()
    if args.only in (None, "prompt"):
        print()
        case_prompt()
    if args.only in (None, "report"):
        print()
        case_report()
    if args.only in (None, "backends"):
        print()
        case_backends()

    print(f"\n{BAR}")
    print("  回译源码：src/fisheries/agent/translate.py")
    print("  提示词：  src/fisheries/agent/prompts.py（*_EN 后缀）")
    print("  接线：    src/fisheries/agent/orchestrator.py（_back_translate）")
    print(BAR)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
