"""ProjectFisheries —— RAG 知识库检索效果演示。

    python scripts/demo_rag.py

不需要 API key。**需要先构建向量库**（或用 --auto-build 现场构建）：

    python scripts/build_knowledge_base.py
    python scripts/demo_rag.py

演示四件事
============
    场景一  知识库概览            —— 有多少篇、多少块、按类别/病种分布
    场景二  直接检索              —— 同一个问题上，纯向量 vs 混合检索的差异
    场景三  按 Agent 状态定向检索 —— 高危 / 正常 / 低温水霉 三种状态各查什么
    场景四  端到端                —— Agent 全链路 + 检索到的依据注入 prompt

为什么值得看场景三
====================
"直接用用户问题去搜向量库"和"按 Agent 已确定的事实去搜"是两回事。
工具已经算出"溶氧是主控风险因子、命中了 DO_REDLINE"，
就没必要让向量检索去猜该找什么——直接按这个事实定向检索：
高危才查应急预案、病种必须硬过滤、投料规范在有病害时也要查。

命令参数
=========
    --auto-build    向量库不存在时现场构建（默认关闭，避免演示时静默跑 8 秒）
    --embedder      向量后端，fastembed（默认）/ hash（秒级，验证管道用）
    --show-prompt   额外打印真正注入大模型的知识块原文
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.rag import AgentKnowledgeState, open_knowledge_base, plan_queries  # noqa: E402

BAR = "=" * 78
THIN = "-" * 78

def head(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")

def sub(title: str) -> None:
    print(f"\n▸ {title}\n{THIN}")

def _fmt_chunk(idx: int, c, *, show_text: int = 0) -> str:
    """一条检索结果的可读行。"""
    flag = "已核对" if c.verified else "⚠ 待核对"
    if c.dense_score is not None and c.sparse_score is not None:
        how = "向量+词面"
    elif c.dense_score is not None:
        how = "仅向量"
    else:
        how = "仅词面"
    line = (
        f"  {idx}. [{c.category}] {c.title} > {c.heading_text}\n"
        f"     相关度 {c.score:.3f}（向量 {c.dense_score if c.dense_score is None else round(c.dense_score, 3)}"
        f" / BM25 {c.sparse_score if c.sparse_score is None else round(c.sparse_score, 2)}"
        f" · {how} · {flag}）"
    )
    if show_text:
        body = c.text.strip().replace("\n", " ")
        line += f"\n     {body[:show_text]}{'…' if len(body) > show_text else ''}"
    return line

# --------------------------------------------------------------------------- #
# 场景一：知识库概览
# --------------------------------------------------------------------------- #

def case_overview(kb) -> None:
    head("场景一 知识库概览")
    info = kb.describe()
    print(f"  检索后端 : {info['backend']}")
    print(f"  知识块数 : {info['n_chunks']}")
    print(f"  按类别   : {json.dumps(info['by_category'], ensure_ascii=False)}")
    print(f"  按病种   : {json.dumps(info['by_disease'], ensure_ascii=False)}")

    man = info.get("manifest") or {}
    if man:
        print(f"\n  构建清单（{man.get('built_at', '?')}）")
        print(f"    文档 {man.get('n_docs', '?')} 篇 → 块 {man.get('n_chunks', '?')} 个，"
              f"含表格 {man.get('n_tables', '?')} 块，共 {man.get('total_chars', '?')} 字")
        print(f"    向量模型 {man.get('embedder', '?')}，维度 {man.get('dim', '?')}")
        print(f"    语料指纹 {man.get('fingerprint', '?')}")

    print("\n  说明：按病种里的『通用』= 不针对某一病种的水质/投料/应急文档。")

# --------------------------------------------------------------------------- #
# 场景二：直接检索（对比向量与混合）
# --------------------------------------------------------------------------- #

DIRECT_QUERIES = [
    ("池塘溶氧太低了怎么办", "水质"),
    ("亚硝酸盐超标有什么危害", "水质"),
    ("水霉病怎么处理", "病害"),
    ("烂鳃病用什么药，要注意什么", "病害"),
    ("水温 20 度的时候投喂率是多少", "投料"),
    ("鱼浮头了人走近也不下沉", "应急"),
]

def case_direct(kb) -> None:
    head("场景二 直接检索（按类别过滤，k=3）")
    print("  这一节回答的是：『知识库里到底有没有我问的东西，排第几？』")
    print("  『向量+词面』表示两路都召回；纯『仅向量』通常意味着措辞对不上字面。")

    for query, category in DIRECT_QUERIES:
        sub(f"{query}    [类别={category}]")
        res = kb.search(query, k=3, category=category)
        if not res.chunks:
            print("  （无命中）")
            continue
        for i, c in enumerate(res.chunks, 1):
            print(_fmt_chunk(i, c, show_text=60))
        for w in res.warnings:
            print(f"  ⚠ {w}")

# --------------------------------------------------------------------------- #
# 场景三：按 Agent 状态定向检索
# --------------------------------------------------------------------------- #

def _state_crashing() -> AgentKnowledgeState:
    return AgentKnowledgeState(
        risk_level="高危",
        limiting_factor="dissolved_oxygen",
        limiting_factor_label="溶氧",
        redline_hits=["DO_REDLINE_PROJECTED"],
        severe_indicators=["dissolved_oxygen", "ammonia_n"],
        disease_type="烂鳃",
        is_healthy=False,
        do_eta_hours=6.9,
        water_temp=29.5,
    )

def _state_normal() -> AgentKnowledgeState:
    return AgentKnowledgeState(
        risk_level="正常",
        limiting_factor="",
        limiting_factor_label="",
        disease_type="健康",
        is_healthy=True,
        water_temp=26.0,
    )

def _state_cold_sapro() -> AgentKnowledgeState:
    return AgentKnowledgeState(
        risk_level="预警",
        limiting_factor="water_temp",
        limiting_factor_label="水温",
        severe_indicators=["water_temp"],
        disease_type="水霉",
        is_healthy=False,
        is_low_confidence=False,
        water_temp=22.0,
    )

def case_planned(kb) -> None:
    head("场景三 按 Agent 状态定向检索")
    print("  同一套语料，三种融合结论 → 三套不同的检索计划。")
    print("  注意『应急处置』只在 高危/命中红线 时出现；")
    print("  『病害处置』永远是硬过滤（查烂鳃绝不会返回水霉）。")

    for name, st in (
        ("正常 · 健康鱼 · 26℃", _state_normal()),
        ("高危 · 溶氧红线 · 检出烂鳃", _state_crashing()),
        ("预警 · 低温 22℃ · 检出水霉", _state_cold_sapro()),
    ):
        sub(name)
        plan = plan_queries(st)
        print("  检索计划（顺序 = 优先级）：")
        for q in plan:
            print(f"    · [{q.purpose}] k={q.k} category={q.category} "
                  f"disease={q.disease} allow_disease_null={q.allow_disease_null}")
            print(f"      query = {q.query}")

        ctx = kb.retrieve_for_state(st, max_chunks=8)
        print(f"\n  → 命中 {ctx.n_chunks} 条，按用途分布："
              f"{ {k: len(v) for k, v in ctx.by_purpose.items()} }")
        for purpose, items in ctx.by_purpose.items():
            for c in items[:2]:
                print(f"    [{purpose}] {c.title} > {c.heading_text}"
                      f"  （相关度 {c.score:.3f}）")
        if ctx.degraded:
            print("  ⚠ 检索降级：")
            for w in ctx.warnings:
                print(f"      {w}")

# --------------------------------------------------------------------------- #
# 场景四：端到端 —— Agent 全链路 + 依据注入
# --------------------------------------------------------------------------- #

def case_end_to_end(kb, show_prompt: bool) -> None:
    head("场景四 端到端：水质时序 + 鱼病图片 → 告警 + 有依据的养殖建议")

    from fisheries.agent import AgentInput, AquacultureAgent
    from fisheries.agent.llm import MockLLMClient

    test_root = _datasets.resolve("fish_disease", "Test")

    def first_image(folder: str) -> str | None:
        d = test_root / folder
        if not d.is_dir():
            return None
        files = sorted(p for p in d.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        return str(files[0]) if files else None

    water = [
        {
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": round(28.0 + 1.5 * i / 23, 2),
            "ph": round(8.05 + 0.20 * i / 23, 2),
            "dissolved_oxygen": round(6.2 - 3.6 * i / 23, 2),
            "ammonia_n": round(0.15 + 0.47 * i / 23, 3),
            "nitrite_n": round(0.05 + 0.10 * i / 23, 3),
        }
        for i in range(24)
    ]
    image = first_image("Bacterial gill disease")

    print("  输入：")
    print("    水质 24 条：溶氧 6.20→2.60 mg/L，氨氮 0.15→0.62 mg/L，"
          "亚硝酸盐 0.05→0.15 mg/L，水温 28→29.5℃")
    print(f"    图片：{image or '（数据集未找到，将只跑单模态）'}")
    print("\n  说明：图像模型尚未训练，mock 从文件夹名取标签以稳定触发多源协同规则；")
    print("        该参数默认关闭，真实 YOLO 推理成功时代码路径根本不读文件名。")

    agent = AquacultureAgent(
        scheduler_client=MockLLMClient(),
        advisor_client=MockLLMClient(),
        knowledge_base=kb,
    )
    result = agent.run(AgentInput(
        pond_id="DEMO-RAG",
        water_readings=water,
        image_path=image,
        mock_source="folder_hint",
    ))

    print(f"\n  ── 结论 ──")
    print(f"  风险等级 : {result.risk_level}（融合分 {result.fused_score}）")
    print(f"  理由     : {result.reason}")
    print(f"  告警     : {[a['code'] for a in result.alerts]}")

    print(f"\n  ── 养殖建议（{len(result.advisories)} 条）──")
    for a in result.advisories:
        print(f"  [{a.priority}] {a.target}｜{a.action}")
        if a.timing:
            print(f"      时机：{a.timing}")

    print(f"\n  ── 引用的规范依据（{len(result.references)} 条）──")
    for ref in result.references:
        print(f"    · {ref}")

    ks = result.knowledge_summary or {}
    if ks:
        print(f"\n  知识库检索摘要：后端 {ks.get('backend')}，"
              f"{ks.get('n_chunks')} 条，用途分布 {ks.get('purposes')}")

    if show_prompt:
        sub("注入建议角色 prompt 的知识块原文")
        print(result.knowledge_block)

    print(f"\n  耗时 {result.elapsed_seconds}s；"
          f"降级模式 {'是' if result.degraded_mode else '否'}；"
          f"调度模型 {result.llm_backends.get('scheduler')}")

# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ProjectFisheries RAG 检索演示")
    ap.add_argument("--auto-build", action="store_true",
                    help="向量库不存在时现场构建（默认关闭）")
    ap.add_argument("--embedder", default=None, choices=["fastembed", "hash"],
                    help="向量后端（默认 fastembed 真实模型）")
    ap.add_argument("--show-prompt", action="store_true",
                    help="额外打印注入大模型的知识块原文")
    ap.add_argument("--only", default=None,
                    choices=["overview", "direct", "planned", "e2e"],
                    help="只跑某一个场景")
    args = ap.parse_args(argv)

    print(BAR)
    print("ProjectFisheries —— RAG 知识库检索演示")
    print(BAR)
    print("  语料：knowledge/corpus 下 9 篇种子语料（水质阈值 / 三病处置 / 病害处置手册 / 投料 / 调控 / 应急 / 标准）")
    print("  引用的标准编号与水质数值已于 2026-10-15 逐条对到 GB 11607-1989 官方原文；")
    print("  未核验的语料在 prompt 里仍会带『⚠ 未与标准原文核对』标记。")

    try:
        kb = open_knowledge_base(auto_build=args.auto_build, embedder_backend=args.embedder)
    except Exception as exc:
        print(f"\n  ❌ 无法加载向量库：{type(exc).__name__}: {exc}")
        print("\n  请先构建：python scripts/build_knowledge_base.py")
        return 2

    runners = {
        "overview": lambda: case_overview(kb),
        "direct": lambda: case_direct(kb),
        "planned": lambda: case_planned(kb),
        "e2e": lambda: case_end_to_end(kb, args.show_prompt),
    }
    if args.only:
        runners[args.only]()
    else:
        case_overview(kb)
        case_direct(kb)
        case_planned(kb)
        case_end_to_end(kb, args.show_prompt)

    print(f"\n{BAR}\n演示结束\n{BAR}")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
