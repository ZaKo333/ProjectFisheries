#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端边界案例演示：把「传感器读数 + 图片识别结果」灌进 AquacultureAgent，
打印完整五步推理链路（工具调用 → 护栏 → 融合定级 → 知识检索 → 建议报告）。

用途
----
1. 复现评审/答辩现场给定的那组测试数据（溶氧 2.8、氨氮 0.35、烂鳃 0.87）；
2. 批量跑边界案例，检验融合层在"单模态强/弱""模态互相矛盾""病害轻但水质优"等
   情形下会不会给出不合常理的等级。

设计说明（重要，不要误读输出）
------------------------------
* **水质侧**：`water_risk_predict` 需要 ≥3 个时序点。评审给的是**单点快照**，
  本脚本据此构造 24 点稳态序列（末点=给定值）。因此趋势项必然为"平稳"，
  定级完全由"当前值落在哪条管控线"驱动——这正是我们要测的东西。
* **图像侧**：本脚本调用 `fish_disease_detect(..., force_mock=True)` 拿到
  结构化字段，再把 `disease_type / confidence / severity_weight / is_healthy`
  覆写为评审给定的值。也就是说：**字段形状是真的，标签与置信度是人工指定的**。
  这里刻意用 `force_mock=True` 而不是"等真实权重"：本脚本测的是**融合层与护栏**
  在给定输入下的行为，不是模型精度，所以输入必须锁死、可复现。
  （真实权重的精度另见 `docs/模型训练说明.md` 与 M8 报告第 8 节。）
* 两个侧面的覆写都在 `vision_stub()` 里显式标注，输出里也打了横幅，
  不允许被当成模型精度证据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# 允许直接 `python scripts\demo_cases.py` 跑，不必先设 PYTHONPATH。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.agent.orchestrator import AquacultureAgent, AgentInput  # noqa: E402
from fisheries.tools.fish_disease import fish_disease_detect            # noqa: E402
from fisheries.tools import TOOL_HANDLERS                               # noqa: E402
from fisheries import datasets as _datasets                             # noqa: E402

DEMO_IMAGE = str(_datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
))

BAR = "=" * 78
SUB = "-" * 78

# 病害严重度权重（与 tools/fish_disease.py 的 LABEL_META 保持一致）
SEVERITY = {"烂鳃": 0.85, "出血": 0.90, "水霉": 0.70, "其他": 0.75, "健康": 0.00}
GROUP = {"烂鳃": "细菌性", "出血": "细菌性", "水霉": "真菌性", "其他": "其他", "健康": "健康"}

# --------------------------------------------------------------------------- #
# 构造输入
# --------------------------------------------------------------------------- #

def steady_readings(
    *,
    do: float,
    ammonia: float,
    ph: float = 7.8,
    temp: float = 27.0,
    nitrite: float | None = None,
    n: int = 24,
) -> list[dict[str, Any]]:
    """构造 n 点稳态序列，末点即评审给定的快照值。

    `nitrite=None` 时整条序列都不带亚硝酸盐字段，用来顺带验证
    "探头缺失的指标不得被当作超标"这条老坑。
    """
    out: list[dict[str, Any]] = []
    for i in range(n):
        point: dict[str, Any] = {
            "timestamp": f"2025-06-01T{i:02d}:00:00",
            "water_temp": temp,
            "ph": ph,
            "dissolved_oxygen": do,
            "ammonia_n": ammonia,
        }
        if nitrite is not None:
            point["nitrite_n"] = nitrite
        out.append(point)
    return out

def vision_stub(label: str, confidence: float) -> Callable[..., dict[str, Any]]:
    """返回一个"字段形状取真实工具、标签与置信度由人指定"的图片检测桩。

    桩的实现方式：调用真实工具拿骨架 → 覆写语义字段。这样即使以后
    `FishDiseaseOutput` 增删字段，本脚本也不会悄悄失真。
    """
    def handler(**kwargs: Any) -> dict[str, Any]:
        base = fish_disease_detect(kwargs.get("image_path") or DEMO_IMAGE, force_mock=True)
        sev = SEVERITY[label]
        base.update(
            disease_type=label,
            disease_group=GROUP[label],
            confidence=round(confidence, 4),
            is_healthy=(label == "健康"),
            severity_weight=sev,
            is_low_confidence=confidence < 0.60,
            all_scores={label: round(confidence, 4),
                        "其他": round(max(0.0, 1 - confidence - 0.02), 4),
                        "健康": 0.02},
            note=(
                f"[人工指定桩] 标签={label} 置信度={confidence:.2f}；"
                f"用于融合逻辑验证，不代表模型精度。"
            ),
        )
        return base

    return handler

# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #

def hr(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")

def show_trace(result: Any) -> None:
    print("\n── 推理轨迹 (trace) ──")
    for t in result.trace:
        print(f"  [{t.step:>2}] {t.kind:<12} {t.title}")
        if t.detail:
            for line in str(t.detail).splitlines():
                print(f"        {line}")

def show_water(water: dict[str, Any] | None) -> None:
    print("\n── ① 水质时序风险 (water_risk_predict) ──")
    if not water or water.get("status") != "ok":
        print(f"  不可用：{water}")
        return
    print(f"  风险等级 : {water.get('risk_level')}   风险分 : {water.get('risk_score')}")
    print(f"  品种档案 : {water.get('species')}   评估窗口 : {water.get('horizon_hours')}h")
    print(f"  指标严重度: {json.dumps(water.get('severity_map'), ensure_ascii=False)}")
    print(f"  生效权重 : {json.dumps(water.get('effective_weights'), ensure_ascii=False)}")
    print(f"  超标指标 : {json.dumps(water.get('exceeded_indicators'), ensure_ascii=False)}")
    print(f"  命中规则 : {water.get('rule_hits')}")
    print(f"  红线命中 : {water.get('redline_hits') or '无'}")
    print(f"  主控因子 : {water.get('limiting_factor_label')}（{water.get('limiting_factor')}）")
    eta = water.get("projected_do_alert_eta_hours")
    if eta is not None:
        print(f"  溶氧跌破下一条线: {eta}h 后 → {water.get('projected_do_alert_threshold')} "
              f"mg/L（{water.get('projected_do_alert_label')}）")
    else:
        print("  溶氧倒计时: 未触及下一条管控线")
    dq = water.get("data_quality") or {}
    print(f"  数据质量 : {dq.get('n_points')} 点 / 覆盖 {dq.get('span_hours')}h / "
          f"缺失探头 {dq.get('missing_indicators') or '无'}")
    if dq.get("warnings"):
        print(f"  质量告警 : {dq.get('warnings')}")
    print(f"  恶化预测 : {water.get('degradation_summary')}")
    print(f"  模型说明 : degraded_mode={water.get('degraded_mode')} "
          f"({water.get('projection_method')})")

def show_vision(vision: dict[str, Any] | None) -> None:
    print("\n── ② 鱼体病害识别 (fish_disease_detect) ──")
    if not vision or vision.get("status") != "ok":
        print(f"  不可用：{vision}")
        return
    print(f"  病害类型 : {vision.get('disease_type')}（{vision.get('disease_group')}）")
    print(f"  置信度   : {vision.get('confidence')}   严重度权重 : {vision.get('severity_weight')}")
    print(f"  是否健康 : {vision.get('is_healthy')}   低置信标记 : {vision.get('is_low_confidence')}")
    print(f"  候选分布 : {json.dumps(vision.get('all_scores'), ensure_ascii=False)}")
    print(f"  提示     : {vision.get('note')}")

def show_fusion(fusion: dict[str, Any]) -> None:
    print("\n── ③ 多源融合 (fusion.fuse_assessment) —— 定级只在这里发生 ──")
    print(f"  风险等级 : {fusion.get('risk_level')}   融合分 : {fusion.get('fused_score')}")
    print(f"  加权公式 : {fusion.get('weighted_score')}   "
          f"等级下限抬升到 : {fusion.get('score_floor')}")
    if fusion.get("level_floors"):
        print(f"  触发下限 : {fusion.get('level_floors')}")
    print(f"  模态权重 : {json.dumps(fusion.get('effective_weights'), ensure_ascii=False)}")
    print(f"  水质分   : {fusion.get('water_score')}   图像分 : {fusion.get('vision_score')}")
    print(f"  协同加成 : +{fusion.get('synergy_score')}")
    for hit in fusion.get("synergy_hits") or []:
        print(f"      · {hit}")
    print(f"  可用模态 : {fusion.get('modalities')}   失败模态 : {fusion.get('failed_modalities')}")
    print(f"  降级模式 : {fusion.get('degraded_mode')}")
    print(f"  定级理由 : {fusion.get('reason')}")
    for a in fusion.get("alerts") or []:
        print(f"      ! [{a.get('level')}] {a.get('title')}：{a.get('detail')}")

def show_report(result: Any) -> None:
    print("\n── ④ 知识检索 (RAG) ──")
    if result.knowledge_block:
        ks = result.knowledge_summary or {}
        print(f"  检索摘要 : backend={ks.get('backend')} n_chunks={ks.get('n_chunks')} "
              f"degraded={ks.get('degraded')}")
        print(f"  用途分布 : {json.dumps(ks.get('purposes'), ensure_ascii=False)}")
        if ks.get("warnings"):
            print(f"  告警     : {ks.get('warnings')}")
        print(f"  引用条款 : {len(result.references or [])} 条")
        for r in (result.references or [])[:8]:
            print(f"      · {r}")
    else:
        print("  本轮未启用知识库")

    print("\n── ⑤ 建议报告 (advisor) ──")
    print(f"  一句话结论 : {result.headline or '（无）'}")
    if result.advisories:
        for adv in result.advisories:
            timing = f"  @{adv.timing}" if adv.timing else ""
            print(f"    P{adv.priority} 【{adv.target}】{adv.action}{timing}")
    else:
        print("    （无建议）")
    if result.observation_window:
        print(f"  观察窗口   : {result.observation_window}")
    if result.confidence_note:
        print(f"  置信度说明 : {result.confidence_note}")

def show_degraded(result: Any) -> None:
    if result.degraded_mode or result.degraded_reasons:
        print("\n── 降级/护栏记录 ──")
        print(f"  degraded_mode = {result.degraded_mode}")
        for r in result.degraded_reasons:
            print(f"      · {r}")

def run_case(
    no: int,
    name: str,
    *,
    readings: list[dict[str, Any]],
    label: str,
    confidence: float,
    expect: str,
    use_knowledge: bool = True,
) -> dict[str, Any]:
    hr(f"案例 {no}：{name}")
    print(f"  输入水质 : 溶氧={readings[-1]['dissolved_oxygen']} mg/L, "
          f"氨氮={readings[-1]['ammonia_n']} mg/L, "
          f"pH={readings[-1]['ph']}, 水温={readings[-1]['water_temp']}℃, "
          f"亚硝酸盐={readings[-1].get('nitrite_n', '探头缺失')}")
    print(f"  输入图像 : {label}  置信度={confidence:.2f}")
    print(f"  预期行为 : {expect}")

    handlers = dict(TOOL_HANDLERS)
    handlers["fish_disease_detect"] = vision_stub(label, confidence)

    agent = AquacultureAgent(tool_handlers=handlers, use_knowledge=use_knowledge)
    result = agent.run(AgentInput(
        pond_id=f"P{no:02d}",
        species="鲤科",
        water_readings=readings,
        image_path=DEMO_IMAGE,
        force_mock_llm=True,          # 无 API key 也要能跑：走确定性 Mock LLM
        use_knowledge=use_knowledge,
    ))

    show_trace(result)
    show_water(result.water)
    show_vision(result.vision)
    show_fusion(result.fusion)
    show_report(result)
    show_degraded(result)

    print(f"\n  >>> 最终判定：{result.risk_level}  融合分={result.fused_score}  "
          f"耗时={result.elapsed_seconds:.2f}s")
    fz = result.fusion
    return {
        "case": no,
        "name": name,
        "expected": expect,
        "risk_level": result.risk_level,
        "fused_score": result.fused_score,
        "weighted_score": fz.get("weighted_score"),
        "score_floor": fz.get("score_floor"),
        "level_floors": list(fz.get("level_floors") or []),
        "water_score": fz.get("water_score"),
        "vision_score": fz.get("vision_score"),
        "synergy": fz.get("synergy_score"),
        "weights": fz.get("effective_weights"),
        "degraded": result.degraded_mode,
        "degraded_reasons": list(result.degraded_reasons),
        "headline": result.headline,
    }

def main() -> int:
    print("AquacultureAgent 端到端边界案例演示")
    print("LLM 后端：MockLLMClient（确定性，无外部依赖）")
    print("图像侧：字段形状取真实工具，标签/置信度为人工指定 —— 不得用于精度评测")

    agent0 = AquacultureAgent(tool_handlers=dict(TOOL_HANDLERS))
    print(f"后端描述：{json.dumps(agent0.describe_backends(), ensure_ascii=False)}")

    rows: list[dict[str, Any]] = []

    # ---- 案例 1：评审现场给定的那组数据 ---------------------------------- #
    rows.append(run_case(
        1,
        "评审给定数据：溶氧 2.8 / 氨氮 0.35 + 烂鳃 0.87",
        readings=steady_readings(do=2.8, ammonia=0.35),
        label="烂鳃",
        confidence=0.87,
        expect="高危。溶氧 2.8 已落危险区（<3.0），叠加高致病烂鳃，"
               "且应触发低溶氧×烂鳃的协同加成。",
    ))

    # ---- 案例 2：水质正常 + 轻微水霉 -------------------------------------- #
    rows.append(run_case(
        2,
        "边界：水质全优 + 轻微水霉（0.62）",
        readings=steady_readings(do=6.5, ammonia=0.05, ph=7.6, temp=22.0, nitrite=0.03),
        label="水霉",
        confidence=0.62,
        expect="应为预警（不是正常、更不是高危）。水霉严重度权重 0.70，"
               "0.70×0.62=0.434→单靠 0.4 的图像权重只有 17.4 分，"
               "必须由「等级下限 DISEASE_DETECTED」抬到预警；"
               "且水温 22℃ 落在 15–25℃ 区间，应命中水霉×低温协同。",
    ))

    # ---- 案例 3：水质高危 + 鱼健康 ---------------------------------------- #
    rows.append(run_case(
        3,
        "边界：水质高危（溶氧 1.9 / 氨氮 0.72）+ 健康鱼（0.95）",
        readings=steady_readings(do=1.9, ammonia=0.72, ph=7.4, temp=29.0, nitrite=0.35),
        label="健康",
        confidence=0.95,
        expect="仍应为高危。图像分=0 不得把水质风险抵消掉：权重仍是水 0.6/图 0.4，"
               "加权公式只有 60.0（落在预警档），必须由「等级下限 WATER_LEVEL:高危」"
               "把等级抬回高危、并把 fused_score 抬到 70.0。"
               "这是本演示最关键的一组——修复前它被判成了预警。",
    ))

    # ---- 案例 4：水质预警 + 鱼健康 ---------------------------------------- #
    rows.append(run_case(
        4,
        "边界：水质预警（溶氧 4.2）+ 健康鱼（0.95）",
        readings=steady_readings(do=4.2, ammonia=0.12, ph=7.5, temp=26.0, nitrite=0.08),
        label="健康",
        confidence=0.95,
        expect="应为预警。溶氧 4.2 在 3–5 警戒带，水质工具自身判预警；"
               "图像健康不加分，加权公式只有 17.1 分（正常档），"
               "必须由「等级下限 WATER_LEVEL:预警」抬到预警。",
    ))

    # ---- 案例 5：水质全优 + 高置信烂鳃 ------------------------------------ #
    rows.append(run_case(
        5,
        "边界：水质全优 + 高置信烂鳃（0.95）—— 单模态归一化检验",
        readings=steady_readings(do=7.0, ammonia=0.03, ph=7.7, temp=25.0, nitrite=0.02),
        label="烂鳃",
        confidence=0.95,
        expect="高危。水质分=0，权重全给图像：0.85×0.95=0.8075→80.75 分 ≥70，"
               "这正是「鱼已经病了但水质还没坏」的真实早期场景，不能漏报。",
    ))

    # ---- 案例 6：水质正常 + 低置信烂鳃 ------------------------------------ #
    rows.append(run_case(
        6,
        "边界：水质全优 + 低置信烂鳃（0.42）",
        readings=steady_readings(do=7.0, ammonia=0.03, ph=7.7, temp=25.0, nitrite=0.02),
        label="烂鳃",
        confidence=0.42,
        expect="预警而非高危：0.85×0.42=0.357→单靠 0.4 的图像权重只有 14.3 分（正常档）。"
               "但烂鳃严重度 0.85 已过下限门槛，置信度 0.42 也在 0.40 之上，"
               "必须由「等级下限 DISEASE_DETECTED」判预警并给出「人工复核/隔离观察」，"
               "绝不放行为「照常管理」。",
    ))

    # ---- 汇总 ------------------------------------------------------------- #
    hr("汇总")
    print(f"{'#':<3}{'等级':<6}{'融合分':<8}{'加权公式':<10}{'水质分':<8}{'图像分':<8}"
          f"{'协同':<6}{'权重(水/图)':<14}场景")
    print(SUB)
    for r in rows:
        w = r["weights"] or {}
        wv = w.get("water", w.get("water_modality", 0.0))
        vv = w.get("vision", w.get("vision_modality", 0.0))
        floor = r["score_floor"]
        wmark = f"{r['weighted_score']:.2f}" + (f"→{floor:g}" if floor is not None else "")
        print(f"{r['case']:<3}{r['risk_level']:<6}{r['fused_score']:<8}{wmark:<10}"
              f"{r['water_score']:<8}{r['vision_score']:<8}{r['synergy']:<6}"
              f"{wv:.2f} / {vv:.2f}   {r['name'][:30]}")
    print(SUB)
    for r in rows:
        flag = "降级" if r["degraded"] else "正常"
        print(f"\n案例 {r['case']}：{r['risk_level']}（{r['fused_score']} 分，{flag}）")
        print(f"   预期：{r['expected']}")
        if r["level_floors"]:
            print(f"   等级下限：{'、'.join(r['level_floors'])}"
                  + (f"   加权公式分 {r['weighted_score']} → 抬升至 {r['score_floor']}"
                     if r["score_floor"] is not None else ""))
        print(f"   结论：{r['headline']}")
        if r["degraded"]:
            print(f"   降级原因：{r['degraded_reasons']}")
    print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
