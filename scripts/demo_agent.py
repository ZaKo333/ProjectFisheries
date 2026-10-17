"""ProjectFisheries —— Agent 主调度完整可运行 demo。

    python scripts/demo_agent.py

无需任何 API key、无需联网、无需额外依赖：默认走离线确定性 MockLLM，
链路（LLM 决策 → 工具调用 → 融合 → 建议）与接入真实模型时完全一致。

想接真实模型，只需设置环境变量，代码一行不用改：

    # 调度角色用 DeepSeek（中文 function calling 可靠）
    $env:FISHERIES_LLM_SCHEDULER_PROVIDER = "deepseek"
    $env:FISHERIES_LLM_SCHEDULER_API_KEY  = "sk-..."

    # 建议角色用 AQUA-7B（本地 Ollama，水产领域知识强）
    $env:FISHERIES_LLM_ADVISOR_PROVIDER   = "aqua-7b-ollama"
    $env:FISHERIES_LLM_ADVISOR_MODEL      = "aqua-7b"
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.agent import AgentInput, AquacultureAgent  # noqa: E402
from fisheries.tools import fish_disease as _fish_disease  # noqa: E402

# 数据集不入库：路径统一由 fisheries.datasets 解析（环境变量 / 工程
# data/datasets / 历史目录 数据集 / kagglehub 缓存）。找不到时它会
# 抛出带「怎么下载、怎么挂载」的说明，而不是一句干巴巴的找不到。
TEST_ROOT = _datasets.resolve("fish_disease", "Test")

BAR = "=" * 78

#: 演示用 mock 取值来源。见 main() 里的说明横幅。
DEMO_MOCK_SOURCE = "folder_hint"

def _vision_banner() -> str:
    """按本机**实际**后端生成说明横幅。

    别写成固定文案：本机有 `models/yolov8-fish.pt` 就走真实 YOLO，没有就落
    mock 兜底。两种情况下「标签是从哪来的」完全不同（模型输出 vs 文件夹名），
    任写一种都会误导读者。
    """
    weights = _fish_disease.resolve_weights()
    if weights and importlib.util.find_spec("ultralytics") is not None:
        return (
            "本机已有训练权重，`fish_disease_detect` 走【真实 YOLOv8 推理】：\n"
            f"  {weights}\n"
            "标签与置信度来自模型输出，**与图片所在文件夹名无关**"
            "（`mock_source` 在真实后端下被忽略）。\n"
            "下面四个场景的病害标签是模型真算出来的，可当作一次端到端链路演示；\n"
            "模型自身的精度另见 docs/模型训练说明.md 与 M8 评测报告第 8 节。"
        )
    return (
        "本机没有可用的训练权重（或 ultralytics 未安装），"
        "`fish_disease_detect` 走【mock 兜底】。\n"
        '本 demo 显式传入 mock_source="folder_hint"，让 mock 从图片所在文件夹名取标签\n'
        "（底层 7 分类英文名 → 映射到 烂鳃/出血/水霉/其他/健康）。\n"
        "\n"
        '为什么这样做：mock 默认用文件哈希随机播种（mock_source="hash"），\n'
        '随机结果无法稳定触发"水质恶化 + 细菌性病害"这类多源协同规则，\n'
        "Agent 的核心增量就演示不出来。\n"
        "\n"
        "该参数默认关闭，且真实 YOLO 推理成功时代码路径**根本不会读文件名**。\n"
        "它只用于模型训练完成前的链路联调，**不用于任何准确率评测**。"
    )

def first_image(folder: str) -> str:
    """取某类别测试目录下的第一张图。不写死文件名——各文件夹起始编号不一样。"""
    directory = TEST_ROOT / folder
    if not directory.is_dir():
        return str(directory / "缺图片.jpg")
    files = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    return str(files[0]) if files else str(directory / "缺图片.jpg")

# --------------------------------------------------------------------------- #
# 数据构造
# --------------------------------------------------------------------------- #

def good_water(hours: int = 24) -> list[dict]:
    """24 小时稳定良好的水质。"""
    return [
        {
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": 26.0 + (0.4 if i % 6 == 0 else 0.0),
            "ph": 7.6 + 0.05 * ((i % 4) - 2),
            "dissolved_oxygen": 6.4 - 0.15 * (i % 5),
            "ammonia_n": 0.12 + 0.01 * (i % 3),
            "nitrite_n": 0.04,
        }
        for i in range(hours)
    ]

def crashing_water(hours: int = 24) -> list[dict]:
    """24 小时持续恶化：溶氧 6.2→2.6，氨氮 0.15→0.62，亚硝酸盐 0.05→0.15。"""
    out = []
    for i in range(hours):
        f = i / (hours - 1)
        out.append({
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": round(28.0 + 1.5 * f, 2),
            "ph": round(8.05 + 0.20 * f, 2),
            "dissolved_oxygen": round(6.2 - 3.6 * f, 2),
            "ammonia_n": round(0.15 + 0.47 * f, 3),
            "nitrite_n": round(0.05 + 0.10 * f, 3),
        })
    return out

def cold_water_sapro(hours: int = 24) -> list[dict]:
    """低温 22℃ 场景：用于触发水霉协同规则（水霉在 15–25℃ 高发）。"""
    out = []
    for i in range(hours):
        f = i / (hours - 1)
        out.append({
            "timestamp": f"2026-11-20T{i:02d}:00:00",
            "water_temp": round(22.0 - 0.3 * f, 2),
            "ph": round(7.2 - 0.1 * f, 2),
            "dissolved_oxygen": round(5.6 - 0.4 * f, 2),
            "ammonia_n": round(0.18 + 0.10 * f, 3),
            "nitrite_n": round(0.06 + 0.04 * f, 3),
        })
    return out

# --------------------------------------------------------------------------- #
# 输出渲染
# --------------------------------------------------------------------------- #

def render(result, verbose_trace: bool = True) -> None:
    icon = {"高危": "[!!!]", "预警": "[ ! ]", "正常": "[ . ]"}.get(result.risk_level, "[ ? ]")
    print(f"\n{icon} 风险等级：{result.risk_level}    融合风险分：{result.fused_score}")
    print(f"    定级理由：{result.reason}")

    if result.water:
        w = result.water
        print("\n  -- 水质工具 (water_risk_predict) --")
        print(f"     status={w.get('status')}  等级={w.get('risk_level')}  分数={w.get('risk_score')}")
        if w.get("trends"):
            for t in w["trends"].values():
                print(f"       {t.get('label'):<8} 当前 {t.get('current')} {t.get('unit', ''):<5}"
                      f" 趋势 {t.get('direction')}")
        if w.get("projected_do_alert_eta_hours") is not None:
            print(f"       溶氧倒计时：{w['projected_do_alert_eta_hours']:.1f} 小时后跌破 "
                  f"{w['projected_do_alert_threshold']} mg/L（{w.get('projected_do_alert_label')}）")
        if w.get("redline_hits"):
            print(f"       命中红线：{'、'.join(w['redline_hits'])}")

    if result.vision:
        v = result.vision
        print("\n  -- 图像工具 (fish_disease_detect) --")
        print(f"     status={v.get('status')}  病害={v.get('disease_type')}"
              f"  置信度={v.get('confidence')}  严重度={v.get('severity_weight')}")
        print(f"     底层原始判定={v.get('raw_class')}  模型={v.get('model_version')}")

    if result.alerts:
        print(f"\n  -- 告警 {len(result.alerts)} 条 --")
        for a in result.alerts:
            print(f"     [{a['level']:<8}] {a['title']}：{a['detail'][:70]}")

    failed = (result.fusion or {}).get("failed_modalities") or {}
    if failed:
        print("\n  -- ⚠ 模态失败（结论已排除这些证据）--")
        for mod, code in failed.items():
            print(f"     {'水质数据' if mod == 'water' else '鱼体图片'}：{code}")

    if result.headline:
        print(f"\n  -- 结论 --\n     {result.headline}")

    # 面向养殖户的结构化报告：四个业务块分开打印，便于看清"哪条是水质、哪条是病害"。
    rep = result.report or {}
    if rep:
        print("\n  == 养殖户报告（AdvisorReport）==")
        print(f"     综合风险等级：{rep.get('risk_level')}")

        w_actions = rep.get("water_actions") or []
        print(f"\n     [水质调控建议] {len(w_actions)} 条")
        if not w_actions:
            print("        （无）")
        for a in w_actions:
            print(f"       {a['priority']}. {a['target']} → {a['action']}")
            if a.get("timing"):
                print(f"          时间：{a['timing']}")

        d_actions = rep.get("disease_actions") or []
        print(f"\n     [病害处置方案] {len(d_actions)} 条")
        if not d_actions:
            print("        （无 —— 本次未检出病害或未做图像识别）")
        for a in d_actions:
            print(f"       {a['priority']}. {a['target']} → {a['action']}")
            if a.get("timing"):
                print(f"          时间：{a['timing']}")

        alerts = rep.get("alerts") or []
        print(f"\n     [告警信息] {len(alerts)} 条")
        if not alerts:
            print("        （无）")
        for a in alerts:
            print(f"       [{a['level']}] {a['title']}")

        if rep.get("observation_window"):
            print(f"\n     复查窗口：{rep['observation_window']}")
        if rep.get("confidence_note"):
            print(f"     可靠性说明：{rep['confidence_note']}")

    if result.observation_window:
        print(f"\n  -- 复查窗口（展平字段）--\n     {result.observation_window}")

    if result.references:
        print(f"\n  -- 引用的规范依据（RAG 检索，{len(result.references)} 条）--")
        for ref in result.references:
            print(f"     · {ref}")
    ks = result.knowledge_summary or {}
    if ks:
        print(
            f"\n  -- 知识库 --\n     {ks.get('backend', '?')}，"
            f"{ks.get('n_chunks', 0)} 条，用途分布 {ks.get('purposes', {})}"
        )
        for w in ks.get("warnings", []):
            print(f"     ⚠ {w}")

    if verbose_trace:
        print(f"\n  -- 推理轨迹（{len(result.trace)} 步）--")
        for t in result.trace:
            print(f"     {t.step:>2}. {t.kind:<12} {t.title}")
            if t.detail:
                print(f"         {t.detail[:100]}")

    if result.degraded_mode:
        print("\n  -- 降级说明 --")
        for r in result.degraded_reasons:
            print(f"     ! {r}")

# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #

def scenario(title: str, inp: AgentInput) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")
    agent = AquacultureAgent()
    print(f"输入：{inp.summary()}")
    print(f"后端：{agent.describe_backends()['scheduler']}")
    result = agent.run(inp)
    render(result)

def main() -> int:
    print(BAR)
    print("ProjectFisheries —— 水产养殖智能监测与决策 Agent")
    print("链路：LLM 决策工具 → 工具执行 → 护栏校验 → 确定性融合 → RAG 定向检索 → 建议生成")
    print(BAR)

    probe = AquacultureAgent()
    print(f"调度角色后端：{probe.describe_backends()['scheduler']}")
    print(f"建议角色后端：{probe.describe_backends()['advisor']}")

    print(f"""
{BAR}
【演示说明】{_vision_banner()}
{BAR}""")

    gill = first_image("Bacterial gill disease")
    septicemia = first_image("Bacterial diseases - Aeromoniasis")
    sapro = first_image("Fungal diseases Saprolegniasis")
    healthy = first_image("Healthy Fish")

    # 场景一：仅水质数据（单模态，权重要归一化）
    scenario(
        "场景一  仅水质时序数据（24 小时持续恶化）→ 单模态权重归一化",
        AgentInput(pond_id="P01", water_readings=crashing_water(), image_path=None),
    )

    # 场景二：仅鱼病图片（单模态）
    scenario(
        "场景二  仅鱼体图片（细菌性烂鳃）→ 单模态权重归一化",
        AgentInput(pond_id="P02", water_readings=None, image_path=gill,
                   mock_source=DEMO_MOCK_SOURCE),
    )

    # 场景三：双模态融合（本项目的核心场景）
    scenario(
        "场景三  水质恶化 + 细菌性病害图片 → 双模态融合 + 协同加成（核心场景）",
        AgentInput(pond_id="P03", water_readings=crashing_water(),
                   image_path=septicemia, species="鲤科",
                   mock_source=DEMO_MOCK_SOURCE),
    )

    # 场景四：一切正常
    scenario(
        "场景四  水质良好 + 健康鱼 → 应判定为正常",
        AgentInput(pond_id="P04", water_readings=good_water(),
                   image_path=healthy, mock_source=DEMO_MOCK_SOURCE),
    )

    # 场景五：低温 + 水霉协同规则
    scenario(
        "场景五  低温 22℃ + 水霉菌图片 → 触发 SYN_SAPRO_ON_COLD 协同规则",
        AgentInput(pond_id="P05", water_readings=cold_water_sapro(),
                   image_path=sapro, mock_source=DEMO_MOCK_SOURCE),
    )

    # 场景六：错误输入（工具报错不应让 Agent 断链，且必须显式上报）
    scenario(
        "场景六  图片路径不存在 → 工具返回结构化错误，Agent 继续出结论但显式标注未分析",
        AgentInput(pond_id="P06", water_readings=good_water(),
                   image_path=r"D:\不存在的图片\none.jpg"),
    )

    print(f"\n{BAR}")
    print("demo 结束。以上全部结果均为确定性输出，可重复执行验证。")
    print(BAR)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
