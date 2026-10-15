"""Web 演示页（Gradio）演示脚本 —— 不开浏览器、不起服务器，直接跑一遍。

用途：
    1. 证明 ``src/fisheries/webui/`` 这一层能独立跑通，不需要真的把页面开起来；
    2. 把页面上那几栏 Markdown **原样打印出来**，给 README / 文档当真实样例；
    3. 给答辩现场留一条退路：万一投影仪/浏览器出问题，终端里也能看懂全流程。

运行：
    cd D:\\设置\\项目\\Agent\\project
    .venv\\Scripts\\python.exe scripts\\demo_webui.py
    .venv\\Scripts\\python.exe scripts\\demo_webui.py --app     # 额外构建一次 Gradio 界面

为什么要有 ``--app``：
    构建界面是唯一会 import gradio 的动作。默认不跑，是因为 gradio 是可选依赖
    （``pyproject.toml`` 的 ``[project.optional-dependencies] webui``）——没装
    gradio 的机器上，其余演示脚本必须照常可跑。加 ``--app`` 才是「我要验界面」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许脚本直接运行时找到 src/ 下的包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fisheries import datasets as _datasets  # noqa: E402
# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.webui.runner import (  # noqa: E402
    WebRequest,
    WebSession,
    example_table_path,
    session_info,
)
from fisheries.webui.table import ingest_table  # noqa: E402

#: 演示用真实病害图片。数据集不入库，所以找不到时脚本要能优雅降级。
#: 用 default_path（找不到也不抛），由脚本自己 .is_file() 判断。
DEMO_IMAGE = (
    _datasets.default_path("fish_disease")
    / "Test" / "Bacterial gill disease"
    / "Bacterial gill disease (100).jpg"
)


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def show(title: str, text: str) -> None:
    hr(title)
    body = (text or "").rstrip()
    print(body if body else "（空）")


def main() -> int:
    parser = argparse.ArgumentParser(description="ProjectFisheries Web 演示页离线演示")
    parser.add_argument(
        "--app", action="store_true",
        help="额外构建一次 Gradio 界面（需要已安装 gradio）",
    )
    parser.add_argument(
        "--image", default=str(DEMO_IMAGE),
        help="鱼体图片路径，默认用数据集里的真实烂鳃图",
    )
    parser.add_argument(
        "--no-knowledge", action="store_true",
        help="关闭 RAG 知识库检索，看纯规则/纯模型链路长什么样",
    )
    args = parser.parse_args()

    use_knowledge = not args.no_knowledge

    # ------------------------------------------------------------------ #
    # 1. 数据接入：和页面上「上传水质表格」是完全同一个函数
    # ------------------------------------------------------------------ #
    table_path = example_table_path()
    if not table_path.is_file():
        print(f"示例数据不存在：{table_path}", file=sys.stderr)
        return 2

    hr("第 1 步｜读取水质表格")
    print(f"文件：{table_path}")
    ingest = ingest_table(table_path)
    print(f"可用：{'是' if ingest.ok else '否'}")
    print(f"读入 {ingest.n_rows_read} 行，采用 {len(ingest.readings)} 行")
    print(f"列映射：{ingest.columns}")
    print(f"采样间隔：{ingest.interval_minutes} 分钟（来源 {ingest.interval_source}）")
    if ingest.warnings:
        print("提醒：")
        for w in ingest.warnings:
            print(f"  - {w}")

    image_path = Path(args.image)
    has_image = image_path.is_file()
    if not has_image:
        print(f"\n⚠ 图片找不到，本次只跑水质单模态：{image_path}")

    # ------------------------------------------------------------------ #
    # 2. 跑 Agent：页面点「运行 Agent」调的就是 session.run
    # ------------------------------------------------------------------ #
    session = WebSession(use_knowledge=use_knowledge)
    print()
    print(session_info(session))

    req = WebRequest(
        readings=ingest.readings,
        image_path=str(image_path) if has_image else None,
        pond_id="demo-pond-1",
        species="鲤科",
        question="今天要不要紧？",
        use_knowledge=use_knowledge,
    )

    hr("第 2 步｜运行 Agent")
    print(f"水质 {len(req.readings)} 点｜图片 {'有' if has_image else '无'}"
          f"｜知识库 {'开' if use_knowledge else '关'}")
    result = session.run(req)
    result.ingest = ingest           # 页面在这里挂上接入质量面板
    print(f"耗时 {result.elapsed_seconds:.2f}s｜"
          f"等级 {result.risk_level}｜融合分 {result.fused_score}")

    if not result.ok:
        hr("运行失败")
        print(result.error)
        return 1

    # ------------------------------------------------------------------ #
    # 3. 渲染：页面上每一栏的内容，原样打印
    # ------------------------------------------------------------------ #
    from fisheries.webui import render

    show("页面顶部｜大结论", render.render_summary_markdown(result))
    show("页面第 1 栏｜告警", render.render_alerts(result))
    show("页面第 2 栏｜处置建议（水质 / 病害分列）", render.render_report(result))
    show("页面第 3 栏｜水质数据接入质量", render.render_data_quality(result))
    show("页面第 4 栏｜鱼体图片识别结果", render.render_vision(result))
    show("页面第 5 栏｜融合依据（含等级下限）", render.render_fusion(result))
    if use_knowledge:
        show("页面第 6 栏｜知识库引用", render.render_knowledge(result))
    show("页面第 7 栏｜推理链路 + 页脚状态",
         render.render_trace(result) + "\n\n---\n\n" + render.render_status(result))

    # ------------------------------------------------------------------ #
    # 4. 可选：真的构建一次界面，证明组件树能装配起来
    # ------------------------------------------------------------------ #
    if args.app:
        hr("第 4 步｜构建 Gradio 界面")
        try:
            from fisheries.webui.app import build_app
        except ImportError as exc:
            print(f"⚠ 未安装 gradio，跳过界面构建：{exc}")
            print("  安装：pip install -e .[webui]")
            return 0
        demo = build_app(session)
        n_blocks = len(demo.blocks)
        print(f"✅ 界面构建成功：{n_blocks} 个组件")
        print("   真实启动：python -m fisheries.webui.app"
              "（默认 http://127.0.0.1:7860）")

    hr("完成")
    print("离线链路全部跑通。上面的 Markdown 就是页面上各栏的实际内容。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
