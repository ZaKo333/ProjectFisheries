"""Gradio 现场演示页。

只做一件事：把「上传 → 点击 → 看结论」这条路径接到真实的 Agent 上。
**这个文件是 ``fisheries.webui`` 里唯一 import gradio 的模块**，
所以 ``table`` / ``runner`` / ``render`` 的单元测试在没装 gradio 的环境里也能跑。

页面结构刻意做成左右两栏：

* 左栏是**输入**（水质表、鱼图、参数、运行按钮）；
* 右栏是**结论**，从上到下按「结论 → 告警 → 建议 → 依据 → 链路」排，
  保证一眼看到的东西是最重要的，评审想深挖再往下翻折叠区。

把推理链路放在折叠区而不是藏起来，是因为「Agent 到底做了什么」正是这类
比赛的评分点；但把它摊开在最上面又会淹没结论——折叠是这两者的折中。
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from .render import (
    render_alerts,
    render_data_quality,
    render_error,
    render_fusion,
    render_knowledge,
    render_report,
    render_status,
    render_summary_markdown,
    render_trace,
    render_vision,
)
from .runner import (
    MOCK_SOURCE_CHOICES,
    SPECIES_CHOICES,
    WebRequest,
    WebSession,
    default_session,
    session_info,
)
from .table import EXAMPLE_TABLE, TableError, TableIngest, ingest_rows, ingest_table
from ..water_templates import all_templates, get_template, materialize

#: 表格最多读多少行。水质工具只做趋势外推，给几万行既慢又没有额外信息，
#: 而且真实数据集单个 CSV 就有 8 万行（IoTPond1），不设上限页面会明显卡顿。
MAX_TABLE_ROWS = 288

_PLACEHOLDER = "_还没有运行。上传水质表格或鱼体图片后点「运行 Agent」。_"

_CSS = """
.gradio-container {max-width: 1400px !important;}
#run-btn {min-height: 46px; font-size: 16px;}
.fisheries-hint {font-size: 12px; color: #6b7684;}
"""


# --------------------------------------------------------------------------- #
# 表格读取
# --------------------------------------------------------------------------- #

def _load_table(
    file_value: Any,
    pasted: str,
    max_rows: int,
) -> TableIngest | None:
    """从「上传的文件」或「粘贴的文本」里取一份表格。

    两条路径都给，是因为现场演示什么意外都有：文件选择器可能因为权限弹不出来，
    而评委临时给一份数据时多半是直接复制粘贴过来的。
    """
    pasted = (pasted or "").strip()
    if pasted:
        lines = [ln for ln in pasted.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise TableError("粘贴的内容至少要有表头和一行数据。")
        import csv as _csv
        import io as _io

        sample = lines[0] + "\n" + lines[1]
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except _csv.Error:
            delimiter = ","
        rows = list(_csv.reader(_io.StringIO("\n".join(lines)), delimiter=delimiter))
        if not rows:
            raise TableError("粘贴的内容解析不出表格。")
        return ingest_rows(
            headers=rows[0],
            rows=rows[1:],
            source="粘贴的文本",
            reader_note=f"分隔符 '{delimiter}'",
            max_rows=max_rows,
        )

    if not file_value:
        raise TableError("请先上传水质表格，或把表格内容粘贴到下面的框里。")

    path = file_value if isinstance(file_value, str) else getattr(file_value, "name", None)
    if not path:
        raise TableError("读不到上传文件的路径，请重新选择一次文件。")
    if not str(path).lower().endswith((".csv", ".tsv", ".xlsx", ".xlsm")):
        raise TableError(
            "只支持 CSV / TSV / XLSX。如果传的是图片，请放到右边的「鱼体图片」里。"
        )
    return ingest_table(path, max_rows=max_rows)


# --------------------------------------------------------------------------- #
# 主回调
# --------------------------------------------------------------------------- #

def _run(
    file_value: Any,
    pasted: str,
    image_path: Any,
    pond_id: str,
    species: str,
    mock_source: str,
    use_knowledge: bool,
    max_rows: int,
    session: WebSession,
):
    """点「运行 Agent」时执行。返回与 ``outputs`` 一一对应的元组。"""
    #: 出错时只填第一个输出（错误卡片），其余留空。
    #: 个数必须与 ``outputs`` 严格一致——差一个 Gradio 会直接报
    #: "The number of outputs does not match"，而这种错在现场演示时最难看。
    empty = ("",) * 7

    # ---- 1. 收数据 --------------------------------------------------- #
    ingest: TableIngest | None = None
    readings: list[dict[str, Any]] = []
    table_note = ""
    try:
        if (pasted or "").strip() or file_value:
            ingest = _load_table(file_value, pasted, int(max_rows))
            if ingest.ok:
                readings = ingest.readings
            elif ingest.errors:
                table_note = "表格有问题：" + "；".join(ingest.errors)
            else:
                table_note = (
                    f"表格只解析出 {ingest.n_points} 个可用点，"
                    "水质趋势至少需要 3 个点，本次将忽略水质模态。"
                )
    except TableError as exc:
        table_note = str(exc)
    except Exception as exc:  # noqa: BLE001
        table_note = f"读取表格时出错：{type(exc).__name__}: {exc}"

    img = image_path if isinstance(image_path, str) else getattr(image_path, "name", None)
    img = str(img) if img else None

    if not readings and not img:
        return (render_error(table_note or "还没有可用的输入。"),) + empty

    # ---- 2. 跑 Agent ------------------------------------------------- #
    req = WebRequest(
        readings=readings,
        image_path=img,
        pond_id=(pond_id or "unknown").strip(),
        species=species if species in SPECIES_CHOICES else SPECIES_CHOICES[0],
        mock_source=mock_source if mock_source in MOCK_SOURCE_CHOICES else "hash",
        use_knowledge=bool(use_knowledge),
    )
    result = session.run(req)
    result.ingest = ingest

    if not result.ok:
        return (render_error(result.error),) + empty

    # ---- 3. 排版 ----------------------------------------------------- #
    heads: list[str] = []
    if table_note:
        heads.append(f"> ⚠️ {table_note}")
    summary = "\n\n".join(heads + [render_summary_markdown(result)])
    return (
        summary,
        render_alerts(result),
        render_report(result),
        render_data_quality(result),
        render_vision(result),
        render_fusion(result),
        render_knowledge(result),
        render_trace(result) + "\n\n---\n\n" + render_status(result),
    )


def _backend_info(session: WebSession) -> str:
    return session_info(session)


def _template_choices() -> list[tuple[str, str]]:
    """下拉框选项：``(给用户看的名字, 传给回调的 key)``。"""
    return [(t.name, t.key) for t in all_templates()]


def _template_fill(key: str | None):
    """「用该模板填充」的回调，返回 ``(表格文件, 粘贴框, 品种, 提示)``。

    两件事必须一起做，否则模板"填了但没生效"：

    1. **清空粘贴框**——``_load_table`` 里粘贴内容优先于文件，
       用户先粘过一段文本、再点模板，跑的就还是那段旧文本。
    2. **切换品种**——模板里虾蟹塘的水温和溶氧是按虾蟹档位调的，
       留在鲤科会拿着对的数字按错的档位判级。
    """
    if not key:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            "⚠️ 请先在上面的下拉框里选一个模板。",
        )
    try:
        template = get_template(key)
        path = materialize(key)
    except KeyError as exc:
        # KeyError 的 str() 是参数的 repr，会多出一层引号；取 args[0] 更干净。
        msg = exc.args[0] if exc.args else str(exc)
        return gr.update(), gr.update(), gr.update(), f"❌ {msg}"
    note = (
        f"✅ 已填入 **{template.name}**（{template.species} · {template.n_points} 点 · "
        f"预期等级 **{template.expected_risk}**）\n\n"
        f"> {template.scenario}\n\n"
        f"数据文件：`{path.name}` —— 在仓库的 `templates/` 目录里，"
        f"直接上传这个文件也会得到同样的结果。点「▶ 运行 Agent」开始。"
    )
    return str(path), "", template.species, note


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #

def build_app(session: WebSession | None = None) -> gr.Blocks:
    """构建页面。``session`` 可注入，便于测试用假会话替换真实 Agent。"""
    sess = session or default_session()

    with gr.Blocks(
        title="ProjectFisheries · 水质—鱼病融合决策演示",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# 🐟 ProjectFisheries · 水质—鱼病融合决策演示\n"
            "上传**水质时序表格**和/或**鱼体图片**，点运行，"
            "Agent 会依次完成：调度工具 → 护栏校验 → 融合定级 → 检索知识库 → 生成建议。\n\n"
            "> 风险等级与告警全部由**规则引擎**计算（可复现、可审计），"
            "大模型只负责把结论写成养殖户能读的话，**不许改数字**。"
        )
        backend_box = gr.Markdown(_backend_info(sess))

        with gr.Row():
            # ------------------------ 左：输入 ------------------------ #
            with gr.Column(scale=4):
                gr.Markdown("### 1. 输入数据")

                gr.Markdown(
                    "**没有现成数据？** 选一个内置场景模板，一键填进去 ——"
                    "表格和品种会一起切过去，点运行就能看到对应的分级结论。"
                )
                with gr.Row():
                    template_pick = gr.Dropdown(
                        label="水质模板",
                        choices=_template_choices(),
                        value=None,
                        scale=4,
                        info="每个模板 24 个点（1 小时 1 点），预期等级已用规则引擎核对过",
                    )
                    template_btn = gr.Button("📥 用该模板填充", size="sm", scale=1)
                template_note = gr.Markdown(
                    "<div class='fisheries-hint'>模板文件在仓库的 "
                    "<code>templates/</code> 目录下，直接上传同名文件也能得到一样的结果。</div>"
                )

                table_file = gr.File(
                    label="水质时序表格（CSV / TSV / XLSX）",
                    file_count="single",
                    file_types=[".csv", ".tsv", ".xlsx", ".xlsm"],
                )
                with gr.Accordion(
                    "或者：直接粘贴表格内容（文件选择器不好用时的备用通道）",
                    open=False,
                ):
                    pasted = gr.Textbox(
                        label="粘贴 CSV / TSV（第一行是表头）",
                        lines=6,
                        placeholder=(
                            "timestamp,water_temp,ph,dissolved_oxygen,ammonia_n,nitrite_n\n"
                            "2026-10-13 08:00:00,26.9,8.05,6.42,0.10,0.05"
                        ),
                    )
                with gr.Row():
                    use_example = gr.Button("📄 填入示例数据", size="sm")
                    max_rows = gr.Slider(
                        label="最多读取行数",
                        minimum=12,
                        maximum=2000,
                        value=MAX_TABLE_ROWS,
                        step=12,
                    )

                image_in = gr.Image(
                    label="鱼体图片（可选，支持上传 / 拍照 / 粘贴）",
                    type="filepath",
                    sources=["upload", "webcam", "clipboard"],
                    height=220,
                )

                gr.Markdown("### 2. 参数")
                with gr.Row():
                    pond_id = gr.Textbox(label="塘口编号", value="demo-pond-1", scale=2)
                    species = gr.Dropdown(
                        label="养殖品种",
                        choices=list(SPECIES_CHOICES),
                        value=SPECIES_CHOICES[0],
                        scale=2,
                    )
                with gr.Row():
                    mock_source = gr.Dropdown(
                        label="病害识别模式",
                        choices=list(MOCK_SOURCE_CHOICES),
                        value="hash",
                        info="hash=按图片内容确定性模拟；folder_hint=按目录名取标签（仅联调）",
                        scale=2,
                    )
                    use_knowledge = gr.Checkbox(
                        label="引用知识库", value=True, scale=1,
                    )

                run_btn = gr.Button("▶ 运行 Agent", variant="primary", elem_id="run-btn")
                gr.Markdown(
                    "<div class='fisheries-hint'>"
                    "没有 GPU、没有 API key、断网也能跑：识别与建议会自动降级为"
                    "确定性模拟 + 规则模板，链路完全一致。"
                    "</div>"
                )

            # ------------------------ 右：结论 ------------------------ #
            with gr.Column(scale=6):
                gr.Markdown("### 决策报告")
                out_summary = gr.Markdown(_PLACEHOLDER, sanitize_html=False)
                out_alerts = gr.HTML()

                with gr.Accordion("📋 处置建议（水质 / 病害分列）", open=True):
                    out_report = gr.Markdown()
                with gr.Accordion("💧 水质数据接入质量", open=False):
                    out_quality = gr.Markdown()
                with gr.Accordion("🔬 鱼体图片识别结果", open=False):
                    out_vision = gr.Markdown()
                with gr.Accordion("⚖️ 融合依据（含等级下限）", open=False):
                    out_fusion = gr.Markdown()
                with gr.Accordion("📚 知识库引用", open=False):
                    out_knowledge = gr.Markdown()
                with gr.Accordion("🧠 推理链路（Agent 到底做了什么）", open=False):
                    out_trace = gr.Markdown()

        outputs = [
            out_summary,
            out_alerts,
            out_report,
            out_quality,
            out_vision,
            out_fusion,
            out_knowledge,
            out_trace,
        ]
        inputs = [
            table_file,
            pasted,
            image_in,
            pond_id,
            species,
            mock_source,
            use_knowledge,
            max_rows,
        ]

        run_btn.click(
            fn=lambda *args: _run(*args, session=sess),
            inputs=inputs,
            outputs=outputs,
        )
        use_example.click(
            fn=lambda: str(EXAMPLE_TABLE),
            inputs=None,
            outputs=table_file,
        )
        template_btn.click(
            fn=_template_fill,
            inputs=[template_pick],
            outputs=[table_file, pasted, species, template_note],
        )
        demo.load(fn=lambda: _backend_info(sess), inputs=None, outputs=backend_box)

        gr.Markdown(
            "---\n"
            "**离线演示说明**：本项目默认不联网。病害识别在未加载真实权重时"
            "按图片内容的 SHA-256 确定性模拟，用于验证**链路**而非精度；"
            "养殖建议在未配置 LLM 时由规则模板生成，与结构化字段完全一致。"
        )

    return demo


def main() -> None:
    """命令行入口：``python -m fisheries.webui.app``。"""
    host = os.environ.get("FISHERIES_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("FISHERIES_WEB_PORT", "7860"))
    share = str(os.environ.get("FISHERIES_WEB_SHARE", "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    open_browser = str(
        os.environ.get("FISHERIES_WEB_OPEN_BROWSER", "1")
    ).strip().lower() in {"1", "true", "yes", "on"}
    print("启动 ProjectFisheries 演示页……首次加载会读取向量库，请稍候。")
    build_app().queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        show_error=True,
        # Gradio 6 把 theme / css 从 Blocks() 挪到了 launch()。
        theme=gr.themes.Soft(primary_hue="teal"),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
