"""现场演示用的 Web 界面（Gradio）。

这一层的定位非常明确：**只做「收数据、摆结果」，不做任何业务判断**。
风险等级、告警、养殖建议全部由 :mod:`fisheries.agent.orchestrator` 产出，
这里干的事只有三件：

1. 把用户上传的表格读成工具层认识的读数（:mod:`~fisheries.webui.table`）；
2. 调 Agent，把一次决策跑完（:mod:`~fisheries.webui.runner`）；
3. 把结果排成人能一眼看懂的页面（:mod:`~fisheries.webui.render`）。

拆成这四个模块还有个现实理由：``table`` / ``runner`` / ``render``
**都不 import gradio**，所以单元测试不需要装 gradio 就能跑；
只有 :mod:`~fisheries.webui.app` 依赖它。Gradio 也因此被声明成
可选依赖（``pip install projectfisheries[webui]``），
不让「换一台机器现场演示」变成「先装 300 MB 前端依赖」。
"""

from __future__ import annotations

from .render import (
    render_alerts,
    render_data_quality,
    render_knowledge,
    render_report,
    render_status,
    render_summary_markdown,
)
from .runner import WebRequest, WebResult, WebSession
from .table import (
    EXAMPLE_TABLE,
    TableError,
    TableIngest,
    ingest_rows,
    ingest_table,
)

__all__ = [
    "EXAMPLE_TABLE",
    "TableError",
    "TableIngest",
    "WebRequest",
    "WebResult",
    "WebSession",
    "build_app",
    "ingest_rows",
    "ingest_table",
    "render_alerts",
    "render_data_quality",
    "render_knowledge",
    "render_report",
    "render_status",
    "render_summary_markdown",
]


def build_app(*args: object, **kwargs: object) -> object:
    """惰性转发到 :func:`fisheries.webui.app.build_app`。

    做成惰性不是为了省几毫秒，而是为了**让没装 gradio 的环境也能
    ``import fisheries.webui``**——否则 ``table`` / ``render`` 的单元测试
    会因为一个顶层 import 而整体失败，而它们根本不需要 gradio。
    """
    from .app import build_app as _build_app

    return _build_app(*args, **kwargs)
