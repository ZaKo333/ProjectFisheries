"""M7 后端接口层。

模块分工
--------
====== ============================================================
模块      职责
====== ============================================================
app      FastAPI 应用装配：lifespan 建单例、中间件、挂路由、uvicorn 入口
deps     进程级状态 ``AppState`` + 配置 ``ApiSettings`` + 依赖注入 ``get_state``
routes   全部端点实现（校验入参 → 调工具/Agent → 转响应契约）
schemas  请求 / 响应数据契约（Pydantic v2，全部 ``extra="forbid"``）
====== ============================================================

设计原则
--------
1. **接口层不实现业务规则。** 阈值判定、融合打分、协同规则全在工具层与
   ``agent/`` 里，接口只负责搬运和契约转换。同一条规则只有一处实现，
   才不会出现「接口算出来是高危、Agent 算出来是预警」这类自相矛盾。
2. **入参复用工具层模型。** ``/assess`` 的 ``readings`` 直接别名到
   ``tools.water_risk.WaterQualityPoint``，接口收得进的一定是工具吃得下的。
3. **出参对嵌套结果保持 ``dict``。** ``water`` / ``vision`` / ``fusion``
   的权威契约在 ``tools/definitions.py`` 的 ``TOOL_SCHEMAS``
   （``GET /api/v1/config/tools`` 可取），手抄一份等价模型只会静默漂移。
4. **降级不报错。** 知识库不可用、图像模型未训练、没配大模型——
   这些都在响应里用 ``degraded_mode`` / ``degraded_reasons`` 显式说明，
   而不是抛 500 让整个服务不可用。

零依赖可跑
----------
除 FastAPI 自身外不引入任何新依赖：Agent 层与 RAG 层的 HTTP 调用只用
标准库 ``urllib``。启动不需要 API key、不需要 GPU，图像模型未就绪时
自动走 mock 并在响应里标注 ``degraded_mode=true``。
"""

import importlib

_LAZY = {
    "app": (".app", "app"),
    "create_app": (".app", "create_app"),
    "main": (".app", "main"),
    "ApiSettings": (".deps", "ApiSettings"),
    "AppState": (".deps", "AppState"),
    "get_state": (".deps", "get_state"),
    "project_root": (".deps", "project_root"),
    "router": (".routes", "router"),
}


def __getattr__(name: str):
    """按需导入。

    ``python -m fisheries.api.app`` 会先导入包 ``fisheries.api``；若包的
    ``__init__`` 里就把 ``.app`` 拉进来，解释器会打印
    ``RuntimeWarning: 'fisheries.api.app' found in sys.modules ...``。
    """
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module_name, __name__), attr)

__all__ = [
    "ApiSettings",
    "AppState",
    "app",
    "create_app",
    "get_state",
    "main",
    "project_root",
    "router",
]
