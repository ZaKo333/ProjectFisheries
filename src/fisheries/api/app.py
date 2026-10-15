"""FastAPI 应用装配（M7 后端接口层）。

职责边界
--------
本模块只做四件事：创建应用、装中间件、挂载路由、在 lifespan 里初始化
进程级的 ``AppState``。所有业务逻辑都在 ``routes.py`` → 工具层 / Agent 层，
**接口层不实现任何判定规则**。

启动方式::

    # 方式一：命令行（推荐，读取同一套环境变量）
    set PYTHONPATH=src
    .venv\\Scripts\\python.exe -m fisheries.api.app

    # 方式二：uvicorn 直接指定
    .venv\\Scripts\\python.exe -m uvicorn fisheries.api.app:app --reload --port 8000

为什么 ``create_app()`` 是个工厂函数而不是模块级建一个 ``app`` 变量：
测试要为每个用例建一个干净的应用（自己的知识库状态、自己的内存缓冲），
模块级单例会让测试之间互相污染，且无法注入替代实现。
但 uvicorn 需要 ``模块:变量`` 形式，所以在文件末尾仍然导出一个 ``app``。
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from .deps import ApiSettings, AppState
from .routes import router

__all__ = ["create_app", "app", "main"]

_STATE_ATTR = "fisheries_state"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时建一次 ``AppState``，关闭时不做特殊处理。

    把初始化放在 lifespan 而不是 ``@app.on_event("startup")``：
    前者是 FastAPI 现行推荐写法（后者已标记 deprecated），
    且能被 ``with TestClient(app) as client:`` 正确触发——
    这是接口测试能不能跑的前提。
    """
    settings: ApiSettings | None = getattr(app.state, "_pending_settings", None)
    state = AppState(settings)
    setattr(app.state, _STATE_ATTR, state)
    try:
        yield
    finally:
        # 内存态数据没有落盘需求，这里只做显式释放，避免测试间残留引用。
        state.buffers.clear()
        state.history.clear()


def create_app(
    settings: ApiSettings | None = None,
    *,
    enable_cors: bool | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。

    参数
    ----
    settings:
        接口层配置。为 None 时从环境变量读取（``ApiSettings.from_env()``）。
    enable_cors:
        是否装 CORS 中间件。默认看环境变量 ``FISHERIES_API_ENABLE_CORS``（默认开）。
        前端独立部署时必须开；生产部署应把允许来源收紧到具体域名。
    """
    application = FastAPI(
        title=(settings or ApiSettings.from_env()).api_title,
        version=__version__,
        description=(
            "水产养殖智能监测与决策 Agent 的后端接口。\n\n"
            "链路：数据校验 → 水质时序风险预测 → 鱼体病害识别 → 多源融合决策 "
            "→ RAG 检索规范依据 → 输出风险等级 / 告警 / 养殖建议。\n\n"
            "**免责声明**：水质指标的标准限值部分已对到 GB 11607-1989（渔业水质标准，现行）原文"
            "（2026-10-15 核验），但本项目的警戒线 / 危险线 / 红线是在标准之上的工程扩展，"
            "措施亦为工程默认值；正式使用前须由有资质的水产技术人员复核；"
            "本服务输出不构成用药指令。"
        ),
        lifespan=_lifespan,
    )

    if enable_cors is None:
        raw = os.environ.get("FISHERIES_API_ENABLE_CORS", "1").strip().lower()
        enable_cors = raw in {"1", "true", "yes", "on"}

    if enable_cors:
        # 演示期用通配来源；一旦对外部署，把这里换成具体前端域名，
        # 并注意 allow_credentials=True 与 allow_origins=["*"] 不能同时用。
        origins = [
            o.strip()
            for o in os.environ.get("FISHERIES_API_CORS_ORIGINS", "*").split(",")
            if o.strip()
        ]
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 让 lifespan 能拿到工厂传入的 settings，又不破坏 `uvicorn 模块:app` 的用法。
    application.state._pending_settings = settings  # type: ignore[attr-defined]

    application.include_router(router)

    @application.get("/", include_in_schema=False)
    def index() -> JSONResponse:
        """根路径给一份指路信息，避免打开首页看到 404 以为服务没起来。"""
        return JSONResponse(
            {
                "service": "ProjectFisheries Agent API",
                "version": __version__,
                "docs": "/docs",
                "openapi": "/openapi.json",
                "health": "/api/v1/health",
                "endpoints": [
                    "POST /api/v1/assess",
                    "POST /api/v1/tools/call",
                    "POST /api/v1/ingest/sensor",
                    "POST /api/v1/ingest/image",
                    "POST /api/v1/chat",
                    "GET  /api/v1/health",
                    "GET  /api/v1/ponds/{pond_id}/readings",
                    "GET  /api/v1/ponds/{pond_id}/history",
                    "GET  /api/v1/config/thresholds",
                    "GET  /api/v1/config/tools",
                ],
            }
        )

    return application


#: uvicorn 入口用的模块级应用。测试请用 ``create_app()`` 建独立实例。
app = create_app()


def main(argv: list[str] | None = None) -> int:
    """``python -m fisheries.api.app`` 的入口。

    默认**不开 reload**：reload 会 fork 子进程，Windows 上偶发端口占用，
    而且它按源码变更重启，会把内存态的会话缓冲清空——
    演示时刚 POST 进去的读数会在下一次改代码后凭空消失。
    需要热重载时显式传 ``--reload``。
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    import uvicorn

    host = os.environ.get("FISHERIES_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FISHERIES_API_PORT", "8000"))
    reload_flag = "--reload" in argv
    argv = [a for a in argv if a != "--reload"]

    print(f"[fisheries] 启动接口服务 http://{host}:{port}  (docs: /docs)")
    uvicorn.run(
        "fisheries.api.app:app",
        host=host,
        port=port,
        reload=reload_flag,
        log_level=os.environ.get("FISHERIES_API_LOG_LEVEL", "info"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
