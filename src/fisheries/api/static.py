"""把 ``web/`` 里的静态前端挂进接口服务（同源托管）。

为什么由 FastAPI 托管而不是另起一个静态服务器
--------------------------------------------
前端本身是零依赖的原生 HTML/CSS/JS，源文件就是产物，没有构建步骤。
让它与接口同源，能一次消掉三个演示现场最容易出问题的地方：

1. **不用配 CORS**。跨源打开 ``file://`` 或另一个端口时，
   浏览器会先发预检请求，演示机上任何一个代理/安全软件都可能把它拦掉。
2. **不用手改基地址**。同源时前端直接用相对路径请求，端口换成多少都对。
3. **只有一条启动命令**。双击 ``启动接口服务.bat`` 就有接口也有页面，
   不存在"忘了起前端"这种故障。

挂载点为什么是 ``/app`` 而不是 ``/``
------------------------------------
``/`` 留给机器可读的服务目录（服务名、版本、端点清单、``/docs`` 地址）——
接口的调用方与评测脚本拿到的应该是一份 JSON，而不是一页 HTML。
``/docs`` / ``/openapi.json`` 也因此在静态目录之外，不受影响。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

__all__ = ["WEB_DIR_ENV", "WEB_MOUNT_PATH", "mount_web", "resolve_web_dir"]

#: 挂载点。与 ``app.py`` 里根路径返回的 ``"app"`` 字段保持一致。
WEB_MOUNT_PATH = "/app"

#: 显式指定前端目录的环境变量（打包分发、或把前端放到别处时用）。
WEB_DIR_ENV = "FISHERIES_WEB_DIR"


def resolve_web_dir() -> Path | None:
    """定位前端目录；找不到返回 ``None``。

    查找顺序：

    1. 环境变量 ``FISHERIES_WEB_DIR``（显式指定优先，方便打包后改位置）；
    2. 源码树 ``<仓库根>/web`` —— 本文件位于 ``src/fisheries/api/``，
       向上四级即仓库根。

    返回 ``None`` 而不是抛异常，是刻意的：一个只提供接口的部署
    （装成 wheel、只拷 ``src/``）不应该因为没有静态目录就起不来。
    """
    override = os.environ.get(WEB_DIR_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            return candidate

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "web"
    if candidate.is_dir():
        return candidate
    return None


def mount_web(app: FastAPI) -> Path | None:
    """把前端挂到 :data:`WEB_MOUNT_PATH`，返回实际使用的目录。

    **必须在所有 API 路由注册之后调用**。Starlette 按注册顺序匹配，
    ``/app`` 这个前缀虽然不会和 ``/api/v1/...`` 冲突，但把静态目录放在
    路由之前会让"先匹配到静态目录、接口 404"这类问题难以定位；
    统一放最后是最省心的顺序。

    ``html=True`` 让请求 ``/app/`` 时返回 ``index.html``。
    """
    web_dir = resolve_web_dir()
    if web_dir is None:
        return None
    app.mount(
        WEB_MOUNT_PATH,
        StaticFiles(directory=str(web_dir), html=True),
        name="web",
    )
    return web_dir
