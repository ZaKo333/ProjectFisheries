"""可选的一次性口令保护（Bearer 头 / 登录 cookie）。

为什么需要它
-----------
接口默认监听 ``127.0.0.1``，本机演示不需要任何鉴权。但一旦按
``docs/部署说明`` 那样放到公网（``FISHERIES_API_HOST=0.0.0.0`` + 端口映射），
任何扫到端口的人都能调 ``/assess``、往库里塞数据，甚至借 ``/chat`` 打满 CPU。
这一层只做一件事：**配置了口令就必须先过这道门**。

设计取舍
--------
* **不配就关**：``FISHERIES_API_TOKEN`` 为空时整套逻辑等于不存在，
  老的本地用法（双击 ``启动接口服务.bat``）一行都不用改。
* **两种凭证，一个口令**：
  - 脚本 / curl 用 ``Authorization: Bearer <token>``；
  - 浏览器用 ``/login`` 表单换成 ``fisheries_auth`` cookie
    （值是口令的 SHA-256，不把口令本身写进 cookie）。
* **公开路径写死在一处**：登录页、静态前端 ``/app``、健康检查、退出登录。
  健康检查必须公开，否则外部的存活探测也要带口令。
* 口令比对用 :func:`hmac.compare_digest`，避免按字节比较的时序差。

它不是什么
---------
这不是多用户账号系统：只有一个共享口令，没有用户名、没有权限分级、
没有审计日志。真要多人协作，应该在前面套 Nginx / 网关，而不是在这里长大。
"""

from __future__ import annotations

import hashlib
import hmac
import os
from urllib.parse import quote, urlparse

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

__all__ = [
    "AUTH_COOKIE",
    "LOGIN_PATH",
    "LOGIN_SUBMIT_PATH",
    "LOGOUT_PATH",
    "TOKEN_ENV",
    "configured_token",
    "cookie_value",
    "is_authorized",
    "is_public_path",
    "register_auth_routes",
    "safe_next",
    "unauthorized_response",
]

#: 存放共享口令的环境变量。空字符串等价于「不启用」。
TOKEN_ENV = "FISHERIES_API_TOKEN"

#: 登录成功后写入的 cookie 名。
AUTH_COOKIE = "fisheries_auth"

LOGIN_PATH = "/login"
LOGIN_SUBMIT_PATH = "/api/v1/auth/login"
LOGOUT_PATH = "/api/v1/auth/logout"
LOGOUT_GET_PATH = "/logout"

#: 不需要口令的精确路径。
PUBLIC_PATHS = frozenset(
    {
        LOGIN_PATH,
        LOGIN_SUBMIT_PATH,
        LOGOUT_PATH,
        LOGOUT_GET_PATH,
        "/api/v1/health",
        "/favicon.ico",
    }
)

#: 不需要口令的路径前缀（登录页自己的静态资源走不到这里，但前端整棵树都要放行，
#: 否则登录页都没法加载 css/js，用户会卡在一个空白页面上）。
PUBLIC_PREFIXES = ("/app",)

#: cookie 里存的是加了盐的口令摘要，salt 变更等于让所有旧 cookie 失效。
_COOKIE_SALT = "fisheries-auth:v1:"

#: 30 天：演示场景下没人愿意每天重新输入口令，与会话 cookie 保持一致。
_COOKIE_MAX_AGE = 30 * 24 * 3600


def configured_token() -> str | None:
    """读环境变量里的口令；未配置（或只有空白）返回 ``None``。"""
    raw = os.environ.get(TOKEN_ENV, "").strip()
    return raw or None


def cookie_value(token: str) -> str:
    """把口令散列成 cookie 里的值（不可逆，泄露 cookie 不等于泄露口令）。"""
    return hashlib.sha256((_COOKIE_SALT + token).encode("utf-8")).hexdigest()


def is_public_path(path: str) -> bool:
    """这个路径是否免口令。"""
    if path in PUBLIC_PATHS:
        return True
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PUBLIC_PREFIXES)


def is_authorized(request: Request, token: str) -> bool:
    """请求是否携带了有效凭证（Bearer 头或登录 cookie，任一即可）。"""
    header = request.headers.get("Authorization", "")
    if header[:7].lower() == "bearer " and hmac.compare_digest(header[7:].strip(), token):
        return True
    cookie = request.cookies.get(AUTH_COOKIE, "")
    return bool(cookie) and hmac.compare_digest(cookie, cookie_value(token))


def safe_next(raw: str | None) -> str:
    """只接受站内相对路径作为登录后的跳转目标，挡掉 ``//evil.example``。"""
    if not raw:
        return "/app/"
    candidate = raw.strip()
    if not candidate.startswith("/") or candidate.startswith("//"):
        return "/app/"
    parsed = urlparse(candidate)
    if parsed.scheme or parsed.netloc:
        return "/app/"
    return candidate


def _wants_html(request: Request) -> bool:
    """浏览器地址栏访问（GET + 接受 html）时给跳转，脚本调用时给 401 JSON。"""
    if request.method != "GET":
        return False
    return "text/html" in request.headers.get("accept", "")


def unauthorized_response(request: Request) -> Response:
    """未通过鉴权时的响应：页面跳登录，接口返回 401 JSON。"""
    path = request.url.path
    if path.startswith("/api/") or path in {"/openapi.json"} or path.startswith("/docs") or path.startswith("/redoc"):
        return JSONResponse(
            status_code=401,
            content={
                "detail": (
                    "需要访问口令：脚本请求请带 Authorization: Bearer <token>；"
                    "浏览器请先打开 /login 登录。"
                )
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    if _wants_html(request):
        return RedirectResponse(f"{LOGIN_PATH}?next={quote(path, safe='')}", status_code=303)
    return JSONResponse(
        status_code=401,
        content={"detail": "需要访问口令：请先打开 /login 登录。"},
        headers={"WWW-Authenticate": "Bearer"},
    )


_LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>需要访问口令 · 水产养殖智能监测</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
      font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
      background: #0f172a; color: #e2e8f0;
    }}
    .card {{
      width: min(380px, 90vw); padding: 28px 26px; border-radius: 14px;
      background: #1e293b; box-shadow: 0 12px 40px rgba(0, 0, 0, .45);
    }}
    h1 {{ margin: 0 0 6px; font-size: 19px; }}
    p.hint {{ margin: 0 0 18px; font-size: 13px; line-height: 1.6; color: #94a3b8; }}
    label {{ display: block; font-size: 13px; margin-bottom: 6px; color: #cbd5e1; }}
    input[type=password] {{
      width: 100%; box-sizing: border-box; padding: 10px 12px; font-size: 15px;
      border-radius: 8px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0;
    }}
    input[type=password]:focus {{ outline: 2px solid #38bdf8; outline-offset: 1px; }}
    button {{
      margin-top: 16px; width: 100%; padding: 10px 12px; font-size: 15px; cursor: pointer;
      border: 0; border-radius: 8px; background: #0284c7; color: #fff; font-weight: 600;
    }}
    button:hover {{ background: #0369a1; }}
    .err {{
      margin: 0 0 14px; padding: 9px 11px; border-radius: 8px; font-size: 13px;
      background: #7f1d1d; color: #fecaca;
    }}
    .foot {{ margin-top: 16px; font-size: 12px; color: #64748b; line-height: 1.6; }}
    code {{ background: #0f172a; padding: 1px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <form class="card" method="post" action="{action}">
    <h1>需要访问口令</h1>
    <p class="hint">该服务开启了口令保护。口令由部署者通过环境变量
      <code>FISHERIES_API_TOKEN</code> 设置。</p>
    {error}
    <label for="token">访问口令</label>
    <input id="token" name="token" type="password" autocomplete="current-password"
           autofocus required />
    <input type="hidden" name="next" value="{next}" />
    <button type="submit">进入</button>
    <p class="foot">脚本 / curl 调用可跳过本页：
      <code>Authorization: Bearer &lt;口令&gt;</code></p>
  </form>
</body>
</html>
"""


def _login_html(*, error: str = "", next_path: str = "/app/") -> str:
    block = f'<p class="err">{error}</p>' if error else ""
    return _LOGIN_HTML.format(action=LOGIN_SUBMIT_PATH, next=next_path, error=block)


def _cookie_response(target: str, token: str) -> RedirectResponse:
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        AUTH_COOKIE,
        cookie_value(token),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


def register_auth_routes(app: FastAPI) -> None:
    """挂上 ``/login``、登录提交与退出登录三个路由（口令未配置时也能访问）。"""

    @app.get(LOGIN_PATH, response_class=HTMLResponse, include_in_schema=False)
    def login_form(next: str = "") -> HTMLResponse:
        if configured_token() is None:
            # 没启用口令时别把人挡在一个假登录页上，直接送回应用。
            return RedirectResponse(safe_next(next), status_code=303)
        return HTMLResponse(_login_html(next_path=safe_next(next)))

    @app.post(LOGIN_SUBMIT_PATH, include_in_schema=False)
    def login_submit(token: str = Form(""), next: str = Form("")) -> Response:
        expected = configured_token()
        target = safe_next(next)
        if expected is None:
            return RedirectResponse(target, status_code=303)
        if not token or not hmac.compare_digest(token.strip(), expected):
            return HTMLResponse(
                _login_html(error="口令不正确。", next_path=target),
                status_code=401,
            )
        return _cookie_response(target, expected)

    @app.get(LOGOUT_GET_PATH, include_in_schema=False)
    @app.post(LOGOUT_PATH, include_in_schema=False)
    def logout() -> RedirectResponse:
        response = RedirectResponse(LOGIN_PATH, status_code=303)
        response.delete_cookie(AUTH_COOKIE, path="/")
        return response
