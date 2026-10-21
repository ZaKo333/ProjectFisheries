"""访问口令（``FISHERIES_API_TOKEN``）的接口层测试。

测试策略
--------
* 口令是**进程级环境变量**，所以每个用例用 ``monkeypatch.setenv`` 单独设置，
  不依赖跑测试时机器上碰巧配没配。
* 全部走 ``TestClient``，验证的是真实中间件链路（不是 mock 一个依赖）——
  鉴权最容易出的错就是「中间件顺序错了，安全路径反被拦、公开路径反被放」。
* 断言只承诺**契约**：状态码、cookie 是否存在、是否跳转；
  不把登录页的具体文案钉死在测试里。
"""

from __future__ import annotations

import hashlib

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from fisheries.api.app import create_app
from fisheries.api.auth import (
    AUTH_COOKIE,
    LOGIN_PATH,
    LOGIN_SUBMIT_PATH,
    LOGOUT_PATH,
    TOKEN_ENV,
    cookie_value,
    is_authorized,
    safe_next,
)
from fisheries.api.deps import ApiSettings

TOKEN = "s3cret-token"


@pytest.fixture
def clean_env(monkeypatch):
    """默认「没配口令」，需要开的用例自己 setenv。"""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    return monkeypatch


@pytest.fixture
def client(tmp_path, clean_env):
    app = create_app(ApiSettings(upload_dir=tmp_path / "uploads"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def secured_client(tmp_path, clean_env):
    clean_env.setenv(TOKEN_ENV, TOKEN)
    app = create_app(ApiSettings(upload_dir=tmp_path / "uploads"))
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# 未配置口令：行为必须和以前完全一样
# --------------------------------------------------------------------------- #


class TestAuthDisabled:
    def test_api_is_open(self, client):
        assert client.get("/api/v1/config/thresholds").status_code == 200

    def test_root_reports_auth_not_required(self, client):
        body = client.get("/").json()
        assert body["auth_required"] is False

    def test_login_redirects_straight_to_app(self, client):
        resp = client.get(LOGIN_PATH, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"].endswith("/app/")

    def test_blank_variable_counts_as_disabled(self, tmp_path, clean_env):
        clean_env.setenv(TOKEN_ENV, "   ")
        app = create_app(ApiSettings(upload_dir=tmp_path / "uploads"))
        with TestClient(app) as c:
            assert c.get("/api/v1/config/tools").status_code == 200


# --------------------------------------------------------------------------- #
# 配置口令：接口必须挡住
# --------------------------------------------------------------------------- #


class TestProtectedApi:
    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/config/thresholds",
            "/api/v1/config/tools",
            "/api/v1/config/templates",
            "/api/v1/species",
            "/openapi.json",
        ],
    )
    def test_401_without_credentials(self, secured_client, path):
        resp = secured_client.get(path)
        assert resp.status_code == 401
        assert "Bearer" in resp.headers.get("www-authenticate", "")

    def test_bearer_allows_access(self, secured_client):
        resp = secured_client.get(
            "/api/v1/config/thresholds", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        assert resp.status_code == 200

    def test_bearer_scheme_is_case_insensitive(self, secured_client):
        resp = secured_client.get(
            "/api/v1/config/tools", headers={"Authorization": f"bearer {TOKEN}"}
        )
        assert resp.status_code == 200

    def test_wrong_bearer_is_rejected(self, secured_client):
        resp = secured_client.get(
            "/api/v1/config/tools", headers={"Authorization": "Bearer nope"}
        )
        assert resp.status_code == 401

    def test_post_endpoint_is_protected(self, secured_client):
        resp = secured_client.post("/api/v1/ingest/sensor", json={"pond_id": "p1", "readings": []})
        assert resp.status_code == 401

    def test_root_reports_auth_required(self, secured_client):
        # 根路径也受保护，所以从 401 的 detail 里确认服务是活的即可。
        body = secured_client.get("/").json()
        assert "口令" in body["detail"]


class TestPublicPaths:
    def test_health_is_public(self, secured_client):
        assert secured_client.get("/api/v1/health").status_code == 200

    def test_static_app_is_public(self, secured_client):
        assert secured_client.get("/app/").status_code == 200

    def test_login_page_is_public(self, secured_client):
        resp = secured_client.get(LOGIN_PATH)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


# --------------------------------------------------------------------------- #
# 浏览器登录流程
# --------------------------------------------------------------------------- #


class TestLoginFlow:
    def test_correct_token_sets_cookie_and_redirects(self, secured_client):
        resp = secured_client.post(
            LOGIN_SUBMIT_PATH,
            data={"token": TOKEN, "next": "/app/"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"].endswith("/app/")
        assert secured_client.cookies.get(AUTH_COOKIE)

    def test_cookie_then_grants_api_access(self, secured_client):
        secured_client.post(LOGIN_SUBMIT_PATH, data={"token": TOKEN, "next": "/app/"})
        assert secured_client.get("/api/v1/config/templates").status_code == 200

    def test_wrong_token_returns_401_without_cookie(self, secured_client):
        resp = secured_client.post(
            LOGIN_SUBMIT_PATH, data={"token": "wrong"}, follow_redirects=False
        )
        assert resp.status_code == 401
        assert not secured_client.cookies.get(AUTH_COOKIE)

    @pytest.mark.parametrize("typed", ["１１４３", "口令", "ｓ3cret-token"])
    def test_non_ascii_login_is_rejected_not_crashed(self, secured_client, typed):
        """全角数字 / 中文口令只会被判错，不能让登录页 500。

        ``hmac.compare_digest`` 拒收非 ASCII 的 ``str``（TypeError），线上曾因此
        把「输错口令」变成「服务器内部错误」。
        """
        resp = secured_client.post(
            LOGIN_SUBMIT_PATH, data={"token": typed, "next": "/app/"}, follow_redirects=False
        )
        assert resp.status_code == 401
        assert not secured_client.cookies.get(AUTH_COOKIE)

    def test_latin1_bearer_header_is_compared_as_bytes(self):
        """HTTP 头按 latin-1 解码，高位字节仍是「非 ASCII 字符」，同样不能抛 TypeError。"""
        token = "pässword"
        request = Request(
            {
                "type": "http",
                "headers": [(b"authorization", b"Bearer " + token.encode("latin-1"))],
            }
        )
        assert is_authorized(request, token) is True
        assert is_authorized(request, "another") is False

    def test_non_ascii_configured_token_can_be_used(self, tmp_path, clean_env):
        """口令本身就允许非 ASCII：配成中文时，照抄一遍必须能登录。"""
        clean_env.setenv(TOKEN_ENV, "口令１２３")
        app = create_app(ApiSettings(upload_dir=tmp_path / "uploads"))
        with TestClient(app) as c:
            assert c.get("/api/v1/config/tools").status_code == 401
            resp = c.post(
                LOGIN_SUBMIT_PATH, data={"token": "口令１２３"}, follow_redirects=False
            )
            assert resp.status_code == 303
            assert c.get("/api/v1/config/tools").status_code == 200

    def test_cookie_does_not_contain_the_token(self, secured_client):
        secured_client.post(LOGIN_SUBMIT_PATH, data={"token": TOKEN})
        stored = secured_client.cookies.get(AUTH_COOKIE)
        assert stored != TOKEN
        assert stored == cookie_value(TOKEN)
        assert len(stored) == 64 and all(ch in "0123456789abcdef" for ch in stored)

    def test_cookie_is_a_sha256_of_the_token(self):
        expected = hashlib.sha256(("fisheries-auth:v1:" + TOKEN).encode("utf-8")).hexdigest()
        assert cookie_value(TOKEN) == expected

    def test_logout_clears_access(self, secured_client):
        secured_client.post(LOGIN_SUBMIT_PATH, data={"token": TOKEN})
        assert secured_client.get("/api/v1/config/tools").status_code == 200
        assert secured_client.post(LOGOUT_PATH, follow_redirects=False).status_code == 303
        assert secured_client.get("/api/v1/config/tools").status_code == 401

    def test_html_navigation_redirects_to_login(self, secured_client):
        resp = secured_client.get(
            "/",
            headers={"Accept": "text/html"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"].startswith(LOGIN_PATH)

    def test_login_page_preserves_next(self, secured_client):
        resp = secured_client.get(f"{LOGIN_PATH}?next=%2Fdocs")
        assert 'value="/docs"' in resp.text


# --------------------------------------------------------------------------- #
# 跳转目标必须是站内路径
# --------------------------------------------------------------------------- #


class TestSafeNext:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("/docs", "/docs"),
            ("/app/#/qa", "/app/#/qa"),
            ("", "/app/"),
            (None, "/app/"),
            ("//evil.example/x", "/app/"),
            ("https://evil.example/x", "/app/"),
            ("javascript:alert(1)", "/app/"),
        ],
    )
    def test_only_site_relative_paths_survive(self, raw, expected):
        assert safe_next(raw) == expected
