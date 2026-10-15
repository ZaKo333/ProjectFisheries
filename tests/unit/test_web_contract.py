"""前端契约测试：钉住 ``web/`` 与后端之间那些「靠人记」的对应关系。

背景：``web/`` 是纯原生 JS，没有构建步骤，也就没有类型系统替我们检查
「前端认的枚举码」和「后端真会发的码」是否一致。曾经就是这么错的 ——
`dict.js` 的 `REDLINES` 里列了两个后端永远不发的键，同时漏了三个真会发的码，
界面上直接显示英文码，而且没有任何测试会失败。

这里用**从后端源码取真值**的方式钉住它们，而不是在前端再抄一份常量表：
抄一份的结果是两边一起错，而且错得一致。
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fisheries.agent.orchestrator import TraceStep
from fisheries.api.app import create_app
from fisheries.tools import water_risk
from fisheries.tools.fish_disease import DISEASE_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
DICT_JS = WEB_DIR / "js" / "dict.js"
API_JS = WEB_DIR / "js" / "api.js"

pytestmark = pytest.mark.skipif(not WEB_DIR.is_dir(), reason="web/ 静态目录不在，跳过前端契约检查")


# --------------------------------------------------------------------------- #
# JS 源码里取对象键（不引入 JS 引擎：这几个对象都是扁平的 key: value 字面量）
# --------------------------------------------------------------------------- #
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def js_object_keys(source: str, const_name: str) -> set[str]:
    """取出 ``const <const_name> = { ... };`` 里的顶层键。

    支持裸键（``DO_REDLINE:``）、带引号的键（``"WATER_LEVEL:高危":`` ——
    内部含冒号，是最容易写错的地方）与数字键（``0.5:``）。
    按顶层逗号切分而不是按行切分：单行写死的对象（``SEVERITY_LABEL``）也要能取全。
    """
    m = re.search(rf"const\s+{re.escape(const_name)}\s*=\s*\{{", source)
    assert m, f"在 JS 里找不到 const {const_name}"

    i = m.end()
    depth = 1
    while i < len(source) and depth:
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    body = source[m.end(): i - 1]
    body = re.sub(r"//[^\n]*", "", body)   # 去掉行注释，注释里的逗号不算

    keys: set[str] = set()
    depth = 0
    in_str = False
    quote = ""
    cur = ""
    entries: list[str] = []
    for ch in body:
        if in_str:
            if ch == "\\":
                cur += ch
                continue
            if ch == quote:
                in_str = False
            cur += ch
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            cur += ch
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            entries.append(cur)
            cur = ""
        else:
            cur += ch
    entries.append(cur)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        hit = re.match(r'^(?:"([^"]+)"|(\d+(?:\.\d+)?)|([A-Za-z_$][\w$]*))\s*:', entry)
        if hit:
            keys.add(hit.group(1) or hit.group(2) or hit.group(3))
    return keys


def js_all_codes(source: str) -> set[str]:
    """JS 里出现的全部全大写码字面量（用于「覆盖性」类断言）。"""
    return set(re.findall(r'"([A-Z][A-Z0-9_]{4,})"', source))


def emitted_codes(func) -> set[str]:
    """从后端函数源码里取出它**真正 append 出去**的码。

    只看 ``hits.append("CODE")``，而不是扫函数体里所有全大写字符串 ——
    函数体里还会引用 ``REDLINE["PH_LOW_REDLINE"]`` 这类**阈值表的键**，
    它们从不作为码发出去，扫进来就会让断言要求前端登记一堆收不到的码
    （这正是本次要修的那个缺陷的镜像）。
    """
    return set(re.findall(r'hits\.append\(\s*"([A-Z][A-Z0-9_]{3,})"', inspect.getsource(func)))


#: 后端写 ``error_code`` 的四种写法，少覆盖一种就漏掉一整层。
_ERROR_CODE_PATTERNS = (
    r'_error\(\s*\w+\s*,\s*"([A-Z][A-Z0-9_]{3,})"',          # _error(path, "UNSUPPORTED_IMAGE", msg)
    r'error_code["\']?\s*[=:]\s*"([A-Z][A-Z0-9_]{3,})"',      # error_code="INVALID_INPUT" / "error_code": "TOOL_CRASHED"
    r'error_code["\']?\s*,\s*"([A-Z][A-Z0-9_]{3,})"',         # result.get("error_code", "UNKNOWN")
    r'error_code["\']?\s*\)\s*or\s*"([A-Z][A-Z0-9_]{3,})"',   # water.get("error_code") or "UNKNOWN"
)


def backend_error_codes() -> set[str]:
    """从 ``src/fisheries/`` 全树抠出所有可能写进 ``error_code`` 的码。

    这里**不写死任何一张表**。上一版只铺开了图像工具那一层、另外几个码靠手抄，
    结果是水质工具或调度层再加一个新码（比如 ``TIMEOUT``）时测试照样绿，
    界面上却开始显示英文 —— 正好是本次要修的那类缺陷的翻版。
    """
    codes: set[str] = set()
    for path in (PROJECT_ROOT / "src" / "fisheries").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in _ERROR_CODE_PATTERNS:
            codes |= set(re.findall(pattern, text))
    return codes


# --------------------------------------------------------------------------- #
# 1. 静态托管
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(create_app()) as c:
        yield c


class TestStaticHosting:
    def test_frontend_is_mounted_at_app(self, client: TestClient) -> None:
        r = client.get("/app/")
        assert r.status_code == 200
        assert "<title>" in r.text
        assert "css/app.css" in r.text

    def test_root_json_points_at_the_frontend(self, client: TestClient) -> None:
        body = client.get("/").json()
        assert body["app"] == "/app"
        assert isinstance(body["endpoints"], list) and len(body["endpoints"]) == 12

    @pytest.mark.parametrize(
        "name",
        ["js/api.js", "js/app.js", "js/charts.js", "js/dict.js", "js/markdown.js", "js/parse.js", "js/ui.js", "css/app.css"],
    )
    def test_every_asset_is_served(self, client: TestClient, name: str) -> None:
        r = client.get(f"/app/{name}")
        assert r.status_code == 200, name
        assert r.content, name

    def test_missing_asset_is_404_not_index(self, client: TestClient) -> None:
        assert client.get("/app/js/does-not-exist.js").status_code == 404

    def test_static_mount_does_not_swallow_the_api(self, client: TestClient) -> None:
        # 挂载顺序错了（静态目录先挂）就会把 /docs、/openapi.json、/api/v1/* 全吃掉
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/api/v1/health").status_code == 200

    def test_openapi_still_lists_every_route(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]
        assert len(paths) == 12
        assert "/api/v1/config/templates" in paths


# --------------------------------------------------------------------------- #
# 2. dict.js 的枚举必须与后端源码一致
# --------------------------------------------------------------------------- #
class TestDictContract:
    def test_redline_codes_match_check_redlines_exactly(self) -> None:
        """红线码：一个不能多（多了显示不出中文），一个不能少（少了显示英文码）。"""
        backend = emitted_codes(water_risk._check_redlines)
        frontend = js_object_keys(_read(DICT_JS), "REDLINES")
        assert frontend == backend, (
            f"前端多出：{sorted(frontend - backend)}；前端缺少：{sorted(backend - frontend)}"
        )

    def test_rule_hit_codes_cover_both_sources(self) -> None:
        """rule_hits = redline_hits + _check_normal_rules(...)，前端必须两边都登记。"""
        backend = emitted_codes(water_risk._check_redlines) | emitted_codes(water_risk._check_normal_rules)
        frontend = js_object_keys(_read(DICT_JS), "RULE_HITS")
        assert frontend == backend, (
            f"前端多出：{sorted(frontend - backend)}；前端缺少：{sorted(backend - frontend)}"
        )

    def test_redline_codes_are_also_rule_codes(self) -> None:
        """后端 rule_hits 把红线码也拼进去了，所以红线码必须同时出现在规则表里。"""
        src = _read(DICT_JS)
        redlines = js_object_keys(src, "REDLINES")
        rules = js_object_keys(src, "RULE_HITS")
        assert redlines <= rules

    def test_trace_kinds_match_the_literal(self) -> None:
        annotation = TraceStep.model_fields["kind"].annotation
        backend = set(getattr(annotation, "__args__", ()))
        assert backend, "TraceStep.kind 不再是 Literal？"
        assert js_object_keys(_read(DICT_JS), "TRACE_KINDS") == backend

    def test_disease_groups_cover_every_label(self) -> None:
        assert js_object_keys(_read(DICT_JS), "DISEASE_GROUPS") == set(DISEASE_LABELS)

    def test_tool_error_codes_match_the_backend_exactly(self) -> None:
        """图像工具 / 水质工具 / 调度层 / 兜底默认值 —— 四种来源一个不漏。"""
        backend = backend_error_codes()
        assert len(backend) >= 8, f"只抠到 {len(backend)} 个码，正则大概失配了：{sorted(backend)}"
        frontend = js_object_keys(_read(DICT_JS), "TOOL_ERRORS")
        assert frontend == backend, (
            f"界面上会显示成英文：{sorted(backend - frontend)}；"
            f"后端从不发、可以删掉：{sorted(frontend - backend)}"
        )

    def test_severity_labels_cover_the_three_levels(self) -> None:
        assert js_object_keys(_read(DICT_JS), "SEVERITY_LABEL") == {"0", "0.5", "1"}

    def test_removed_mock_only_config_does_not_come_back(self) -> None:
        """`DISEASE_SEVERITY` 是前端重算严重度的死配置，已删；别再有人加回来。"""
        assert "DISEASE_SEVERITY" not in _read(DICT_JS)


# --------------------------------------------------------------------------- #
# 3. api.js 调的路径必须是后端真有的
# --------------------------------------------------------------------------- #
class TestApiPathContract:
    def test_every_called_path_exists(self, client: TestClient) -> None:
        declared = set(client.get("/openapi.json").json()["paths"])
        called = set(re.findall(r'"(/api/v1/[^"]*)"', _read(API_JS)))

        def known(path: str) -> bool:
            if path in declared:
                return True
            # 拼出来的路径：`"/api/v1/ponds/" + id + "/readings"` 这类前缀，
            # 只有后面紧跟 `{...}` 路径段时才认，避免把随便什么前缀都放过。
            return any(d.startswith(path) and d[len(path):].startswith("{") for d in declared)

        unknown = {p for p in called if not known(p)}
        assert not unknown, f"前端调了后端没有的路径：{sorted(unknown)}"

    def test_no_mock_only_base_remains(self) -> None:
        src = _read(API_JS)
        assert "mock-server.mjs" not in src
        assert "assessScoped" not in src
        # 8787 只允许出现在「历史遗留地址、要忽略」的名单里
        assert src.count("8787") <= 2, "mock 端口只该出现在 LEGACY_BASES 里"

    def test_frontend_never_recomputes_a_level(self) -> None:
        """等级与分数只许来自后端。"""
        for name in ("app.js", "ui.js", "charts.js"):
            src = _read(WEB_DIR / "js" / name)
            assert "fused_score =" not in src
            assert "risk_level =" not in src
