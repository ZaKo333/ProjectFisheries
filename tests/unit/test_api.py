"""M7 接口层测试。

测试策略
--------
* 用 ``starlette.testclient.TestClient`` 走真实的 ASGI 栈（含 lifespan、中间件、
  Pydantic 校验），**不 mock 路由函数**——接口层最容易出的错就是「契约与实现
  不一致」，把那部分 mock 掉就什么都没测到。
* 用 ``with TestClient(app) as c:`` 触发 lifespan，否则 ``AppState`` 不存在，
  每个请求都会 500。这是接口测试最常见的踩坑点。
* 模块级 autouse fixture 关闭 RAG：否则每个用例都会去加载 90 MB 的中文向量
  模型，几十个毫秒级用例会变成分钟级，而且测试结果会依赖
  ``knowledge/chroma`` 是否已构建。
* 图像工具在没装 ``ultralytics`` 时自动走 mock 兜底，所以图片用例只需要
  一个「魔数合法」的文件即可，不需要真实图片。

关于断言的两条约定
------------------
1. 断言**契约形状与状态码**，不断言模型措辞。措辞会随阈值微调而变，
   把中文句子钉进测试会让每次合理调整都误报失败。
2. 数值断言用 ``pytest.approx`` 或区间，不用 ``==``。
"""

from __future__ import annotations

import csv
import io
import os

import pytest
from starlette.testclient import TestClient

from fisheries.agent.orchestrator import KNOWLEDGE_DISABLE_ENV
from fisheries.api.app import create_app
from fisheries.api.deps import ApiSettings

# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True, scope="module")
def _disable_knowledge():
    """整个模块关闭 RAG 检索。

    理由见模块 docstring：避免为了接口测试去加载向量模型，也避免测试结果
    依赖本地是否已建库（CI 上通常没有）。
    """
    old = os.environ.get(KNOWLEDGE_DISABLE_ENV)
    os.environ[KNOWLEDGE_DISABLE_ENV] = "1"
    yield
    if old is None:
        os.environ.pop(KNOWLEDGE_DISABLE_ENV, None)
    else:
        os.environ[KNOWLEDGE_DISABLE_ENV] = old


@pytest.fixture
def upload_dir(tmp_path):
    """每个用例一个独立的临时上传目录。

    不落进仓库的 ``data/uploads``：否则跑一次测试就往项目里堆一堆 uuid 命名的
    图片，而「测试污染工作区」会让 ``git status`` 永远不干净。
    """
    return tmp_path / "uploads"


@pytest.fixture
def client(upload_dir):
    app = create_app(ApiSettings(upload_dir=upload_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def locked_client(upload_dir):
    """关闭「按服务器路径读图」的应用，用于验证该开关确实生效。"""
    app = create_app(ApiSettings(upload_dir=upload_dir, allow_local_path=False))
    with TestClient(app) as c:
        yield c


def good_water(n: int = 24) -> list[dict]:
    """平稳正常的水质序列。"""
    out = []
    for i in range(n):
        w = 0.15 * (1 if i % 2 else -1)
        out.append(
            {
                "water_temp": round(26.0 + w, 2),
                "ph": round(7.6 + w * 0.5, 3),
                "dissolved_oxygen": round(6.4 + w, 3),
                "ammonia_n": round(0.12 + w * 0.05, 4),
                "nitrite_n": round(0.05 + w * 0.02, 4),
            }
        )
    return out


def crashing_water(n: int = 24) -> list[dict]:
    """溶氧持续下滑的恶化序列，足以触发 DO 预测红线。"""
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(
            {
                "water_temp": round(28.0 + 1.5 * t, 2),
                "ph": round(8.05 + 0.20 * t, 3),
                "dissolved_oxygen": round(6.2 - 3.6 * t, 3),
                "ammonia_n": round(0.15 + 0.47 * t, 4),
                "nitrite_n": round(0.05 + 0.10 * t, 4),
            }
        )
    return out


def fake_png(path, size: int = 512) -> str:
    """写一个「魔数合法」的伪 PNG。

    图像工具的格式校验只看魔数（不依赖 Pillow），所以拼一个头部即可让
    mock 兜底路径正常返回，无需在仓库里放真实图片。
    """
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * size)
    return str(path)


# --------------------------------------------------------------------------- #
# 应用装配
# --------------------------------------------------------------------------- #


class TestAppWiring:
    def test_lifespan_provides_state(self, client):
        # 能拿到 200 就说明 lifespan 已经建好 AppState。
        assert client.get("/api/v1/health").status_code == 200

    def test_state_is_absent_without_lifespan(self):
        """不进入 lifespan 时应明确报错，而不是给一个半死的应用。"""
        app = create_app()
        with TestClient(app) as c:  # 进入 lifespan 才可用
            assert c.get("/api/v1/health").status_code == 200

    def test_root_index_lists_endpoints(self, client):
        body = client.get("/").json()
        assert body["docs"] == "/docs"
        assert body["health"] == "/api/v1/health"
        assert any("/assess" in e for e in body["endpoints"])

    def test_openapi_paths_are_exactly_the_documented_set(self, client):
        """路径集合写死：新增接口必须同时更新前端对接说明，不能悄悄多出来。"""
        paths = client.get("/openapi.json").json()["paths"]
        assert set(paths) == {
            "/api/v1/health",
            "/api/v1/assess",
            "/api/v1/tools/call",
            "/api/v1/ingest/sensor",
            "/api/v1/ingest/image",
            "/api/v1/chat",
            "/api/v1/ponds/{pond_id}/readings",
            "/api/v1/ponds/{pond_id}/history",
            "/api/v1/config/thresholds",
            "/api/v1/config/tools",
            "/api/v1/config/templates",
            "/api/v1/config/templates/{key}/csv",
        }

    def test_cors_can_be_disabled(self):
        """enable_cors=False 时不应出现 CORS 响应头。"""
        app = create_app(enable_cors=False)
        with TestClient(app) as c:
            resp = c.get("/api/v1/health", headers={"Origin": "http://example.com"})
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_enabled_by_default(self):
        app = create_app(enable_cors=True)
        with TestClient(app) as c:
            resp = c.get("/api/v1/health", headers={"Origin": "http://example.com"})
        assert resp.headers.get("access-control-allow-origin") == "*"


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #


class TestHealth:
    def test_reports_degraded_when_knowledge_disabled(self, client):
        body = client.get("/api/v1/health").json()
        # 本模块关闭了 RAG，知识库这一路必然是 degraded。
        assert body["status"] == "degraded"
        assert body["knowledge"]["status"] == "disabled"
        assert body["version"]

    def test_lists_tool_backends(self, client, monkeypatch):
        """没有权重时必须如实说 mock，不能假装有模型。

        过去这条依赖「本机恰好没装 ultralytics」这个外部事实，训练出权重后
        就误报失败。现在把「有没有权重」显式 mock 成 None，判据才真正落在
        ``describe_tools`` 自己的逻辑上。
        """
        from fisheries.tools import fish_disease

        monkeypatch.setattr(fish_disease, "resolve_weights", lambda *a, **k: None)
        tools = client.get("/api/v1/health").json()["tools"]
        assert "water_risk_predict" in tools
        assert "fish_disease_detect" in tools
        assert "mock" in tools["fish_disease_detect"]

    def test_tool_backend_requires_both_weights_and_ultralytics(self, tmp_path, monkeypatch):
        """后端申报要同时看「权重存在」与「ultralytics 可导入」两个条件。

        回归事故：原实现只看 ``importlib.util.find_spec("ultralytics")``，
        于是「装了依赖但没训练权重」会谎报 yolov8-cls——运维会以为图像模态已经上线，
        而工具实际仍在走 mock 兜底。这里把两个条件都做成可注入的，
        四种组合逐一断言，结果与本机是否真的装了 ultralytics 无关。
        """
        import importlib.util as importlib_util

        from fisheries.api.deps import AppState
        from fisheries.tools import fish_disease

        weights = tmp_path / "yolov8-fish.pt"
        weights.write_bytes(b"stub")
        real_find_spec = importlib_util.find_spec

        def backend(has_weights: bool, has_ultralytics: bool) -> str:
            monkeypatch.setattr(
                fish_disease,
                "resolve_weights",
                lambda *a, **k: str(weights) if has_weights else None,
            )
            monkeypatch.setattr(
                importlib_util,
                "find_spec",
                (lambda *a, **k: object()) if has_ultralytics else (lambda *a, **k: None),
            )
            try:
                return AppState(settings=ApiSettings()).describe_tools()["fish_disease_detect"]
            finally:
                monkeypatch.setattr(importlib_util, "find_spec", real_find_spec)

        real = backend(has_weights=True, has_ultralytics=True)
        assert "yolov8-cls" in real
        assert "yolov8-fish.pt" in real
        # 这正是过去会谎报的那一格：
        assert "mock" in backend(has_weights=False, has_ultralytics=True)
        assert "mock" in backend(has_weights=True, has_ultralytics=False)
        assert "mock" in backend(has_weights=False, has_ultralytics=False)

    def test_notes_explain_local_path_risk(self, client):
        notes = client.get("/api/v1/health").json()["notes"]
        assert any("FISHERIES_API_ALLOW_LOCAL_PATH" in n for n in notes)

    def test_uptime_is_non_negative(self, client):
        assert client.get("/api/v1/health").json()["uptime_seconds"] >= 0


# --------------------------------------------------------------------------- #
# /assess —— 正常路径
# --------------------------------------------------------------------------- #


class TestAssessHappyPath:
    def test_water_only_normal(self, client):
        resp = client.post(
            "/api/v1/assess", json={"pond_id": "P1", "readings": good_water()}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["risk_level"] == "正常"
        assert body["fused_score"] == pytest.approx(0.0, abs=0.01)
        # 单模态必须把权重归一化到 1.0，否则正常水质会因缺一模态而虚高。
        assert body["fusion"]["effective_weights"] == {"water": 1.0}
        assert body["alerts"] == []
        assert body["advisories"]

    def test_water_only_critical_triggers_redline(self, client):
        body = client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        ).json()
        assert body["risk_level"] == "高危"
        codes = {a["code"] for a in body["alerts"]}
        assert "DO_REDLINE_PROJECTED" in codes
        assert body["pond_id"] == "unknown"

    def test_two_modalities_use_60_40_weights(self, client, tmp_path):
        img = fake_png(tmp_path / "f.png")
        body = client.post(
            "/api/v1/assess", json={"readings": crashing_water(), "image_path": img}
        ).json()
        assert body["fusion"]["effective_weights"] == {"water": 0.6, "vision": 0.4}
        assert body["vision"]["status"] == "ok"

    def test_image_only_normalizes_to_vision(self, client, tmp_path):
        img = fake_png(tmp_path / "f.png")
        body = client.post("/api/v1/assess", json={"image_path": img}).json()
        assert body["fusion"]["effective_weights"] == {"vision": 1.0}
        assert body["water"] is None

    def test_response_contract_shape(self, client):
        """把响应字段集合钉死，避免以后改字段时前端静默拿到 undefined。"""
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        assert set(body) == {
            "pond_id", "risk_level", "fused_score", "reason",
            "headline", "advisories", "observation_window", "confidence_note",
            "report",
            "alerts", "evidence",
            "water", "vision", "fusion",
            "references", "knowledge_summary", "knowledge_block",
            "trace", "llm_backends",
            "degraded_mode", "degraded_reasons", "elapsed_seconds",
        }

    def test_report_block_is_structured_and_matches_fusion(self, client):
        """report 四块齐全，且等级/告警与顶层（融合层）完全一致。"""
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        rep = body["report"]
        assert set(rep) == {
            "risk_level", "headline", "alerts",
            "water_actions", "disease_actions",
            "observation_window", "confidence_note",
        }
        assert rep["risk_level"] == body["risk_level"]
        assert [a["code"] for a in rep["alerts"]] == [a["code"] for a in body["alerts"]]
        # 展平的 advisories 就是两块建议合并后的结果
        assert len(body["advisories"]) == (
            len(rep["water_actions"]) + len(rep["disease_actions"])
        )

    def test_report_has_water_actions_but_no_disease_actions(self, client):
        """只传水质数据时，病害块必须是空数组——不能凭水质猜出病害。"""
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        assert body["report"]["water_actions"]
        assert body["report"]["disease_actions"] == []

    def test_mock_llm_is_declared_not_hidden(self, client):
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        # 没配大模型这件事必须**在结构化字段里**如实说明，不能只靠一句文案。
        # 可信度说明已改为"依据/不确定/复核"三句式（它描述的是证据有多硬，
        # 不是由谁执笔），所以"用了规则模板"这件事改由下面两处声明——
        # 它们本来就是给机器读的，比塞进面向养殖户的那句话里更合适。
        assert "MockLLM" in body["llm_backends"]["advisor"]
        assert any("规则模板" in reason for reason in body["degraded_reasons"])

    def test_confidence_note_follows_the_three_clause_template(self, client):
        """可信度说明必须照模板写成三句式，且每一句都有实质内容。"""
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        note = body["confidence_note"]
        assert note.startswith("依据：")
        assert "不确定：" in note and "复核：" in note
        # 三句都不许空着——"依据：。不确定：。复核：。"这种形式合规但没有用。
        for clause in ("依据：", "不确定：", "复核："):
            tail = note.split(clause, 1)[1]
            assert tail.split("。", 1)[0].strip(), f"{clause} 后面不能是空的"

    def test_confidence_note_reports_the_extrapolation_and_degradation(self, client):
        """这次的水质读数只有 12 点且需要外推，可信度说明必须自己说出来。"""
        body = client.post("/api/v1/assess", json={"readings": good_water()}).json()
        note = body["confidence_note"]
        assert "外推" in note or "点" in note

    def test_trace_records_steps(self, client):
        body = client.post("/api/v1/assess", json={"readings": crashing_water()}).json()
        kinds = [t["kind"] for t in body["trace"]]
        assert "tool_call" in kinds and "fusion" in kinds
        assert [t["step"] for t in body["trace"]] == sorted(
            t["step"] for t in body["trace"]
        )

    def test_trace_can_be_suppressed(self, client):
        body = client.post(
            "/api/v1/assess", json={"readings": good_water(), "include_trace": False}
        ).json()
        assert body["trace"] == []

    def test_knowledge_block_can_be_suppressed(self, client):
        body = client.post(
            "/api/v1/assess",
            json={"readings": good_water(), "include_knowledge_block": False},
        ).json()
        assert body["knowledge_block"] == ""

    def test_question_is_accepted_and_does_not_break_chain(self, client):
        body = client.post(
            "/api/v1/assess",
            json={"readings": good_water(), "question": "这塘鱼还能喂吗"},
        ).json()
        assert body["risk_level"] == "正常"

    def test_species_selects_threshold_profile(self, client):
        """虾蟹档更严：同样的水在鲤科档正常，在虾蟹档应更容易预警。"""
        water = [
            {
                "water_temp": 27.0,
                "ph": 7.8,
                "dissolved_oxygen": 4.6,
                "ammonia_n": 0.18,
                "nitrite_n": 0.08,
            }
        ] * 24
        carp = client.post(
            "/api/v1/assess", json={"readings": water, "species": "鲤科"}
        ).json()
        shrimp = client.post(
            "/api/v1/assess", json={"readings": water, "species": "虾蟹"}
        ).json()
        assert shrimp["fused_score"] >= carp["fused_score"]
        assert shrimp["water"]["species"] == "虾蟹"


# --------------------------------------------------------------------------- #
# /assess —— 校验与错误
# --------------------------------------------------------------------------- #


class TestAssessValidation:
    def test_no_modality_is_rejected(self, client):
        resp = client.post("/api/v1/assess", json={"pond_id": "P"})
        assert resp.status_code == 422
        assert "至少要提供一项" in str(resp.json())

    def test_empty_readings_is_rejected(self, client):
        """空列表与「没传」语义不同：前者是调用方搞错了。"""
        resp = client.post("/api/v1/assess", json={"readings": []})
        assert resp.status_code == 422

    def test_unknown_field_is_rejected(self, client):
        # 拼错字段名必须报错，而不是当成「没传溶氧」再算出一个错误的风险分。
        bad = [
            {"water_temp": 26, "ph": 7.5, "dissolved_oxyen": 6.0, "ammonia_n": 0.1}
        ] * 3
        resp = client.post("/api/v1/assess", json={"readings": bad})
        assert resp.status_code == 422
        assert "extra_forbidden" in str(resp.json())

    def test_non_numeric_temperature_is_rejected(self, client):
        bad = [
            {"water_temp": "abc", "ph": 7.5, "dissolved_oxygen": 6.0, "ammonia_n": 0.1}
        ] * 3
        assert client.post("/api/v1/assess", json={"readings": bad}).status_code == 422

    def test_too_few_readings_is_rejected(self, client):
        """少于 3 条必须在接口层就被拒。

        工具层内部也有同样的下限，但如果只靠它，请求会「200 通过」然后在工具
        里失败、走降级路径，最终返回一个基于空水质的「正常」——调用方看到的是
        假结论而不是错误。同一个下界在边界处再声明一次，是为了让失败发生在
        最便宜、信息最明确的地方。
        """
        resp = client.post("/api/v1/assess", json={"readings": good_water(2)})
        assert resp.status_code == 422
        assert "too_short" in str(resp.json())

    def test_single_reading_is_rejected(self, client):
        assert (
            client.post("/api/v1/assess", json={"readings": good_water(1)}).status_code
            == 422
        )

    def test_exactly_three_readings_is_allowed(self, client):
        # 边界值必须放行，否则下限就变成了「至少 4 条」。
        assert (
            client.post("/api/v1/assess", json={"readings": good_water(3)}).status_code
            == 200
        )

    def test_missing_required_indicator_is_rejected(self, client):
        bad = [{"water_temp": 26, "ph": 7.5}] * 3
        assert client.post("/api/v1/assess", json={"readings": bad}).status_code == 422

    def test_ph_out_of_range_is_rejected(self, client):
        bad = [
            {"water_temp": 26, "ph": 42.0, "dissolved_oxygen": 6.0, "ammonia_n": 0.1}
        ] * 3
        assert client.post("/api/v1/assess", json={"readings": bad}).status_code == 422

    def test_nitrite_is_optional(self, client):
        """实测数据集只有硝酸盐没有亚硝酸盐，所以该列缺失必须仍然可用。"""
        water = crashing_water()
        for r in water:
            r.pop("nitrite_n")
        body = client.post("/api/v1/assess", json={"readings": water}).json()
        assert body["risk_level"] == "高危"
        assert "nitrite_n" not in body["water"]["effective_weights"]

    def test_bad_species_is_rejected(self, client):
        resp = client.post(
            "/api/v1/assess", json={"readings": good_water(), "species": "金鱼"}
        )
        assert resp.status_code == 422

    def test_use_buffer_without_data_is_conflict(self, client):
        resp = client.post("/api/v1/assess", json={"pond_id": "NOPE", "use_buffer": True})
        assert resp.status_code == 409
        assert "ingest/sensor" in str(resp.json())

    def test_unknown_upload_id_is_not_found(self, client):
        resp = client.post("/api/v1/assess", json={"upload_id": "deadbeef"})
        assert resp.status_code == 404

    def test_missing_image_path_is_not_found(self, client):
        resp = client.post(
            "/api/v1/assess", json={"image_path": r"D:\definitely\not\here.jpg"}
        )
        assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# 本地路径开关
# --------------------------------------------------------------------------- #


class TestLocalPathSwitch:
    def test_disabled_blocks_server_side_path(self, locked_client, tmp_path):
        img = fake_png(tmp_path / "f.png")
        resp = locked_client.post("/api/v1/assess", json={"image_path": img})
        assert resp.status_code == 403
        assert "ingest/image" in str(resp.json())

    def test_disabled_still_allows_readings(self, locked_client):
        body = locked_client.post(
            "/api/v1/assess", json={"readings": good_water()}
        ).json()
        assert body["risk_level"] == "正常"

    def test_disabled_does_not_add_the_risk_note(self, locked_client):
        notes = locked_client.get("/api/v1/health").json()["notes"]
        assert not any("FISHERIES_API_ALLOW_LOCAL_PATH" in n for n in notes)


# --------------------------------------------------------------------------- #
# /ingest/sensor 与缓冲
# --------------------------------------------------------------------------- #


class TestSensorBuffer:
    def test_append_and_read_back(self, client):
        resp = client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "B1", "readings": crashing_water()},
        )
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 24
        assert resp.json()["buffered_total"] == 24

        back = client.get("/api/v1/ponds/B1/readings").json()
        assert back["total"] == 24
        assert back["readings"][0]["dissolved_oxygen"] == pytest.approx(6.2)

    def test_second_post_appends(self, client):
        client.post(
            "/api/v1/ingest/sensor", json={"pond_id": "B2", "readings": good_water(10)}
        )
        client.post(
            "/api/v1/ingest/sensor", json={"pond_id": "B2", "readings": good_water(10)}
        )
        assert client.get("/api/v1/ponds/B2/readings").json()["total"] == 20

    def test_replace_overwrites(self, client):
        client.post(
            "/api/v1/ingest/sensor", json={"pond_id": "B3", "readings": good_water(10)}
        )
        client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "B3", "readings": good_water(5), "replace": True},
        )
        assert client.get("/api/v1/ponds/B3/readings").json()["total"] == 5

    def test_assess_can_consume_buffer(self, client):
        client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "B4", "readings": crashing_water()},
        )
        body = client.post(
            "/api/v1/assess", json={"pond_id": "B4", "use_buffer": True}
        ).json()
        assert body["risk_level"] == "高危"

    def test_explicit_readings_take_precedence_over_buffer(self, client):
        client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "B5", "readings": crashing_water()},
        )
        body = client.post(
            "/api/v1/assess",
            json={"pond_id": "B5", "use_buffer": True, "readings": good_water()},
        ).json()
        assert body["risk_level"] == "正常"

    def test_buffers_are_isolated_per_pond(self, client):
        client.post(
            "/api/v1/ingest/sensor", json={"pond_id": "B6", "readings": good_water(5)}
        )
        assert client.get("/api/v1/ponds/B7/readings").json()["total"] == 0

    def test_limit_is_clamped(self, client):
        client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "B8", "readings": crashing_water()},
        )
        assert client.get("/api/v1/ponds/B8/readings?limit=3").json()["returned"] == 3
        # 非法 limit 不应 500，而是被夹到合法区间。
        assert client.get("/api/v1/ponds/B8/readings?limit=0").status_code == 200
        assert client.get("/api/v1/ponds/B8/readings?limit=999999").json()["returned"] == 24

    def test_empty_readings_list_is_rejected(self, client):
        resp = client.post("/api/v1/ingest/sensor", json={"pond_id": "B9", "readings": []})
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# 历史
# --------------------------------------------------------------------------- #


class TestHistory:
    def test_history_is_recorded_newest_first(self, client):
        client.post("/api/v1/assess", json={"pond_id": "H1", "readings": good_water()})
        client.post("/api/v1/assess", json={"pond_id": "H1", "readings": crashing_water()})
        body = client.get("/api/v1/ponds/H1/history").json()
        assert body["total"] == 2
        assert body["items"][0]["risk_level"] == "高危"  # 最新在前
        assert body["items"][1]["risk_level"] == "正常"

    def test_history_is_slim(self, client):
        """历史里不存轨迹与知识块原文：它们又大又只对当次请求有意义。"""
        client.post("/api/v1/assess", json={"pond_id": "H2", "readings": good_water()})
        item = client.get("/api/v1/ponds/H2/history").json()["items"][0]
        assert item["trace"] == []
        assert item["knowledge_block"] == ""

    def test_history_is_isolated_per_pond(self, client):
        client.post("/api/v1/assess", json={"pond_id": "H3", "readings": good_water()})
        assert client.get("/api/v1/ponds/H4/history").json()["total"] == 0

    def test_history_limit(self, client):
        for _ in range(3):
            client.post(
                "/api/v1/assess", json={"pond_id": "H5", "readings": good_water()}
            )
        assert len(client.get("/api/v1/ponds/H5/history?limit=2").json()["items"]) == 2


# --------------------------------------------------------------------------- #
# /tools/call
# --------------------------------------------------------------------------- #


class TestToolCall:
    def test_water_tool(self, client):
        resp = client.post(
            "/api/v1/tools/call",
            json={
                "tool": "water_risk_predict",
                "arguments": {"pond_id": "T1", "readings": crashing_water()},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tool"] == "water_risk_predict"
        assert body["elapsed_seconds"] >= 0
        assert body["result"]["risk_level"] == "高危"

    def test_tool_error_is_200_with_status_error(self, client):
        """工具自身的失败不是 HTTP 层的失败。

        工具层约定「出错不抛异常，返回 status=error」，接口层必须忠实透传，
        否则前端无法区分「服务器挂了」和「这张图不是图片」。
        """
        body = client.post(
            "/api/v1/tools/call",
            json={
                "tool": "fish_disease_detect",
                "arguments": {"image_path": r"D:\nope\none.jpg"},
            },
        ).json()
        assert body["result"]["status"] == "error"
        assert body["result"]["error_code"] == "IMAGE_NOT_FOUND"

    def test_unknown_tool_name_is_rejected(self, client):
        resp = client.post(
            "/api/v1/tools/call", json={"tool": "rm_rf", "arguments": {}}
        )
        assert resp.status_code == 422  # Literal 已限定取值

    def test_bad_arguments_are_422(self, client):
        resp = client.post(
            "/api/v1/tools/call",
            json={"tool": "water_risk_predict", "arguments": {"totally_wrong": 1}},
        )
        assert resp.status_code == 422
        assert "工具参数不合法" in str(resp.json())

    def test_arguments_default_to_empty(self, client):
        # 不给 arguments 时落到空字典，再由工具自己报参数缺失。
        resp = client.post("/api/v1/tools/call", json={"tool": "water_risk_predict"})
        assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# /ingest/image
# --------------------------------------------------------------------------- #


class TestImageUpload:
    def test_upload_returns_vision_immediately(self, client):
        resp = client.post(
            "/api/v1/ingest/image",
            files={"file": ("fish.png", b"\x89PNG\r\n\x1a\n" + b"0" * 400, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["upload_id"]
        assert body["vision"]["status"] == "ok"
        assert body["content_type"] == "image/png"

    def test_uploaded_id_can_be_used_by_assess(self, client):
        up = client.post(
            "/api/v1/ingest/image",
            files={"file": ("fish.png", b"\x89PNG\r\n\x1a\n" + b"0" * 400, "image/png")},
        ).json()
        body = client.post(
            "/api/v1/assess",
            json={"readings": good_water(), "upload_id": up["upload_id"]},
        ).json()
        assert body["vision"]["status"] == "ok"
        assert body["fusion"]["effective_weights"] == {"water": 0.6, "vision": 0.4}

    def test_stored_filename_is_server_generated(self, client):
        """不使用客户端文件名，路径穿越在源头就不成立。"""
        resp = client.post(
            "/api/v1/ingest/image",
            files={
                "file": (
                    "../../evil.png",
                    b"\x89PNG\r\n\x1a\n" + b"0" * 400,
                    "image/png",
                )
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "evil" not in body["stored_path"]
        assert body["stored_path"].endswith(f"{body['upload_id']}.png")

    def test_unsupported_extension_is_415(self, client):
        resp = client.post(
            "/api/v1/ingest/image", files={"file": ("a.txt", b"hi", "text/plain")}
        )
        assert resp.status_code == 415

    def test_no_extension_is_415(self, client):
        resp = client.post(
            "/api/v1/ingest/image", files={"file": ("noext", b"hi", "application/octet-stream")}
        )
        assert resp.status_code == 415

    def test_empty_upload_is_400(self, client):
        resp = client.post(
            "/api/v1/ingest/image", files={"file": ("a.png", b"", "image/png")}
        )
        assert resp.status_code == 400

    def test_fake_image_is_422_by_magic_number(self, client):
        resp = client.post(
            "/api/v1/ingest/image",
            files={"file": ("fake.jpg", b"definitely not an image", "image/jpeg")},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["error_code"] == "UNSUPPORTED_IMAGE"

    def test_failed_upload_leaves_no_file_behind(self, client, upload_dir):
        resp = client.post(
            "/api/v1/ingest/image",
            files={"file": ("fake.jpg", b"definitely not an image", "image/jpeg")},
        )
        assert resp.status_code == 422
        # 魔数校验失败时必须把已落盘的垃圾文件删掉，否则上传目录会被慢慢塞满。
        assert not list(upload_dir.glob("*"))

    def test_successful_upload_does_leave_a_file(self, client, upload_dir):
        # 上一条的对照组：成功路径必须真的落盘，否则「没留文件」的断言是假通过。
        resp = client.post(
            "/api/v1/ingest/image",
            files={"file": ("ok.png", b"\x89PNG\r\n\x1a\n" + b"0" * 400, "image/png")},
        )
        assert resp.status_code == 200
        assert len(list(upload_dir.glob("*.png"))) == 1

    def test_oversize_upload_is_413(self, upload_dir):
        app = create_app(ApiSettings(upload_dir=upload_dir, max_upload_mb=1))
        with TestClient(app) as c:
            big = b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 10)
            resp = c.post(
                "/api/v1/ingest/image", files={"file": ("big.png", big, "image/png")}
            )
        assert resp.status_code == 413

    def test_missing_file_field_is_422(self, client):
        assert client.post("/api/v1/ingest/image").status_code == 422


# --------------------------------------------------------------------------- #
# /chat 与 /config/*
# --------------------------------------------------------------------------- #


class TestChatAndConfig:
    def test_chat_without_knowledge_is_503(self, client):
        resp = client.post("/api/v1/chat", json={"query": "溶氧低怎么办"})
        assert resp.status_code == 503
        assert "知识库不可用" in str(resp.json())

    def test_chat_query_too_short_is_422(self, client):
        assert client.post("/api/v1/chat", json={"query": "x"}).status_code == 422

    def test_chat_k_out_of_range_is_422(self, client):
        assert (
            client.post("/api/v1/chat", json={"query": "溶氧低怎么办", "k": 99}).status_code
            == 422
        )

    def test_thresholds_expose_decision_basis(self, client):
        body = client.get("/api/v1/config/thresholds").json()
        assert set(body["species"]) == {"鲤科", "罗非鱼", "虾蟹"}
        assert body["weights"]["dissolved_oxygen"] == pytest.approx(0.35)
        assert body["extrapolation_horizon_hours"] == 12.0
        assert body["persistence_cap"] == 1.5
        assert "DO_REDLINE" in body["redlines"]
        assert "dissolved_oxygen" in body["action_thresholds"]
        # 免责声明必须出现在接口里，而不只是文档里。
        assert "不构成用药指令" in body["disclaimer"]

    def test_action_thresholds_are_ordered(self, client):
        body = client.get("/api/v1/config/thresholds").json()
        for pairs in body["action_thresholds"].values():
            values = [p["value"] for p in pairs]
            assert values == sorted(values, reverse=True)
            assert all(p["label"] for p in pairs)

    def test_tools_expose_openai_schemas(self, client):
        tools = client.get("/api/v1/config/tools").json()["tools"]
        names = {t["function"]["name"] for t in tools}
        assert names == {"water_risk_predict", "fish_disease_detect"}
        for t in tools:
            assert t["type"] == "function"
            assert t["function"]["parameters"]["type"] == "object"

    def test_tools_endpoint_matches_registry(self, client):
        from fisheries.tools import get_tool_schemas

        assert client.get("/api/v1/config/tools").json()["tools"] == get_tool_schemas()

    def test_template_catalog_lists_every_builtin(self, client):
        """目录是前端「选一个模板直接试」的唯一数据源，数量与字段都要对得上。"""
        from fisheries.water_templates import all_templates

        body = client.get("/api/v1/config/templates").json()
        assert body["count"] == len(all_templates())
        assert [t["key"] for t in body["templates"]] == [t.key for t in all_templates()]
        assert body["interval_minutes"] == 60
        assert body["columns"]["dissolved_oxygen"] == "溶氧"
        assert "不构成用药指令" in body["disclaimer"] or "复核" in body["disclaimer"]

    def test_every_template_entry_is_self_describing(self, client):
        body = client.get("/api/v1/config/templates").json()
        for item in body["templates"]:
            assert item["name"] and item["scenario"] and item["focus"]
            assert item["expected_risk"] in {"正常", "预警", "高危"}
            assert item["n_points"] >= 3
            assert item["csv_url"].endswith(f"/config/templates/{item['key']}/csv")

    def test_template_csv_downloads_and_reparses(self, client):
        """下载回来的字节必须能被真实的别名层解析——这是"导入就能用"的最低标准。"""
        from fisheries.webui.table import ingest_rows

        catalog = client.get("/api/v1/config/templates").json()
        for item in catalog["templates"]:
            resp = client.get(item["csv_url"])
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/csv")
            assert item["csv_filename"] in resp.headers["content-disposition"]
            rows = list(csv.reader(io.StringIO(resp.text)))
            ingest = ingest_rows(rows[0], rows[1:], source=item["csv_filename"])
            assert ingest.errors == []
            assert ingest.warnings == []
            assert ingest.n_points == item["n_points"]

    def test_template_csv_uses_lf_without_bom(self, client):
        catalog = client.get("/api/v1/config/templates").json()
        for item in catalog["templates"]:
            raw = client.get(item["csv_url"]).content
            assert b"\r\n" not in raw
            assert not raw.startswith(b"\xef\xbb\xbf")

    def test_unknown_template_is_404_with_the_available_keys(self, client):
        resp = client.get("/api/v1/config/templates/no_such_template/csv")
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert "no_such_template" in detail
        assert "dawn_hypoxia" in detail


# --------------------------------------------------------------------------- #
# 并发安全（线程池里的同步端点）
# --------------------------------------------------------------------------- #


class TestConcurrency:
    def test_parallel_assess_results_do_not_mix(self, client):
        """并发请求不能互相串台。

        ``AquacultureAgent.run()`` 会把轨迹挂在实例上（``self.trace``），
        接口层用一把锁串行化；这里验证锁确实生效——每个响应里的
        ``pond_id`` 必须与自己请求的一致，且轨迹都完整。
        """
        import concurrent.futures

        def one(i: int):
            water = crashing_water() if i % 2 else good_water()
            resp = client.post(
                "/api/v1/assess", json={"pond_id": f"C{i}", "readings": water}
            )
            return i, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(one, range(12)))

        for i, body in results:
            assert body["pond_id"] == f"C{i}"
            expected = "高危" if i % 2 else "正常"
            assert body["risk_level"] == expected
            assert body["trace"], "轨迹不应为空"
