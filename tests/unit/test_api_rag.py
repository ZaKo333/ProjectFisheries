"""接口层 + RAG 联动的测试。

为什么单独一个模块
------------------
``test_api.py`` 整个模块关闭了 RAG（为了不去加载 90 MB 的向量模型），
所以那里测不到「知识库可用时接口长什么样」。这里换一条路：用
``HashEmbeddings``（确定性伪向量，毫秒级、无模型下载）在临时目录里
从一份**合成小语料**建一个真向量库，再把 ``open_knowledge_base`` 换成
返回它。

这样测的是**接口层怎么暴露知识库结果**，而不是检索质量——检索质量
（混合检索、RRF、元章节排除、列表原子性）已经在 ``test_rag.py`` 里覆盖。
在接口测试里重建一套检索断言，只会让同一件事有两处会失败的副本。
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from fisheries.agent.orchestrator import KNOWLEDGE_DISABLE_ENV
from fisheries.api.app import create_app
from fisheries.api.deps import ApiSettings
from fisheries.rag.embedder import HashEmbeddings
from fisheries.rag.retriever import KnowledgeBase

# --------------------------------------------------------------------------- #
# 合成语料（三篇，覆盖「有病种的病害语料」与「无病种的其他语料」两种情况）
# --------------------------------------------------------------------------- #

WATER_DOC = """---
doc_id: t_water
title: 水质阈值测试文档
category: 水质
disease: null
indicators: [溶解氧, 氨氮]
risk_levels: [预警, 高危]
source: 测试语料
verified: false
---

## 1. 溶解氧分级

溶解氧高于 5 mg/L 为适宜区间。

溶解氧在 3 至 5 mg/L 之间为警戒区间，需要开启增氧机。

溶解氧低于 3 mg/L 为危险区间，低于 1.5 mg/L 存在泛塘风险。

## 2. 氨氮分级

氨氮不高于 0.2 mg/L 为适宜区间。

氨氮高于 0.5 mg/L 进入危险区间，需要立即换水并停止投饵。
"""

GILL_DOC = """---
doc_id: t_gill
title: 烂鳃病处置测试文档
category: 病害
disease: 烂鳃
indicators: [溶解氧]
risk_levels: [高危]
source: 测试语料
verified: false
---

## 1. 症状识别

病鱼鳃丝肿胀、颜色发白，鳃盖内表皮充血，常浮于水面呼吸。

## 2. 处置方案

### 2.1 第一步：立即措施

立即停止投饵，开启增氧机保持溶氧高于 5 mg/L。

### 2.2 第二步：水体消毒

按说明书剂量使用国标渔药进行水体消毒，并送样做病原确证。
"""

FEED_DOC = """---
doc_id: t_feed
title: 投料管理测试文档
category: 投料
disease: null
indicators: [溶解氧, 氨氮]
risk_levels: [预警]
source: 测试语料
verified: false
---

## 1. 投喂量必须下调的情况

溶氧偏低时投喂量应下调 30% 至 50%。

氨氮高于 0.5 mg/L 或亚硝酸盐高于 0.3 mg/L 时应停止投料。

## 2. 投喂频次

水温 25 至 30 摄氏度时每日投喂 3 次，水温低于 20 摄氏度时减至 2 次。
"""


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_kb(tmp_path_factory):
    """在临时目录里从合成语料建一个真向量库（HashEmbeddings，无模型下载）。"""
    corpus = tmp_path_factory.mktemp("rag_corpus")
    (corpus / "01_water.md").write_text(WATER_DOC, encoding="utf-8")
    (corpus / "02_gill.md").write_text(GILL_DOC, encoding="utf-8")
    (corpus / "03_feed.md").write_text(FEED_DOC, encoding="utf-8")

    # persist_dir 必须显式给绝对临时路径：默认值是相对仓库的
    # `knowledge/chroma`，一旦落到那里就会把真实向量库覆盖掉。
    persist = tmp_path_factory.mktemp("rag_chroma")
    kb, stats = KnowledgeBase.build(
        corpus_dir=corpus, persist_dir=persist, embeddings=HashEmbeddings()
    )
    assert stats.n_docs == 3
    assert stats.n_chunks > 0
    return kb


@pytest.fixture
def upload_dir(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def rag_client(tiny_kb, upload_dir, monkeypatch):
    """知识库可用的客户端。

    ``deps._prepare_knowledge()`` 里是「函数内 import」，所以替换
    ``fisheries.rag.retriever.open_knowledge_base`` 就能生效，
    不必去改 ``deps`` 的实现。
    """
    monkeypatch.delenv(KNOWLEDGE_DISABLE_ENV, raising=False)
    monkeypatch.setattr(
        "fisheries.rag.retriever.open_knowledge_base", lambda **_: tiny_kb
    )
    app = create_app(ApiSettings(upload_dir=upload_dir))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def broken_kb_client(upload_dir, monkeypatch):
    """知识库打开失败时的客户端。"""

    def _boom(**_):
        raise RuntimeError("向量库目录不存在")

    monkeypatch.delenv(KNOWLEDGE_DISABLE_ENV, raising=False)
    monkeypatch.setattr("fisheries.rag.retriever.open_knowledge_base", _boom)
    app = create_app(ApiSettings(upload_dir=upload_dir))
    with TestClient(app) as c:
        yield c


def good_water(n: int = 24) -> list[dict]:
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


# --------------------------------------------------------------------------- #
# /health —— 知识库状态
# --------------------------------------------------------------------------- #


class TestHealthWithKnowledge:
    def test_reports_ready(self, rag_client):
        body = rag_client.get("/api/v1/health").json()
        assert body["knowledge"]["status"] == "ready"
        assert body["knowledge"]["n_chunks"] > 0
        assert "hash" in body["knowledge"]["backend"]

    def test_ready_knowledge_still_degraded_if_vision_is_mock(self, rag_client, monkeypatch):
        """知识库就绪不代表整体健康：图像模型没训练同样是降级。

        「没训练」这个前提要显式造出来——过去它靠的是「本机没装 ultralytics」，
        一旦训练出权重这条就失真了。
        """
        from fisheries.tools import fish_disease

        monkeypatch.setattr(fish_disease, "resolve_weights", lambda *a, **k: None)
        assert rag_client.get("/api/v1/health").json()["status"] == "degraded"

    def test_ready_and_vision_ready_is_healthy(self, rag_client, monkeypatch, tmp_path):
        """反过来也要成立：知识库就绪 + 图像模型就绪 = 不再误报降级。"""
        import importlib.util as importlib_util

        from fisheries.tools import fish_disease

        weights = tmp_path / "yolov8-fish.pt"
        weights.write_bytes(b"stub")
        real_find_spec = importlib_util.find_spec
        monkeypatch.setattr(fish_disease, "resolve_weights", lambda *a, **k: str(weights))
        # 只把 ultralytics 伪装成已安装，其它模块照旧走真实查询——
        # 否则请求链路上任何一次惰性 import 都会被这个假 find_spec 打乱。
        monkeypatch.setattr(
            importlib_util,
            "find_spec",
            lambda name, *a, **k: object()
            if name == "ultralytics"
            else real_find_spec(name, *a, **k),
        )
        body = rag_client.get("/api/v1/health").json()
        assert "yolov8-cls" in body["tools"]["fish_disease_detect"]
        assert body["status"] != "degraded"

    def test_reports_unavailable_with_hint(self, broken_kb_client):
        body = broken_kb_client.get("/api/v1/health").json()
        assert body["knowledge"]["status"] == "unavailable"
        assert "向量库目录不存在" in body["knowledge"]["detail"]
        assert "build_knowledge_base" in body["knowledge"]["hint"]
        assert any("知识库不可用" in n for n in body["notes"])


# --------------------------------------------------------------------------- #
# /assess —— 建议带上了规范依据
# --------------------------------------------------------------------------- #


class TestAssessWithKnowledge:
    def test_readings_with_timestamps_do_not_break_serialization(self, rag_client):
        """带 timestamp 的读数是**真实客户端的默认用法**，必须能跑通。

        回归事故：``WaterQualityPoint.timestamp`` 是 ``datetime``，
        接口层用默认 python 模式 ``model_dump()`` 把它原样交给调度层，
        而调度提示词要用 ``json.dumps`` 把上下文塞进 ``##CTX##`` 段 →
        ``TypeError: Object of type datetime is not JSON serializable``，
        ``/assess`` 直接 500。

        之前的用例全部不带 timestamp，所以整条链路都没暴露这个问题；
        而 timestamp 恰恰是工具用来推断采样间隔的输入，现实中不会省。
        """
        readings = [
            {
                "timestamp": f"2026-10-14T{i:02d}:00:00+08:00",
                "water_temp": 28.0 + i * 0.06,
                "ph": 8.05,
                "dissolved_oxygen": round(6.2 - i * 0.15, 3),
                "ammonia_n": round(0.15 + i * 0.02, 3),
                "nitrite_n": 0.05,
            }
            for i in range(24)
        ]
        resp = rag_client.post("/api/v1/assess", json={"readings": readings})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["risk_level"] in {"正常", "预警", "高危"}
        # 间隔应当是从 timestamp 推断出来的 60 分钟，而不是回落到入参默认值。
        assert body["water"]["data_quality"]["interval_source"] == "timestamp"
        assert body["water"]["data_quality"]["interval_minutes"] == 60.0

    def test_references_are_returned(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        ).json()
        assert body["references"], "高危场景必须检索到并附上规范依据"
        assert all(r.startswith("《") for r in body["references"])

    def test_knowledge_block_is_the_prompt_payload(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        ).json()
        # 返回注入 prompt 的原文，是为了让前端能证明「建议有依据」而不是自说自话。
        assert len(body["knowledge_block"]) > 100
        assert "知识" in body["knowledge_block"] or "##" in body["knowledge_block"]

    def test_knowledge_summary_counts_by_purpose(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        ).json()
        summary = body["knowledge_summary"]
        assert summary["n_chunks"] > 0
        assert summary["degraded"] is False
        assert isinstance(summary["purposes"], dict)

    def test_unverified_corpus_warning_is_surfaced(self, rag_client):
        """语料全是 verified:false 时，必须把「未核对原文」的警告传给调用方。"""
        body = rag_client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        ).json()
        assert any("未与标准原文核对" in w for w in body["knowledge_summary"]["warnings"])
        assert any("未核对标准原文" in r for r in body["references"])

    def test_knowledge_block_can_be_suppressed_but_references_stay(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess",
            json={"readings": crashing_water(), "include_knowledge_block": False},
        ).json()
        assert body["knowledge_block"] == ""
        # 关掉长文本只是省流量，引用清单是结论的一部分，不该一起消失。
        assert body["references"]

    def test_normal_scenario_also_gets_feeding_guidance(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess", json={"readings": good_water()}
        ).json()
        assert body["references"]

    def test_assess_survives_broken_knowledge_base(self, broken_kb_client):
        """知识库挂了不能连告警一起挂掉——这是降级而非失败。"""
        resp = broken_kb_client.post(
            "/api/v1/assess", json={"readings": crashing_water()}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["risk_level"] == "高危"  # 核心结论不受影响
        assert body["references"] == []
        assert body["knowledge_block"] == ""
        assert any("知识库" in r for r in body["degraded_reasons"])

    def test_use_knowledge_false_skips_retrieval(self, rag_client):
        body = rag_client.post(
            "/api/v1/assess", json={"readings": crashing_water(), "use_knowledge": False}
        ).json()
        assert body["references"] == []
        assert body["risk_level"] == "高危"


# --------------------------------------------------------------------------- #
# /chat
# --------------------------------------------------------------------------- #


class TestChat:
    def test_returns_chunks(self, rag_client):
        resp = rag_client.post("/api/v1/chat", json={"query": "溶氧低于多少要开增氧机"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks"]
        for c in body["chunks"]:
            assert set(c) == {
                "chunk_id", "title", "heading_text", "text", "category",
                "disease", "verified", "score",
                "dense_score", "sparse_score", "matched_by", "citation_keys",
            }

    def test_non_disease_chunk_has_null_disease(self, rag_client):
        """这是真实踩过的坑：把 disease 声明成 str 会让水质类语料序列化时 500。"""
        body = rag_client.post(
            "/api/v1/chat", json={"query": "溶解氧分级", "category": "水质"}
        ).json()
        assert body["chunks"]
        assert all(c["disease"] is None for c in body["chunks"])

    def test_disease_chunk_carries_disease_label(self, rag_client):
        body = rag_client.post(
            "/api/v1/chat", json={"query": "烂鳃怎么处理", "disease": "烂鳃"}
        ).json()
        assert body["chunks"]
        assert any(c["disease"] == "烂鳃" for c in body["chunks"])

    def test_answer_is_marked_as_raw_retrieval_without_llm(self, rag_client):
        """没配大模型时不能编一段像模型写的回答。"""
        body = rag_client.post("/api/v1/chat", json={"query": "氨氮高了怎么办"}).json()
        assert body["answer_is_raw_retrieval"] is True
        assert "【" in body["answer"]  # 拼的是检索原文

    def test_answer_with_llm_flag_is_ignored_under_mock(self, rag_client):
        body = rag_client.post(
            "/api/v1/chat", json={"query": "氨氮高了怎么办", "answer_with_llm": True}
        ).json()
        assert body["answer_is_raw_retrieval"] is True

    def test_k_limits_chunk_count(self, rag_client):
        body = rag_client.post(
            "/api/v1/chat", json={"query": "溶氧 氨氮 投喂", "k": 1}
        ).json()
        assert len(body["chunks"]) == 1

    def test_category_filter_is_applied(self, rag_client):
        body = rag_client.post(
            "/api/v1/chat", json={"query": "阈值 处置 投喂", "category": "投料", "k": 5}
        ).json()
        assert body["chunks"]
        assert {c["category"] for c in body["chunks"]} == {"投料"}

    def test_warnings_are_returned(self, rag_client):
        body = rag_client.post("/api/v1/chat", json={"query": "溶解氧分级"}).json()
        assert any("未与标准原文核对" in w for w in body["warnings"])

    def test_verified_flag_is_exposed(self, rag_client):
        body = rag_client.post("/api/v1/chat", json={"query": "溶解氧分级"}).json()
        assert all(c["verified"] is False for c in body["chunks"])

    def test_backend_is_described(self, rag_client):
        body = rag_client.post("/api/v1/chat", json={"query": "溶解氧分级"}).json()
        assert "hash" in body["backend"]

    def test_chat_with_broken_kb_is_503(self, broken_kb_client):
        resp = broken_kb_client.post("/api/v1/chat", json={"query": "溶氧低怎么办"})
        assert resp.status_code == 503

    def test_empty_result_is_not_an_error(self, rag_client):
        """检索不到不该报错——「没有依据」本身就是一个有效答案。"""
        resp = rag_client.post(
            "/api/v1/chat", json={"query": "宇宙飞船轨道高度计算", "k": 3}
        )
        assert resp.status_code == 200
        body = resp.json()
        if not body["chunks"]:
            assert "未检索到" in body["answer"]


# --------------------------------------------------------------------------- #
# 核验标记的渲染口径（接口引用列表 / 返回文本 与报告层共用同一处实现）
# --------------------------------------------------------------------------- #


def _chunk(**overrides):
    from fisheries.rag.schemas import RetrievedChunk

    base = dict(
        chunk_id="c1",
        text="溶氧低于 3 mg/L 必须立即增氧。",
        heading_text="溶解氧分级",
        title="水质阈值测试文档",
        category="水质",
    )
    base.update(overrides)
    return RetrievedChunk(**base)


class TestProvenanceMark:
    def test_verified_chunk_names_the_standard(self):
        mark = _chunk(
            verified=True,
            standard_codes=["GB 11607-1989"],
            verified_on="2026-10-15",
        ).provenance_mark()
        assert "依据 GB 11607-1989" in mark
        assert "核验日期 2026-10-15" in mark

    def test_verified_without_codes_does_not_invent_one(self):
        """核验过但没登记编号时，宁可说"已核对"也不能编一个标准号出来。"""
        mark = _chunk(verified=True).provenance_mark()
        assert mark == "（已核对标准原文）"

    def test_verified_without_date_does_not_invent_one(self):
        mark = _chunk(verified=True, standard_codes=["GB 11607-1989"]).provenance_mark()
        assert mark == "（已核对标准原文）"

    def test_unverified_chunk_keeps_the_warning(self):
        mark = _chunk(verified=False, standard_codes=["GB 11607-1989"]).provenance_mark()
        assert mark == "（未核对标准原文）"

    def test_reference_line_shape(self):
        line = _chunk(
            verified=True,
            standard_codes=["GB 11607-1989"],
            verified_on="2026-10-15",
        ).reference_line()
        assert line.startswith("《水质阈值测试文档》> 溶解氧分级（依据 GB 11607-1989")

    def test_api_answer_block_shares_the_same_mark(self):
        from fisheries.api.routes import _provenance_block

        block = _provenance_block(
            _chunk(
                verified=True,
                standard_codes=["GB 11607-1989"],
                verified_on="2026-10-15",
            )
        )
        assert block.startswith("【水质阈值测试文档 > 溶解氧分级】（依据 GB 11607-1989")
        assert block.endswith("溶氧低于 3 mg/L 必须立即增氧。")

    def test_api_answer_block_warns_on_unverified(self):
        from fisheries.api.routes import _provenance_block

        assert "未核对标准原文" in _provenance_block(_chunk(verified=False))


class TestLlmFailureFallback:
    """大模型挂掉时的退路必须和「没配模型」那条路给出同一种文本块。

    模型不可用时调用方恰恰最需要看清数值出处——那是它唯一能自己判断
    可信度的线索。所以这条退路不能是手写的 ``f"【…】\\n{text}"``。
    """

    @staticmethod
    def _state(advisor):
        from types import SimpleNamespace

        return SimpleNamespace(agent=SimpleNamespace(advisor=advisor))

    @staticmethod
    def _result(chunks):
        from types import SimpleNamespace

        return SimpleNamespace(chunks=chunks)

    def test_fallback_keeps_the_provenance_mark(self):
        from fisheries.api.routes import _answer_from_chunks

        class Exploding:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("模型服务连不上")

        chunks = [
            _chunk(
                verified=True,
                standard_codes=["GB 11607-1989"],
                verified_on="2026-10-15",
            )
        ]
        text, is_raw = _answer_from_chunks(
            self._state(Exploding()), _payload(), self._result(chunks)
        )
        assert is_raw is True
        assert "依据 GB 11607-1989" in text
        assert "核验日期 2026-10-15" in text

    def test_fallback_names_the_exception_type_but_not_its_message(self):
        """只写异常类型名：LLM 客户端的异常消息里常带 URL 甚至 Authorization。"""
        from fisheries.api.routes import _answer_from_chunks

        class Exploding:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("https://api.example.com?key=sk-secret")

        text, _ = _answer_from_chunks(
            self._state(Exploding()), _payload(), self._result([_chunk()])
        )
        assert "RuntimeError" in text
        assert "sk-secret" not in text


def _payload():
    from fisheries.api.schemas import ChatRequest

    return ChatRequest(query="溶氧低了怎么办", answer_with_llm=True)

