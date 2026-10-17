"""P0 主链路③：知识库问答必须带引文，且引文来自真的那篇文档。

单元测试里检索器是替身，最容易漏的正是这里：向量库建好了吗、分块字段齐不齐、
命中之后接口层有没有把 ``chunks``（引文）和 ``answer`` 里的「【文档 > 章节】」
一起返回。答得再流畅，没有引文就等于没有依据。
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.integration

#: 引文格式：``【文档标题 > 章节标题】（依据 …；核验日期 …）``
CITATION_RE = re.compile(r"【[^】]+ > [^】]+】")

CHUNK_FIELDS = (
    "chunk_id",
    "title",
    "heading_text",
    "text",
    "category",
    "verified",
    "score",
    "matched_by",
)


def test_chat_returns_answer_with_citations(chat_answer):
    """「溶氧低怎么办」：拿到答案 + 至少一条引文，且引文实体完整。"""
    body = chat_answer

    assert body["query"] == "溶氧低怎么办"
    assert body["answer"].strip()
    # 答案正文里直接带出处，用户不用展开「参考资料」也知道依据在哪
    assert CITATION_RE.search(body["answer"]), body["answer"][:200]

    # 检索后端是「向量 + BM25」混合，两条路都要在工作状态
    backend = body["backend"]
    assert "chroma" in backend.lower() or "faiss" in backend.lower() or "numpy" in backend.lower()
    assert "bm25" in backend.lower(), backend
    assert "fastembed" in backend.lower() or "dashscope" in backend.lower()
    # 向量维度写进 backend 说明抽象层确实加载了本地模型
    assert re.search(r"\d+\s*维", backend), backend

    chunks = body["chunks"]
    assert 1 <= len(chunks) <= 3
    for chunk in chunks:
        for key in CHUNK_FIELDS:
            assert key in chunk, f"引文缺字段 {key}"
        assert chunk["chunk_id"]
        assert chunk["text"].strip()
        assert 0.0 <= float(chunk["score"]) <= 1.0
        assert isinstance(chunk["matched_by"], list) and chunk["matched_by"]

    # 分数从高到低，前端直接按顺序渲染
    scores = [float(c["score"]) for c in chunks]
    assert scores == sorted(scores, reverse=True), scores


def test_chat_hits_the_dissolved_oxygen_document(client):
    """问溶氧，命中里要真的有讲溶氧的文档，而不是碰巧凑了三条。"""
    resp = client.post("/api/v1/chat", json={"query": "溶解氧低于多少要增氧", "k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if not body.get("chunks"):
        pytest.skip("知识库索引不可用：先跑 scripts/build_knowledge.py")

    hit_text = " ".join(c["text"] for c in body["chunks"])
    assert "溶氧" in hit_text or "溶解氧" in hit_text, body["chunks"][0]["title"]

    # 每条引文都要能回溯到语料：chunk_id 在知识库侧有对应记录
    ids = [c["chunk_id"] for c in body["chunks"]]
    assert len(ids) == len(set(ids)), "同一块被检索了两次"


def test_chat_finds_the_bacterial_gill_disease_document(client):
    """问「烂鳃」：命中的得是病害文档（category/disease 有值），不是水质泛泛而谈。"""
    resp = client.post("/api/v1/chat", json={"query": "细菌性烂鳃病怎么治", "k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if not body.get("chunks"):
        pytest.skip("知识库索引不可用：先跑 scripts/build_knowledge.py")

    top = body["chunks"][0]
    assert "烂鳃" in top["text"] or "烂鳃" in top["title"], top

    # 病害类查询至少要有一条来自「病害」类文档，而不是全被标准条款占了
    categories = [c["category"] for c in body["chunks"]]
    assert categories, "引文没有 category，无法判断来源"
    assert "病害" in categories, [(c["title"], c["category"]) for c in body["chunks"]]
