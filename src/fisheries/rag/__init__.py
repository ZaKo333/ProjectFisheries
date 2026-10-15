"""RAG 层 —— 水产知识库的加载、分块、向量化与检索。

模块职责
==========
    schemas.py     数据契约（DocMeta / ChunkMeta / RetrievedChunk / RetrievalQuery）
    loader.py      语料发现 + YAML front-matter 解析
    splitter.py    标题感知分块（表格整体保留，不切散）
    embedder.py    中文向量模型封装（fastembed + BAAI/bge-small-zh-v1.5）
    store.py       Chroma 持久化与一致性守卫
    retriever.py   混合检索（稠密 + BM25）与按 Agent 状态定向检索

为什么是"混合检索"而不是纯向量检索
====================================
纯向量检索在这个场景里有三个稳定失败的案例：

    1. **数字**：``"氨氮 0.5"`` 与 ``"氨氮 1.5"`` 的向量几乎一样，
       但一个是警戒、一个是红线。BM25 能靠字面把正确的块顶上来。
    2. **编号**：``"GB 11607"``、``"NY 5051"`` 这类标准号是精确 token，
       向量模型会把它们糊成一团。
    3. **专有名词**：``"亚硝酸盐"`` 被 jieba 切成 ``"亚硝"/"酸盐"`` 后，
       查询"亚硝酸盐"反而匹配不上标准原文里的写法。
       解决方式是给 ``DOMAIN_WORDS`` 里约 55 个水产术语注册自定义词典。

因此 :func:`retriever.KnowledgeBase.search` 同时跑稠密与稀疏两路，
再用 RRF（Reciprocal Rank Fusion）在**排名**层面融合，而不是在分数层面。
理由：余弦相似度（0–1）与 BM25 分数（无上界）量纲不可比，
归一化再相加需要为每个语料库调参；RRF 只用排名，换语料不用重调。

检索不是"总是查一遍"
======================
:func:`retriever.plan_queries` 按融合结论**定向**决定查什么：

    - 「应急处置」只在**高危或命中红线**时检索——预案里全是"立即/紧急"，
      正常状态下混进 prompt 会让模型把日常巡检写成紧急告警。
    - 「病害处置」用硬过滤锁定病种，避免"烂鳃的建议"出现在"水霉"的告警里。
    - 「投料管理」在**有病害时也要查**——"停料几天、怎么恢复"只写在投料规范里。

快速上手
==========
    >>> from fisheries.rag import open_knowledge_base
    >>> kb = open_knowledge_base(auto_build=False)      # 已有向量库时加载
    >>> ctx = kb.retrieve_for_agent(water=water, vision=vision, fusion=fusion)
    >>> print(ctx.render()[:200])

首次使用（或语料有改动）时用::

    python scripts/build_knowledge_base.py

知识库的语料可信度分级见 ``knowledge/README.md``。
"""

from .embedder import (
    DEFAULT_DIM,
    DEFAULT_MODEL,
    FastEmbedEmbeddings,
    HashEmbeddings,
    build_embeddings,
    default_cache_dir,
    describe_embeddings,
    project_root,
)
from .loader import (
    CorpusError,
    LoadedDoc,
    discover_corpus,
    load_corpus,
    load_document,
    stable_hash,
)
from .retriever import (
    AgentKnowledgeState,
    KnowledgeBase,
    KnowledgeContext,
    open_knowledge_base,
    plan_queries,
    state_from_agent,
)
from .schemas import (
    ChunkMeta,
    DocMeta,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from .splitter import build_chunk_meta, build_embedding_text, chunk_corpus, chunk_document
from .store import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    BuildStats,
    StoreMismatchError,
    build_store,
    load_store,
    read_manifest,
    store_exists,
)

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_DIM",
    "DEFAULT_MODEL",
    "DEFAULT_PERSIST_DIR",
    "AgentKnowledgeState",
    "BuildStats",
    "ChunkMeta",
    "CorpusError",
    "DocMeta",
    "FastEmbedEmbeddings",
    "HashEmbeddings",
    "KnowledgeBase",
    "KnowledgeContext",
    "LoadedDoc",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedChunk",
    "StoreMismatchError",
    "build_chunk_meta",
    "build_embedding_text",
    "build_embeddings",
    "build_store",
    "chunk_corpus",
    "chunk_document",
    "default_cache_dir",
    "describe_embeddings",
    "discover_corpus",
    "load_corpus",
    "load_document",
    "load_store",
    "open_knowledge_base",
    "plan_queries",
    "project_root",
    "read_manifest",
    "stable_hash",
    "state_from_agent",
    "store_exists",
]
