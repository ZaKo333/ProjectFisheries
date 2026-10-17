"""向量化：中文文本 → 稠密向量。

## 模型选型

本项目用 **``BAAI/bge-small-zh-v1.5``**（智源 BGE 系列中文小模型），
通过 **``fastembed``**（ONNX Runtime 后端）加载。选择的理由：

| 候选 | 结论 |
|---|---|
| ``bge-small-zh-v1.5`` + fastembed | ✅ **采用**。512 维，90 MB，**不需要 PyTorch**，CPU 上毫秒级，中文检索效果在同类小模型中第一梯队 |
| ``text-embedding-3-*``（OpenAI） | ❌ 需要 API key 与联网，本项目**无任何 API key 的离线环境必须能跑** |
| ``bge-large-zh`` | ❌ 1.3 GB，对 8 GB 显存的演示机不划算，且本项目语料只有 8 篇、收益不明显 |
| LLM 自带的 embedding（如 AQUA-7B） | ❌ Mistral 系的 decoder 不适合直接当 embedding 用，且要占显存 |

**关键工程考虑**：模型文件已经**预先下载到项目内**
``models/fastembed/``（90.8 MB），因此首次运行**无需联网**。
比赛现场断网、或评审机没有外网时，这一步不会失败。

## 为什么同时提供一个 HashEmbeddings

``HashEmbeddings`` 是**纯 Python 的确定性伪向量**（无模型、无依赖、瞬间完成），
用途只有一个：**让单元测试能在不加载 90 MB 模型的前提下跑完检索链路**。

它**不产生语义向量**——它只保证"同一段文本每次得到同一个向量"，
因此只能测通链路，不能测检索质量。
使用它时检索结果会带上 ``backend`` 标记，**绝不允许被误当成真实检索效果**。
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import threading
from collections.abc import Iterable, Sequence

from langchain_core.embeddings import Embeddings

#: 默认中文向量模型。必须是 fastembed 的 ``model_name`` 取值。
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

#: 模型的输出维度。用于在 store 层校验向量库维度是否与模型匹配
#: （换模型但没重建库，是 RAG 最常见的静默故障）。
DEFAULT_DIM = 512


def project_root() -> pathlib.Path:
    """定位项目根目录（``pyproject.toml`` 所在层）。

    从本文件向上找：``src/fisheries/rag/embedder.py`` → 上溯 4 层。
    不用 ``os.getcwd()``——脚本从别处调用时 cwd 不可靠。
    """
    return pathlib.Path(__file__).resolve().parents[3]


def default_cache_dir() -> pathlib.Path:
    """项目内的模型缓存目录。"""
    return project_root() / "models" / "fastembed"


def _normalize(vec: Sequence[float]) -> list[float]:
    """L2 归一化。

    归一化后，余弦相似度等价于内积，且与 Chroma 的 L2 距离呈单调关系——
    这样无论向量库配的是 cosine 还是 l2，排序结果一致，少一类环境差异。
    """
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return [0.0] * len(vec)
    return [float(x) / norm for x in vec]


class FastEmbedEmbeddings(Embeddings):
    """基于 fastembed(ONNX) 的中文向量模型封装，实现 LangChain ``Embeddings`` 接口。

    懒加载：构造时不加载模型，第一次调用 ``embed_*`` 才加载。
    这样只是 import 本模块、或构造知识库对象做参数校验时，
    不会白白付出一次 90 MB 模型的加载开销。

    ``threads`` 默认为 None（交给 onnxruntime 自行决定）。
    在 CI 或演示机上如果发现占满 CPU，可以显式设为 1。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | pathlib.Path | None = None,
        *,
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = pathlib.Path(cache_dir) if cache_dir else default_cache_dir()
        self.threads = threads
        self.batch_size = batch_size
        self._model: object | None = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    # ---------------------------------------------------------------- 模型加载

    def _ensure_model(self) -> object:
        """线程安全地加载模型（只加载一次）。

        加锁的原因：FastAPI/Streamlit 下多个请求可能同时触发首次加载，
        不加锁会重复加载 90 MB 模型并浪费显存/内存。
        """
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            # 延迟 import：fastembed 会连带加载 onnxruntime，
            # 而上层代码（如只做分块统计的脚本）根本用不到它。
            from fastembed import TextEmbedding

            kwargs: dict[str, object] = {
                "model_name": self.model_name,
                "cache_dir": str(self.cache_dir),
            }
            if self.threads is not None:
                kwargs["threads"] = self.threads

            self._model = TextEmbedding(**kwargs)
            return self._model

    @property
    def dim(self) -> int:
        """向量维度。首次访问会触发模型加载。"""
        if self._dim is None:
            self._dim = DEFAULT_DIM if self.model_name == DEFAULT_MODEL else self._probe_dim()
        return self._dim

    def _probe_dim(self) -> int:
        vecs = list(self._ensure_model().embed(["维度探测"]))  # type: ignore[attr-defined]
        return len(list(vecs[0]))

    # ------------------------------------------------------------ LangChain 接口

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把一批文档向量化。

        ``passage_embed`` 而不是 ``embed``：BGE 系列对"文档"与"查询"
        在概念上是两个入口。本项目用的 bge-small-zh-v1.5 官方说明是
        "query/document 前缀非必需"，因此两者数值上等价，
        但显式区分入口可以在将来换成需要前缀的模型时不改调用方。
        """
        if not texts:
            return []
        model = self._ensure_model()
        out: list[list[float]] = []
        # 分批：一次喂几千条会把内存顶满，尤其是一句话一个 chunk 时。
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = list(model.passage_embed(batch))  # type: ignore[attr-defined]
            out.extend(_normalize(v) for v in vecs)
        return out

    def embed_query(self, text: str) -> list[float]:
        """把检索词向量化。"""
        model = self._ensure_model()
        vec = next(iter(model.query_embed([text])))  # type: ignore[attr-defined]
        return _normalize(vec)


class HashEmbeddings(Embeddings):
    """确定性伪向量：只为让测试不依赖 90 MB 模型。

    🔴 **它不是语义模型。** 它把文本按字符 n-gram 哈希累加到固定维度上，
    因此"用词重叠多的文本"才会相似，**完全不具备同义改写能力**。
    用它跑出来的检索质量没有参考价值，只用于验证：
    分块 → 入库 → 过滤 → 融合排序 → 渲染 prompt 这条链路是否通。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        chars = text.strip()
        if not chars:
            return vec
        for n in (1, 2, 3):
            for i in range(len(chars) - n + 1):
                gram = chars[i : i + n]
                h = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                vec[idx] += sign
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def build_embeddings(
    backend: str | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
    cache_dir: str | pathlib.Path | None = None,
) -> Embeddings:
    """按名字构造 embedding 后端。

    ``backend`` 取值：
        * ``"fastembed"`` / ``None`` —— 真实模型（默认）
        * ``"hash"`` —— 伪向量，**仅供测试**，调用方会被告知
        * 其它取值 → ``ValueError``（不静默降级：
          拼错后端名却拿到 hash 向量，是比报错严重得多的故障）
    """
    import os

    resolved = (backend or os.environ.get("FISHERIES_RAG_EMBEDDER") or "fastembed").lower()
    if resolved == "hash":
        return HashEmbeddings()
    if resolved in {"fastembed", "onnx", "bge"}:
        return FastEmbedEmbeddings(model_name=model_name, cache_dir=cache_dir)
    raise ValueError(
        f"未知的 embedding 后端：{backend!r}。可选 'fastembed'（真实模型）或 'hash'（仅测试）。"
    )


def describe_embeddings(emb: Embeddings) -> str:
    """给出一句人类可读的后端描述，写进检索结果供排查。"""
    if isinstance(emb, FastEmbedEmbeddings):
        return f"fastembed:{emb.model_name}（{emb.dim} 维）"
    if isinstance(emb, HashEmbeddings):
        return f"hash:{emb.dim} 维（伪向量，非语义模型，仅供测试）"
    return type(emb).__name__


def iter_batches(items: Sequence[str], size: int) -> Iterable[list[str]]:
    """把序列切成固定大小的批次（供外部复用）。"""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
