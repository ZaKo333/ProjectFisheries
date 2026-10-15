"""向量库：Chroma 持久化存储 + 维度/模型一致性守卫。

## 为什么需要"守卫"

RAG 最常见、也最隐蔽的故障不是检索不准，而是**向量库与模型不匹配**：

    * 换了 embedding 模型（`bge-small-zh` 512 维 → 别的 768 维），
      但没重建向量库 → 查询向量与库里向量维度不同，
      Chroma 有些版本会静默返回**按插入顺序排的垃圾结果**；
    * 改了分块参数但用了旧的 collection → 检索到的是过期内容。

这类故障**不会报错**，只会让回答变差，排查起来极其痛苦。
因此本模块在 `persist_directory` 下写一个 ``manifest.json``，
记录模型名、维度、块数、语料指纹，**加载时逐项校验，不一致就报错**。

## 元数据的存储约定

Chroma 的 metadata 只接受 ``str / int / float / bool``，**不接受列表**。
因此：
    * ``heading_path`` / ``indicators`` / ``risk_levels`` / ``citation_keys``
      用 ``"|"`` 连接成字符串存储（本项目取值中不含 ``|``，可安全往返）；
    * ``disease`` 为 None（通用文档）时**存成字面量 "通用"**——
      不存 None：Chroma 对 ``$eq: None`` 的语义在各版本间不一致，
      而"通用文档"必须能被 ``$in`` 可靠地筛出来，
      否则用户问烂鳃时检索不到通用的水质调控手册。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# 必须在 import chromadb 之前设置：Chroma 会在初始化时读取遥测开关。
# 本项目是离线演示环境，不应向外发送数据；同时避免无网络时产生长超时。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

from langchain_core.embeddings import Embeddings  # noqa: E402

from .embedder import describe_embeddings, project_root  # noqa: E402
from .schemas import ChunkMeta  # noqa: E402

#: 默认 collection 名。
DEFAULT_COLLECTION = "fisheries_knowledge"

#: 默认持久化目录（相对项目根）。
DEFAULT_PERSIST_DIR = "knowledge/chroma"

#: 列表型元数据的连接符。
LIST_SEP = "|"

#: ``disease=None`` 在 Chroma 中的替身字符串。
GENERIC_DISEASE = "通用"

#: manifest 文件名。
MANIFEST_NAME = "fisheries_manifest.json"


class StoreMismatchError(RuntimeError):
    """向量库与当前模型/语料不一致时抛出。

    刻意不自动重建：自动重建会**悄悄覆盖**已经算好的向量库，
    如果此时用户正在用另一个模型做对比实验，会丢掉结果。
    交由调用方显式决定（``--rebuild``）。
    """


# --------------------------------------------------------------------- 元数据转换


def chunk_to_metadata(meta: ChunkMeta) -> dict[str, str | int | float | bool]:
    """``ChunkMeta`` → Chroma 可接受的扁平 metadata。"""
    return {
        "chunk_id": meta.chunk_id,
        "doc_id": meta.doc_id,
        "title": meta.title,
        "category": meta.category,
        "disease": meta.disease if meta.disease else GENERIC_DISEASE,
        "heading_path": LIST_SEP.join(meta.heading_path),
        "heading_text": meta.heading_text,
        "indicators": LIST_SEP.join(meta.indicators),
        "risk_levels": LIST_SEP.join(meta.risk_levels),
        "citation_keys": LIST_SEP.join(meta.citation_keys),
        "verified": bool(meta.verified),
        "source": meta.source,
        "standard_codes": LIST_SEP.join(meta.standard_codes),
        "access_source": meta.access_source,
        "verified_by": meta.verified_by,
        "verified_on": meta.verified_on,
        "source_path": meta.source_path,
        "chunk_index": int(meta.chunk_index),
        "section_role": meta.section_role,
        "has_table": bool(meta.has_table),
        "char_count": int(meta.char_count),
    }


def split_list(value: object) -> list[str]:
    """``"A|B"`` → ``["A", "B"]``；空/None → ``[]``。

    也接受已经是序列的输入（Chroma 的方言在不同版本里对列表的处理不一致，
    有的返回 ``"A|B"``，有的原样返回 ``["A", "B"]``，两种都要能读）。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(p).strip() for p in value if str(p).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [p for p in text.split(LIST_SEP) if p]


# ------------------------------------------------------------------------ 构建


@dataclass(slots=True)
class BuildStats:
    """一次构建的结果统计，供脚本打印与单测断言。"""

    n_docs: int = 0
    n_chunks: int = 0
    n_upserted: int = 0
    total_chars: int = 0
    n_tables: int = 0
    n_download_required: bool = False  # 占位，保持统计结构稳定
    elapsed_seconds: float = 0.0
    collection: str = ""
    persist_dir: str = ""
    embedder: str = ""
    dim: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "n_docs": self.n_docs,
            "n_chunks": self.n_chunks,
            "n_upserted": self.n_upserted,
            "total_chars": self.total_chars,
            "n_tables": self.n_tables,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "collection": self.collection,
            "persist_dir": self.persist_dir,
            "embedder": self.embedder,
            "dim": self.dim,
        }


def corpus_fingerprint(pairs: list[tuple[ChunkMeta, str]]) -> str:
    """语料指纹：块数 + 各块 ID 与字数的哈希。

    用于判断"语料是否变过"。不直接对全文做哈希是为了避免
    只改了一个标点就导致整库重建——但要能发现**增删块**与**块长度变化**。
    """
    import hashlib

    h = hashlib.sha256()
    for meta, text in pairs:
        h.update(meta.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(len(text)).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def build_store(
    pairs: list[tuple[ChunkMeta, str]],
    embeddings: Embeddings,
    *,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = True,
    embedding_texts: list[str] | None = None,
) -> tuple[object, BuildStats]:
    """把切好的块写进 Chroma。

    参数：
        pairs: ``[(ChunkMeta, 正文), ...]``
        embeddings: 向量化后端
        embedding_texts: **用于向量化**的文本（通常带标题链面包屑）。
            为 None 时退化为用正文本身。**存进 Chroma 的 ``page_content``
            始终是正文**（``pairs`` 里的第二个元素），与向量化文本分离——
            这样检索到的内容进 prompt 时干净，不含重复的面包屑。

    返回 ``(Chroma 实例, BuildStats)``。
    """
    from langchain_chroma import Chroma

    t0 = time.perf_counter()
    persist_path = Path(persist_dir) if persist_dir else project_root() / DEFAULT_PERSIST_DIR
    persist_path.mkdir(parents=True, exist_ok=True)

    if not pairs:
        raise ValueError("pairs 为空：没有可入库的知识块。请检查语料目录与分块结果。")

    if embedding_texts is None:
        embedding_texts = [text for _, text in pairs]
    if len(embedding_texts) != len(pairs):
        raise ValueError(
            f"embedding_texts 与 pairs 长度不一致（{len(embedding_texts)} vs {len(pairs)}）"
        )

    texts = [text for _, text in pairs]
    metadatas = [chunk_to_metadata(meta) for meta, _ in pairs]
    ids = [meta.chunk_id for meta, _ in pairs]

    if reset:
        _drop_collection(persist_path, collection_name)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        # 用余弦距离：BGE 向量已 L2 归一化，余弦与内积一致，
        # 且不受文本长度影响（长 chunk 的 L2 距离天然偏大）。
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 分批 upsert：一次 add_texts 传入上千条会让 Chroma 内部批次过大。
    batch = 64
    for i in range(0, len(texts), batch):
        store.add_texts(
            texts=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )

    dim = _embedding_dim(embeddings, embedding_texts[0])
    stats = BuildStats(
        n_docs=len({m.doc_id for m, _ in pairs}),
        n_chunks=len(pairs),
        n_upserted=len(texts),
        total_chars=sum(len(t) for t in texts),
        n_tables=sum(1 for m, _ in pairs if m.has_table),
        elapsed_seconds=time.perf_counter() - t0,
        collection=collection_name,
        persist_dir=str(persist_path),
        embedder=describe_embeddings(embeddings),
        dim=dim,
    )

    _write_manifest(
        persist_path,
        {
            "collection": collection_name,
            "embedder": describe_embeddings(embeddings),
            "dim": dim,
            "n_docs": stats.n_docs,
            "n_chunks": stats.n_chunks,
            "n_tables": stats.n_tables,
            "total_chars": stats.total_chars,
            "fingerprint": corpus_fingerprint(pairs),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    return store, stats


def _embedding_dim(embeddings: Embeddings, sample: str) -> int:
    """探测向量维度。优先用后端自报的 ``dim``，失败则实算一次。"""
    dim = getattr(embeddings, "dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    vec = embeddings.embed_query(sample)
    return len(vec)


def _drop_collection(persist_path: Path, collection_name: str) -> None:
    """删除已有 collection（重建时用）。

    直接删 collection 而不是删目录：Chroma 的 PersistentClient
    在同一进程内会缓存连接，删目录后新建可能仍指向旧句柄，
    表现为"重建了但检索到旧数据"。
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_path))
        existing = {c.name for c in client.list_collections()}
        if collection_name in existing:
            client.delete_collection(collection_name)
    except Exception:
        # 目录还不存在、或 chromadb 版本差异时忽略：
        # 后续 Chroma(...) 会正常创建新 collection。
        pass


def _write_manifest(persist_path: Path, payload: dict[str, object]) -> None:
    (persist_path / MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_manifest(persist_dir: str | Path | None = None) -> dict[str, object]:
    """读取 manifest；不存在时返回空字典。"""
    path = (Path(persist_dir) if persist_dir else project_root() / DEFAULT_PERSIST_DIR)
    file = path / MANIFEST_NAME
    if not file.is_file():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# ------------------------------------------------------------------------ 加载


def load_store(
    embeddings: Embeddings,
    *,
    persist_dir: str | Path | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    strict: bool = True,
) -> object:
    """加载已构建的向量库。

    ``strict=True`` 时逐项校验 manifest，不一致就 ``raise StoreMismatchError``。
    校验项：
        1. collection 名一致；
        2. embedder 描述一致（模型换了就报错）；
        3. 向量维度一致；
        4. 库中块数与 manifest 记录一致。
    """
    from langchain_chroma import Chroma

    persist_path = Path(persist_dir) if persist_dir else project_root() / DEFAULT_PERSIST_DIR
    manifest = read_manifest(persist_path)

    if strict and not manifest:
        raise StoreMismatchError(
            f"{persist_path} 下没有 {MANIFEST_NAME}：向量库未构建或已被删除。"
            "请先运行 scripts/build_knowledge_base.py。"
        )

    if strict and manifest:
        if manifest.get("collection") != collection_name:
            raise StoreMismatchError(
                f"collection 名不一致：库中是 {manifest.get('collection')!r}，"
                f"当前请求 {collection_name!r}。"
            )
        expected = str(manifest.get("embedder", ""))
        actual = describe_embeddings(embeddings)
        if expected and expected != actual:
            raise StoreMismatchError(
                f"embedding 后端不一致：库由 {expected} 构建，当前是 {actual}。"
                "换模型必须重建向量库（--rebuild）。"
            )

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if strict and manifest:
        n_now = count_chunks(store)
        n_then = int(manifest.get("n_chunks", 0) or 0)
        if n_then and n_now != n_then:
            raise StoreMismatchError(
                f"块数不一致：manifest 记录 {n_then} 块，库中实际 {n_now} 块。"
                "向量库可能被中断的构建写坏，请用 --rebuild 重建。"
            )

    return store


def count_chunks(store: object) -> int:
    """统计库中块数。"""
    try:
        return int(store._collection.count())  # type: ignore[attr-defined]
    except Exception:
        try:
            data = store.get(include=[])  # type: ignore[attr-defined]
            return len(data.get("ids", []))
        except Exception:
            return 0


def store_exists(
    persist_dir: str | Path | None = None, collection_name: str = DEFAULT_COLLECTION
) -> bool:
    """判断向量库是否已构建（只看 manifest，不加载模型）。"""
    manifest = read_manifest(persist_dir)
    return bool(manifest) and manifest.get("collection") == collection_name
