"""文档加载：Markdown 语料 → 带元数据的原始文档。

本模块负责 RAG 流水线的第一段：

    文件系统 → (front-matter 解析) → LoadedDoc(元数据 + 正文)

**为什么不用 LangChain 的 ``DirectoryLoader`` / ``UnstructuredMarkdownLoader``：**

1. 本项目语料的元数据在 **YAML front-matter** 里，而 LangChain 的
   Markdown 加载器会把 front-matter 当作正文一起送进向量库——
   结果是"doc_id: disease_gill"这类字段污染了每一个 chunk 的语义；
2. ``UnstructuredMarkdownLoader`` 依赖 ``unstructured`` 包，
   它会重新推断结构，**表格经常被拆散**，而阈值表格拆散就废了；
3. 语料格式是我们自己定的，自己解析只要 40 行，且行为完全可预测。

代价是：**只支持本项目约定的语料格式**。这对知识库来说是合理的——
语料是受控的，不是用户上传的任意文档。
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

from .schemas import DocMeta

#: front-matter 的分隔符。要求文件**以 ``---`` 开头**。
_FRONT_MATTER_FENCE = "---"

#: 正文中引用锚点的正则：``[GB 11607-1989 §4.1]``、``[NY 5051 表1]``。
#: 只匹配方括号内含标准编号样式的内容，避免把普通方括号说明也抓进来。
CITATION_PATTERN = r"\[((?:GB|NY|SC|GB/T|SC/T|DB)\s*[\w./-]+(?:\s+[^\]]{0,40})?)\]"


@dataclass(slots=True)
class LoadedDoc:
    """一个已解析的语料文档：元数据 + 原始正文（未分块）。"""

    meta: DocMeta
    body: str
    path: pathlib.Path
    warnings: list[str] = field(default_factory=list)


class CorpusError(RuntimeError):
    """语料格式不符合约定时抛出。

    这里刻意用异常而不是静默跳过：语料是我们自己写的**受控资产**，
    格式错误一定是需要立刻修的问题，静默跳过只会让知识库悄悄缺内容。
    """


def _split_front_matter(raw: str, path: pathlib.Path) -> tuple[dict[str, Any], str]:
    """把文件切成 (front-matter 字典, 正文)。

    只接受两种写法，不猜：
        * 以 ``---`` 开头，到下一个 ``---`` 行为止是 YAML；
        * 完全不写 front-matter（此时元数据全空，由调用方报错）。
    """
    text = raw.lstrip("\ufeff")
    lines = text.splitlines()

    # 跳过开头的空行再判断，避免首行空行导致 front-matter 被当成正文。
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1

    if start >= len(lines) or lines[start].strip() != _FRONT_MATTER_FENCE:
        raise CorpusError(
            f"{path.name} 缺少 YAML front-matter（文件必须以 '---' 开头）。"
            "知识库语料是受控资产，不接受无元数据的文件。"
        )

    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == _FRONT_MATTER_FENCE:
            end = i
            break
    if end is None:
        raise CorpusError(f"{path.name} 的 front-matter 没有闭合的 '---'。")

    raw_yaml = "\n".join(lines[start + 1 : end])
    try:
        parsed = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        raise CorpusError(f"{path.name} 的 front-matter 不是合法 YAML：{exc}") from exc

    if not isinstance(parsed, dict):
        raise CorpusError(f"{path.name} 的 front-matter 必须是键值映射。")

    body = "\n".join(lines[end + 1 :])
    return parsed, body


def parse_doc_meta(raw_meta: dict[str, Any], path: pathlib.Path) -> DocMeta:
    """把 front-matter 字典校验成 :class:`DocMeta`。

    ``source_path`` 由加载器填入，语料文件里不必写——
    路径是运行时事实，写死在文件里会在挪目录时变成谎言。
    """
    payload = {k: v for k, v in raw_meta.items() if k != "source_path"}
    payload["source_path"] = path.as_posix()
    try:
        return DocMeta.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError，避免顶层导入
        raise CorpusError(f"{path.name} 的 front-matter 校验失败：{exc}") from exc


def load_document(path: pathlib.Path | str) -> LoadedDoc:
    """加载单个 Markdown 语料文件。"""
    p = pathlib.Path(path)
    if not p.is_file():
        raise CorpusError(f"语料文件不存在：{p}")

    raw = p.read_text(encoding="utf-8")
    raw_meta, body = _split_front_matter(raw, p)
    meta = parse_doc_meta(raw_meta, p)

    warnings: list[str] = []
    if not body.strip():
        warnings.append("正文为空")
    if meta.verified:
        warnings.append(
            "verified=True：请确认确实已逐条核对标准原文，否则应改回 false"
        )

    return LoadedDoc(meta=meta, body=body, path=p, warnings=warnings)


def discover_corpus(corpus_dir: pathlib.Path | str) -> list[pathlib.Path]:
    """列出语料目录下所有 ``.md`` 文件，按文件名排序。

    排序的意义：让 chunk 的生成顺序、进而 ``chunk_index`` **完全确定**。
    否则不同文件系统的遍历顺序会让同一份语料产生不同的 chunk_id，
    增量更新时会误判为"全部新增"。
    """
    d = pathlib.Path(corpus_dir)
    if not d.is_dir():
        raise CorpusError(f"语料目录不存在：{d}")

    # 只取 corpus 顶层，不递归进 raw/（raw 放标准原文 PDF，格式不同）。
    files = sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".md"),
        key=lambda p: p.name,
    )
    if not files:
        raise CorpusError(
            f"{d} 下没有任何 .md 语料文件。请先放入种子语料，"
            "或运行 scripts/build_knowledge_base.py 前确认 corpus 目录。"
        )
    return files


def load_corpus(corpus_dir: pathlib.Path | str) -> list[LoadedDoc]:
    """加载整个语料目录。

    遇到不合法文件**直接抛异常并指出是哪个文件**——
    宁可构建失败，也不要构建出一个默默缺了一块的知识库。
    """
    docs = [load_document(p) for p in discover_corpus(corpus_dir)]

    seen: dict[str, pathlib.Path] = {}
    for doc in docs:
        dup = seen.get(doc.meta.doc_id)
        if dup is not None:
            raise CorpusError(
                f"doc_id 重复：'{doc.meta.doc_id}' 同时出现在 "
                f"{dup.name} 与 {doc.path.name}。doc_id 必须全局唯一。"
            )
        seen[doc.meta.doc_id] = doc.path

    return docs


def stable_hash(*parts: object, length: int = 12) -> str:
    """生成稳定短哈希。

    用途是 chunk_id：由 ``源文件路径 + 标题路径 + 序号`` 决定，
    **跨运行、跨机器一致**。不用 Python 内置 ``hash()``——
    它对 str 加了随机盐，进程一重启就变，会让向量库的增量更新彻底失效。
    """
    joined = "\u0001".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]
