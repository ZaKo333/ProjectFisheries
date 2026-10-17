#!/usr/bin/env python
"""构建水产知识库向量库（RAG 索引）。

用法
=====
    # 1) 全量重建（语料改动后跑这个）
    python scripts/build_knowledge_base.py

    # 2) 只看当前状态，不做任何修改
    python scripts/build_knowledge_base.py --check

    # 3) 不加载 90 MB 向量模型，用确定性伪向量（仅用于验证分块与链路，不做真实语义检索）
    python scripts/build_knowledge_base.py --embedder hash

    # 4) 只做分块体检，不写向量库
    python scripts/build_knowledge_base.py --dry-run

产物
=====
    knowledge/chroma/chroma.sqlite3              向量库本体
    knowledge/chroma/fisheries_manifest.json     构建清单（块数、指纹、模型、维度）

清单文件是**一致性守卫**：加载时会逐项校验 collection 名、向量模型、维度、块数，
任何一项对不上就报错而不是悄悄返回一堆不相干的检索结果。

为什么不用 LangChain 的 `RecursiveCharacterTextSplitter`
=========================================================
它按 `\\n` 递归切分，会做两件在本项目里不可接受的事：
    1. **拆散 Markdown 表格**——水质阈值表被切成两半后，
       检索到的"氨氮 ≤0.2 适宜"可能丢掉单位与"适宜/警戒/危险"的分级列。
    2. **不认识标题层级**——切出来的块没有"它属于哪一节"的信息，
       向量化时缺少上下文，短块（如"见上表"）几乎无法被召回。
本项目用 `fisheries.rag.splitter`：标题感知 + 表格整体保留 + 面包屑向量化。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接 `python scripts/build_knowledge_base.py` 而无需先设 PYTHONPATH。
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.rag.embedder import (  # noqa: E402
    DEFAULT_MODEL,
    build_embeddings,
    default_cache_dir,
    describe_embeddings,
    project_root,
)
from fisheries.rag.loader import load_corpus  # noqa: E402
from fisheries.rag.retriever import KnowledgeBase  # noqa: E402
from fisheries.rag.splitter import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_corpus,
)
from fisheries.rag.store import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    store_exists,
)

RULE = "=" * 72

def _print_doc_table(docs) -> None:
    print(f"\n{RULE}\n语料清单（{len(docs)} 篇）\n{RULE}")
    print(f"{'doc_id':<22} {'category':<6} {'disease':<6} {'核对':<6} 标题")
    print("-" * 72)
    for d in docs:
        m = d.meta
        verified = f"{_console.mark(True)}已核对" if m.verified else f"{_console.warn()}待核对"
        print(
            f"{m.doc_id:<22} {m.category:<6} {m.disease or '-':<6} "
            f"{verified:<6} {m.title}"
        )
    # 同一类告警合并成一行。每篇语料都带 verified=True 的自查提示，逐条打印会
    # 刷出十几行黄字，让人以为构建出了问题——其实那只是催作者去核对原文。
    buckets: dict[str, list[str]] = {}
    for d in docs:
        for w in d.warnings:
            buckets.setdefault(w, []).append(d.meta.doc_id)
    for w, ids in buckets.items():
        if len(ids) == 1:
            print(f"  {_console.warn()} [{ids[0]}] {w}")
        else:
            print(f"  {_console.warn()} {len(ids)} 篇语料：{w}")
            print(f"      {'、'.join(ids)}")

def _print_chunk_report(pairs) -> None:
    lens = [len(t) for _m, t in pairs]
    ids = [m.chunk_id for m, _t in pairs]
    dup = len(ids) - len(set(ids))
    n_tables = sum(1 for m, _t in pairs if m.has_table)

    print(f"\n{RULE}\n分块结果\n{RULE}")
    print(f"  块数          : {len(pairs)}")
    print(f"  唯一 chunk_id : {len(set(ids))}"
          + (f"  {_console.mark(False)} 有 {dup} 个重复" if dup
             else f"  {_console.mark(True)} 无重复"))
    if lens:
        print(f"  字数          : min={min(lens)}  max={max(lens)}  "
              f"mean={sum(lens) / len(lens):.1f}")
    print(f"  含表格的块    : {n_tables}")
    print(f"  超出 recommend: {sum(1 for x in lens if x > DEFAULT_MAX_CHARS * 1.6)}"
          "  （表格独占块时允许超出，不截断）")

    # 抽样展示 3 个块的标题链，人工确认层级拼接是否正确。
    print("\n  抽样（标题链 → 前 40 字）：")
    step = max(1, len(pairs) // 3)
    for meta, text in pairs[::step][:3]:
        chain = " > ".join(meta.heading_path) or "(无标题)"
        preview = text.replace("\n", " ")[:40]
        print(f"    [{meta.category}] {chain}")
        print(f"        {preview}…")

def _print_store_report(kb: KnowledgeBase) -> None:
    info = kb.describe()
    print(f"\n{RULE}\n向量库概览\n{RULE}")
    print(f"  后端      : {info['backend']}")
    print(f"  块数      : {info['n_chunks']}")
    print(f"  按类别    : {json.dumps(info['by_category'], ensure_ascii=False)}")
    print(f"  按病种    : {json.dumps(info['by_disease'], ensure_ascii=False)}")
    man = info.get("manifest") or {}
    if man:
        print(f"  清单      : 构建于 {man.get('built_at', '?')}  "
              f"指纹 {man.get('fingerprint', '?')}")
        print(f"              {man.get('n_docs', '?')} 篇 / {man.get('dim', '?')} 维 / "
              f"{man.get('embedder', '?')}")
        print(f"              含表格 {man.get('n_tables', '?')} 块 / "
              f"共 {man.get('total_chars', '?')} 字")

        # 语料是否已过期：比对分块指纹。语料改动后忘了重建，
        # 检索到的仍是旧内容——这是 RAG 最隐蔽的一类故障。
        rows = _current_corpus_rows()
        if rows is not None:
            if rows["fingerprint"] == man.get("fingerprint"):
                print(f"  语料状态  : {_console.mark(True)} 与向量库一致")
            else:
                print(f"  语料状态  : {_console.warn()} 已过期（当前 {rows['n_chunks']} 块 / "
                      f"指纹 {rows['fingerprint']}）")
                print("              → 请重新运行构建（去掉 --check）")

def _current_corpus_rows() -> dict[str, object] | None:
    """算出当前语料的块数与指纹（不加载向量模型，毫秒级）。

    用于 `--check` 判断"语料改了但忘了重建向量库"。
    """
    try:
        from fisheries.rag.store import corpus_fingerprint

        docs = load_corpus(project_root() / "knowledge" / "corpus")
        pairs = chunk_corpus(docs)
        return {"n_chunks": len(pairs), "fingerprint": corpus_fingerprint(pairs)}
    except Exception:  # noqa: BLE001 - 读不到语料就当成没有语料
        return None

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="构建 ProjectFisheries 水产知识库向量库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus-dir", default=None,
                    help="语料目录（默认 knowledge/corpus）")
    ap.add_argument("--persist-dir", default=None,
                    help=f"向量库目录（默认 {DEFAULT_PERSIST_DIR}）")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Chroma collection 名（默认 {DEFAULT_COLLECTION}）")
    ap.add_argument("--embedder", default=None, choices=["fastembed", "hash"],
                    help=f"向量后端（默认 fastembed / {DEFAULT_MODEL}）")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"单块目标字数（默认 {DEFAULT_MAX_CHARS}）")
    ap.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS,
                    help=f"相邻块重叠字数（默认 {DEFAULT_OVERLAP_CHARS}）")
    ap.add_argument("--check", action="store_true",
                    help="只报告当前状态，不构建")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做语料加载与分块体检，不写向量库")
    args = ap.parse_args(argv)

    root = project_root()
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else root / "knowledge" / "corpus"
    persist_dir = Path(args.persist_dir) if args.persist_dir else root / DEFAULT_PERSIST_DIR

    print(RULE)
    print("ProjectFisheries —— 水产知识库构建")
    print(RULE)
    print(f"  语料目录 : {corpus_dir}")
    print(f"  向量库   : {persist_dir}")

    if args.check:
        exists = store_exists(persist_dir)
        print(f"  状态     : {_console.mark(True) + ' 已构建' if exists else _console.mark(False) + ' 未构建'}")
        if exists:
            try:
                kb = KnowledgeBase.open(
                    persist_dir=persist_dir,
                    collection_name=args.collection,
                    embedder_backend=args.embedder,
                )
            except Exception as exc:  # noqa: BLE001 - 维度/模型不匹配等
                print(f"  {_console.mark(False)} 加载失败：{type(exc).__name__}: {exc}")
                return 2
            _print_store_report(kb)
        else:
            print("\n  运行 `python scripts/build_knowledge_base.py` 构建。")
        return 0 if exists else 1

    if not corpus_dir.is_dir():
        print(f"  {_console.mark(False)} 语料目录不存在：{corpus_dir}", file=sys.stderr)
        return 2

    # ---- 1) 加载语料 ----
    t0 = time.perf_counter()
    try:
        docs = load_corpus(corpus_dir)
    except Exception as exc:  # noqa: BLE001 - 语料格式错误要给一行摘要，而不是整片栈
        print(f"  {_console.mark(False)} 语料加载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not docs:
        print(f"  {_console.mark(False)} 语料目录里没有 .md 文件。", file=sys.stderr)
        return 2
    _print_doc_table(docs)

    # ---- 2) 分块 ----
    pairs = chunk_corpus(
        docs, max_chars=args.max_chars, overlap_chars=args.overlap_chars
    )
    _print_chunk_report(pairs)

    if args.dry_run:
        print(f"\n  ⏱ 体检耗时 {time.perf_counter() - t0:.2f} 秒（--dry-run，未写入向量库）")
        return 0

    # ---- 3) 向量化 + 落库 ----
    emb = build_embeddings(args.embedder)
    print(f"\n{RULE}\n向量化\n{RULE}")
    print(f"  模型     : {describe_embeddings(emb)}")
    if args.embedder != "hash":
        print(f"  模型缓存 : {default_cache_dir()}  （已内置项目内，断网可用）")
    if not store_exists(persist_dir):
        print("  提示     : 首次运行需要加载模型，约 1–3 秒；之后的检索无需再加载清单。")

    try:
        kb, stats = KnowledgeBase.build(
            corpus_dir=corpus_dir,
            persist_dir=persist_dir,
            collection_name=args.collection,
            embeddings=emb,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except Exception as exc:  # noqa: BLE001 - 构建失败同样只给一行摘要
        print(f"  {_console.mark(False)} 构建失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    _print_store_report(kb)

    print(f"\n{RULE}\n完成\n{RULE}")
    print(f"  文档 {stats.n_docs} 篇 → 块 {stats.n_chunks} 个"
          f"（含表格 {stats.n_tables} 块，共 {stats.total_chars} 字）")
    print(f"  写入 {stats.n_upserted} 条向量，维度 {stats.dim}，"
          f"耗时 {stats.elapsed_seconds:.2f} 秒（总计 {time.perf_counter() - t0:.2f} 秒）")
    print("\n  自检命令：")
    print("    python scripts/demo_rag.py            # 检索效果演示")
    print("    python -m pytest tests/unit/test_rag.py -q")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
