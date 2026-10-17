"""临时探针：验证 loader + splitter 的分块结果是否合理。"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fisheries.rag.loader import load_corpus  # noqa: E402
from fisheries.rag.splitter import build_embedding_text, chunk_corpus  # noqa: E402

docs = load_corpus(ROOT / "knowledge" / "corpus")
print("docs:", len(docs))
for d in docs:
    print(f"  {d.meta.doc_id:<28} {d.meta.category:<6} disease={d.meta.disease:<4} "
          f"verified={d.meta.verified} len={len(d.body)}")

pairs = chunk_corpus(docs)
print()
print("total chunks:", len(pairs))
lens = [m.char_count for m, _ in pairs]
print(f"char_count: min={min(lens)} max={max(lens)} mean={sum(lens) / len(lens):.1f}")
print("tables:", sum(1 for m, _ in pairs if m.has_table))

ids = [m.chunk_id for m, _ in pairs]
print("chunk_id unique:", len(ids) == len(set(ids)))

print()
print("=" * 78)
for m, _t in pairs:
    warn = "" if not m.has_table else " [表]"
    print(f"[{m.doc_id}] {m.heading_text[:42]:<42} {m.char_count:4d}{warn}")

print()
print("=" * 78)
print("前 3 个 chunk 的完整内容（含用于向量化的文本头）:")
for m, t in pairs[:3]:
    print("-" * 70)
    print("chunk_id:", m.chunk_id)
    print("heading_path:", m.heading_path)
    print("citations:", m.citation_keys)
    print("embedding text head:", build_embedding_text(m, t)[:90].replace("\n", " | "))
    print("body:")
    print(t[:400])
