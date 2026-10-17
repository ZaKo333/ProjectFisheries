"""临时探针：查看 fastembed 对 bge-small-zh-v1.5 的模型配置与 query 前缀。"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastembed import TextEmbedding  # noqa: E402

for m in TextEmbedding.list_supported_models():
    name = m.get("model") or m.get("model_name") or ""
    if "bge-small-zh" in str(name) or "bge-small" in str(name):
        print({k: v for k, v in m.items() if k != "model_file"})
        print("-" * 60)

cache = str(ROOT / "models" / "fastembed")
model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", cache_dir=cache)
inner = getattr(model, "model", None)
print("model class:", type(inner))
for attr in ("query_prefix", "passage_prefix", "model_name", "dim"):
    print(f"  {attr:<16} = {getattr(inner, attr, '<none>')!r}")
print("has query_embed:", hasattr(model, "query_embed"))
print("has passage_embed:", hasattr(model, "passage_embed"))
print("has embed:", hasattr(model, "embed"))
