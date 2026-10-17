import importlib
import pathlib

MODULES = [
    "langchain_chroma",
    "langchain_community.retrievers",
    "langchain_community.vectorstores",
    "langchain_core.vectorstores",
    "langchain_core.documents",
    "langchain_core.embeddings",
    "langchain_core.retrievers",
    "langchain_core.prompts",
    "langchain_core.runnables",
    "langchain_core.output_parsers",
    "langchain_text_splitters",
]
for name in MODULES:
    try:
        importlib.import_module(name)
        print(f"{name:<42} OK")
    except Exception as exc:  # noqa: BLE001, PERF203 - 探针：逐个模块试导入，失败就是结果
        print(f"{name:<42} MISSING ({type(exc).__name__})")

print()
for attr in [
    "langchain_core.vectorstores.InMemoryVectorStore",
    "langchain_core.embeddings.Embeddings",
    "langchain_core.documents.Document",
    "langchain_core.retrievers.BaseRetriever",
    "langchain_text_splitters.MarkdownHeaderTextSplitter",
    "langchain_text_splitters.RecursiveCharacterTextSplitter",
    "langchain_community.retrievers.BM25Retriever",
    "langchain_community.embeddings.FastEmbedEmbeddings",
]:
    mod_name, _, attr_name = attr.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
        getattr(mod, attr_name)
        print(f"{attr:<60} OK")
    except Exception as exc:  # noqa: BLE001 - 同上：缺属性/缺模块都是探针的结果
        print(f"{attr:<60} MISSING ({type(exc).__name__})")

print()
cache = pathlib.Path.home() / ".cache" / "fastembed"
print("fastembed cache:", cache, "exists=", cache.exists())
if cache.exists():
    for p in sorted(cache.rglob("*")):
        if p.is_dir() and len(p.parts) - len(cache.parts) <= 2:
            print("   ", p.relative_to(cache))
