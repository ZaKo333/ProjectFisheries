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
        print("%-42s OK" % name)
    except Exception as exc:
        print("%-42s MISSING (%s)" % (name, type(exc).__name__))

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
        print("%-60s OK" % attr)
    except Exception as exc:
        print("%-60s MISSING (%s)" % (attr, type(exc).__name__))

print()
cache = pathlib.Path.home() / ".cache" / "fastembed"
print("fastembed cache:", cache, "exists=", cache.exists())
if cache.exists():
    for p in sorted(cache.rglob("*")):
        if p.is_dir() and len(p.parts) - len(cache.parts) <= 2:
            print("   ", p.relative_to(cache))
