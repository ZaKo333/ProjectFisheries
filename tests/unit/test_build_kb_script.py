"""``scripts/build_knowledge_base.py --rebuild`` 的行为约定。

为什么单独钉这条：``src/fisheries/rag/store.py`` 的报错文案一直在说「请用
``--rebuild`` 重建」，但脚本里**以前根本没有这个参数**——用户照着提示敲，
只会拿到 ``unrecognized arguments: --rebuild``。这个文件把「参数存在」和
「真的会删旧库」两件事都钉住，避免文案与实现再次各说各话。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "build_knowledge_base.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
        timeout=300,
        check=False,
    )


def _tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # 语料是受控资产：没有 YAML front-matter 的文件会被 loader 直接拒掉
    # （CorpusError），所以这里必须写一份最小的合规头部。
    (corpus / "oxygen.md").write_text(
        "---\n"
        "doc_id: test_oxygen\n"
        "title: 溶解氧与氨氮的应急处理（测试语料）\n"
        "category: 水质\n"
        "indicators: [dissolved_oxygen, ammonia_n]\n"
        "risk_levels: [正常, 预警, 高危]\n"
        "source: 单元测试构造\n"
        "verified: false\n"
        "---\n\n"
        "# 溶解氧\n\n溶解氧低于 3 mg/L 时要开增氧机，并减少投喂。\n\n"
        "# 氨氮\n\n氨氮超过 0.5 mg/L 要换水。\n",
        encoding="utf-8",
    )
    return corpus


class TestRebuildFlag:
    def test_help_lists_the_flag(self) -> None:
        done = _run("--help")

        assert done.returncode == 0
        assert "--rebuild" in done.stdout

    def test_rebuild_drops_the_previous_store(self, tmp_path: Path) -> None:
        corpus = _tiny_corpus(tmp_path)
        store = tmp_path / "chroma"
        store.mkdir()
        stale = store / "stale.txt"
        stale.write_text("old store", encoding="utf-8")

        done = _run(
            "--corpus-dir",
            str(corpus),
            "--persist-dir",
            str(store),
            "--embedder",
            "hash",
            "--rebuild",
        )

        assert done.returncode == 0, done.stdout + done.stderr
        assert not stale.exists(), "旧库没被删掉，--rebuild 等于没生效"
        assert (store / "fisheries_manifest.json").exists()

    def test_check_mode_never_deletes_anything(self, tmp_path: Path) -> None:
        """`--check` 是只读体检，跟着 `--rebuild` 一起敲也不许删东西。"""
        store = tmp_path / "chroma"
        store.mkdir()
        stale = store / "stale.txt"
        stale.write_text("old store", encoding="utf-8")

        _run("--persist-dir", str(store), "--check", "--rebuild")

        assert stale.exists()
