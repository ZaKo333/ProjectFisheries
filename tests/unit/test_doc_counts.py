"""文档里的测试数量必须能被 pytest 的收集结果复算出来。

以前这些数字是手写的：加几条用例，README 改了、``使用说明.txt`` 忘了，
``docs/技术方案设计.md`` 里还留着上一版的「1221 通过 + 1 条件跳过」。
现在唯一来源是 ``pytest --collect-only``，由 ``scripts/sync_test_counts.py`` 写回去，
本文件负责在单元测试里守门：

* ``sync(write=False)`` 必须返回空列表（文档 = 收集结果）；
* 表格里逐文件项数之和必须等于标题里的总数（防止只改了总数没改明细）；
* 技术方案设计.md 的测试树里每个文件也必须是同一个数；
* 文档里不许再出现「N 通过 + M 条件跳过」这种运行期口径（它随装没装 kagglehub 变）。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "sync_test_counts.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_test_counts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sync_test_counts", module)
    spec.loader.exec_module(module)
    return module


sync_counts = _load_script()


@pytest.fixture(scope="module")
def counts():
    return sync_counts.collect()


@pytest.fixture(scope="module")
def readme() -> str:
    return sync_counts.README.read_bytes().decode("utf-8")


def _rows(text: str, *, prefix: str) -> dict[str, int]:
    pattern = re.compile(r"^\| `(test_[a-z0-9_]+\.py)` \| (\d+) \|", re.MULTILINE)
    rows = {name: int(number) for name, number in pattern.findall(text)}
    if prefix == "integration":
        return {name: n for name, n in rows.items() if name.startswith("test_p0_")}
    return {name: n for name, n in rows.items() if not name.startswith("test_p0_")}


class TestDocsAreInSync:
    def test_documents_match_the_collection(self):
        changes = sync_counts.sync(write=False)
        assert changes == [], (
            "文档里的测试数量与 pytest 收集结果不一致，请运行 "
            "`python scripts/sync_test_counts.py` 后重跑：\n" + "\n".join(changes)
        )

    def test_readme_unit_rows_sum_to_the_heading(self, readme: str):
        heading = re.search(r"### 11\.1 单元测试（(\d+) 项）", readme)
        assert heading, "README 缺少 §11.1 标题"
        rows = _rows(readme, prefix="unit")
        assert sum(rows.values()) == int(heading.group(1)), (
            f"§11.1 明细合计 {sum(rows.values())} 与标题 {heading.group(1)} 不符"
        )

    def test_readme_integration_rows_sum_to_the_heading(self, readme: str):
        heading = re.search(r"### 11\.2 端到端集成测试（(\d+) 项）", readme)
        assert heading, "README 缺少 §11.2 标题"
        rows = _rows(readme, prefix="integration")
        assert sum(rows.values()) == int(heading.group(1)), (
            f"§11.2 明细合计 {sum(rows.values())} 与标题 {heading.group(1)} 不符"
        )

    def test_every_test_module_has_a_documented_row(self, readme: str):
        documented = set(_rows(readme, prefix="unit")) | set(_rows(readme, prefix="integration"))
        collected = {Path(path).name for path in sync_counts.collect().per_file}
        missing = sorted(collected - documented)
        assert not missing, f"这些测试文件没进 README 的表格：{missing}"

    def test_design_doc_tree_numbers_match_per_file_counts(self, counts):
        text = sync_counts.DESIGN.read_bytes().decode("utf-8")
        pattern = re.compile(r"(test_[a-z0-9_]+\.py)\s+#\s+✅ (\d+) 项")
        rows = {name: int(number) for name, number in pattern.findall(text)}
        assert rows, "技术方案设计.md 的测试树里没有解析到任何 ✅ 项数"
        for name, number in rows.items():
            expected = sync_counts._per_file_count(counts, name)
            assert expected is not None, f"测试树里的 {name} 收集不到"
            assert number == expected, f"测试树里 {name} 写的是 {number}，实际 {expected}"


class TestNoStaleWording:
    @pytest.mark.parametrize(
        "path",
        [
            sync_counts.README,
            sync_counts.USAGE,
            sync_counts.DESIGN,
            sync_counts.FRONTEND,
        ],
        ids=lambda p: p.name,
    )
    def test_documents_do_not_hardcode_run_time_counts(self, path: Path):
        text = path.read_bytes().decode("utf-8")
        # 变更记录记的是「当时跑出来的结果」，属于历史，不改写；只查正文。
        body = "\n".join(
            line for line in text.split("\n") if not line.startswith("| v0.1")
        )
        stale = re.findall(r"\d+ (?:通过 \+ \d+ 条件跳过|passed, \d+ skipped)", body)
        assert not stale, (
            f"{path.name} 里还有运行期口径的数字 {stale}：它会随「装没装 kagglehub」变，"
            "文档统一用收集数（pytest --collect-only）"
        )


class TestFileEncoding:
    """这四份文档都是 CRLF，改动必须按字节写回去。

    踩过两次坑：``write_text(text, newline="\\r\\n")`` 会把已有的 CRLF 再转一次，
    得到 ``\\r\\r\\n``，git 里表现为整份文件重写；手工插一行又可能只写了 ``\\n``，
    混进一行裸 LF，既躲过 ``git diff`` 的观感也让按 ``\\r\\n`` 切分的脚本对不齐。
    """

    @pytest.mark.parametrize(
        "path",
        [
            sync_counts.README,
            sync_counts.USAGE,
            sync_counts.DESIGN,
            sync_counts.FRONTEND,
        ],
        ids=lambda p: p.name,
    )
    def test_documents_are_crlf_only(self, path: Path):
        raw = path.read_bytes()
        assert b"\r\r\n" not in raw, f"{path.name} 里有重复转换过的 \\r\\r\\n"
        bare = raw.count(b"\n") - raw.count(b"\r\n")
        assert bare == 0, f"{path.name} 里有 {bare} 行裸 LF（应全部是 CRLF）"

    def test_usage_keeps_its_bom(self):
        # 使用说明.txt 是给记事本看的，BOM 不能丢，丢了中文可能被当成 ANSI。
        assert sync_counts.USAGE.read_bytes().startswith(b"\xef\xbb\xbf")
