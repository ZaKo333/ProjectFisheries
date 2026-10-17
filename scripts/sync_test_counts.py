"""把文档里的测试数量同步成 pytest 的真实收集结果。

之前 README、``使用说明.txt``、``docs/技术方案设计.md``、``docs/前端对接说明.md``
里是手写「1222 项」「13 项」这类数字：加几条测试就得挨个改，漏一处文档就自相矛盾，
而且没人知道哪个数字是新的。现在这些数字只有一个来源——``pytest --collect-only``，
本脚本负责把它们写回去。

用法::

    python scripts/sync_test_counts.py            # 按收集结果重写四份文档
    python scripts/sync_test_counts.py --check    # 只比对不落盘（过期则 exit 1）
    python scripts/sync_test_counts.py --json     # 只打印统计

``--check`` 挂在 ``scripts/selftest.py`` 里，所以「文档过期」会被自检直接点出来。

口径说明：这里统计的是**收集数**（``pytest --collect-only`` 的 ``N tests collected``），
与运行环境的可选依赖无关；某条用例因为缺 kagglehub 而在运行时 ``skip``，不影响这里的数字。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
USAGE = PROJECT_ROOT / "使用说明.txt"
DESIGN = PROJECT_ROOT / "docs" / "技术方案设计.md"
FRONTEND = PROJECT_ROOT / "docs" / "前端对接说明.md"

_UNIT_PREFIX = "tests/unit/"
_INTEGRATION_PREFIX = "tests/integration/"


@dataclass(frozen=True)
class Counts:
    """一次收集得到的分层计数。"""

    unit: int
    integration: int
    per_file: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.unit + self.integration

    def as_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "integration": self.integration,
            "total": self.total,
            "per_file": dict(sorted(self.per_file.items())),
        }


def collect() -> Counts:
    """跑一次 ``pytest --collect-only`` 并解析出分层计数。"""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "tests",
    ]
    done = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if done.returncode != 0:
        raise RuntimeError(
            "pytest --collect-only 失败，文档数字无法同步：\n"
            + (done.stdout or "")
            + (done.stderr or "")
        )
    per_file: dict[str, int] = {}
    for line in done.stdout.splitlines():
        node = line.strip()
        if not node.startswith("tests/"):
            continue
        path = node.split("::", 1)[0]
        per_file[path] = per_file.get(path, 0) + 1
    unit = sum(n for path, n in per_file.items() if path.startswith(_UNIT_PREFIX))
    integration = sum(n for path, n in per_file.items() if path.startswith(_INTEGRATION_PREFIX))
    if not unit or not integration:
        raise RuntimeError(
            "收集结果里没有同时找到 tests/unit 与 tests/integration，"
            f"unit={unit} integration={integration}；请确认 pytest 配置未被改动。"
        )
    return Counts(unit=unit, integration=integration, per_file=per_file)


# --------------------------------------------------------------------------- #
# 重写规则：每条都是「整段替换成规范写法」，因此既能改数字，也能顺手统一措辞
# --------------------------------------------------------------------------- #

_GLOBAL_RULES: dict[Path, list[tuple[str, str]]] = {
    README: [
        (r"# 单元测试：\d+ 项(?:（\d+ 通过 \+ \d+ 条件跳过）)?", "# 单元测试：{unit} 项"),
        (r"# 集成测试：\d+ 项", "# 集成测试：{integration} 项"),
        (r"### 11\.1 单元测试（\d+ 项）", "### 11.1 单元测试（{unit} 项）"),
        (r"### 11\.2 端到端集成测试（\d+ 项）", "### 11.2 端到端集成测试（{integration} 项）"),
    ],
    USAGE: [
        (r"会跑一遍单元测试（\d+ 项）", "会跑一遍单元测试（{unit} 项）"),
        (
            r"（\d+ 项，缺权重/缺索引的会自动跳过并说明补法）",
            "（{integration} 项，缺权重/缺索引的会自动跳过并说明补法）",
        ),
        (r"全部单元测试（\d+ 项，约 6 分钟）", "全部单元测试（{unit} 项，约 6 分钟）"),
        (r"端到端集成测试（\d+ 项，10 秒；", "端到端集成测试（{integration} 项，10 秒；"),
    ],
    DESIGN: [
        (
            r"> 测试现状：单元测试 `\d+` 项（.*?）\+ 端到端集成测试 `\d+ passed`（.*?）。",
            "> 测试现状：单元测试 `{unit}` 项 + 端到端集成测试 `{integration}` 项"
            "（分层明细见 README 的 §11.1 / §11.2 两张表；这里的数字由 "
            "`scripts/sync_test_counts.py` 从 `pytest --collect-only` 同步，不手写）。",
        ),
        (
            r"\*\*实施现状（单元测试 \d+ 项(?: = \d+ passed \+ \d+ 条件跳过)?，集成测试 \d+ 项全绿）\*\*：",
            "**实施现状（单元测试 {unit} 项，集成测试 {integration} 项全绿）**：",
        ),
        (
            r"合计 \*\*\d+ 项单元测试\*\* \+ \*\*\d+ 项端到端集成测试\*\*",
            "合计 **{unit} 项单元测试** + **{integration} 项端到端集成测试**",
        ),
        (
            r"合计 \*\*\d+ 项单元测试\*\*（\d+ 通过 \+ \d+ 条件跳过）与 \*\*\d+ 项集成测试\*\*",
            "合计 **{unit} 项单元测试** 与 **{integration} 项集成测试**",
        ),
    ],
    FRONTEND: [
        (r"# 单元测试 \d+ 项(?:（\d+ 通过 \+ \d+ 条件跳过）)?", "# 单元测试 {unit} 项"),
        (r"# 端到端集成测试 \d+ 项", "# 端到端集成测试 {integration} 项"),
    ],
}

#: README / 技术方案里逐文件的表格与树，行内数字按文件名对表
_TABLE_ROW = re.compile(r"^(\| `)(test_[a-z0-9_]+\.py)(` \| )(\d+)( \|)", re.MULTILINE)
_TREE_ROW = re.compile(r"(#\s+✅ )(\d+)( 项)")


def _read(path: Path) -> str:
    # 不用 read_text()：Windows 上会把 CRLF 折成 LF，写回去整份 diff。
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def _per_file_count(counts: Counts, filename: str) -> int | None:
    for prefix in (_UNIT_PREFIX, _INTEGRATION_PREFIX):
        found = counts.per_file.get(prefix + filename)
        if found is not None:
            return found
    return None


def _apply_rules(text: str, rules: list[tuple[str, str]], counts: Counts) -> str:
    for pattern, template in rules:
        replacement = template.format(unit=counts.unit, integration=counts.integration)
        # 模板里没有反斜杠，直接当普通字符串用，省掉 lambda 的闭包绑定问题
        text = re.sub(pattern, replacement, text)
    return text


def _sync_table_rows(text: str, counts: Counts) -> str:
    def repl(match: re.Match[str]) -> str:
        count = _per_file_count(counts, match.group(2))
        if count is None:  # 表里出现了收集不到的旧文件名
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{count}{match.group(5)}"

    return _TABLE_ROW.sub(repl, text)


def _sync_tree_rows(text: str, counts: Counts) -> str:
    out: list[str] = []
    for original in text.split("\n"):
        line = original
        if "✅" in line and _TREE_ROW.search(line):
            if "integration/" in line:
                count = counts.integration
            else:
                name_match = re.search(r"(test_[a-z0-9_]+\.py)", line)
                count = _per_file_count(counts, name_match.group(1)) if name_match else None
            if count is not None:
                line = _TREE_ROW.sub(
                    lambda m, number=count: f"{m.group(1)}{number}{m.group(3)}", line, count=1
                )
        out.append(line)
    return "\n".join(out)


def sync(*, write: bool) -> list[str]:
    """按收集结果重写文档；返回「需要改但没改」或「已改」的描述列表。"""
    counts = collect()
    changes: list[str] = []
    for path, rules in _GLOBAL_RULES.items():
        before = _read(path)
        after = _apply_rules(before, rules, counts)
        if path is README:
            after = _sync_table_rows(after, counts)
        if path is DESIGN:
            after = _sync_tree_rows(after, counts)
        if after == before:
            continue
        for old_line, new_line in zip(
            before.split("\n"), after.split("\n"), strict=False
        ):
            if old_line != new_line:
                changes.append(f"{path.name}: {old_line.strip()[:70]} -> {new_line.strip()[:70]}")
        if write:
            _write(path, after)
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_test_counts",
        description="把文档里的测试数量同步成 pytest --collect-only 的结果",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="只检查文档是否过期，不落盘（过期 exit 1）",
    )
    group.add_argument(
        "--json",
        action="store_true",
        help="打印分层计数后退出，不改任何文件",
    )
    args = parser.parse_args(argv)

    if args.json:
        print(json.dumps(collect().as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    changes = sync(write=not args.check)
    if args.check:
        if changes:
            print("文档里的测试数量已过期：")
            for line in changes:
                print(f"  - {line}")
            print("请运行：python scripts/sync_test_counts.py")
            return 1
        print("文档里的测试数量与 pytest 收集结果一致。")
        return 0

    if not changes:
        print("文档里的测试数量已是最新。")
        return 0
    for line in changes:
        print(f"  - {line}")
    print(f"已更新 {len(_GLOBAL_RULES)} 份文档里的测试数量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
