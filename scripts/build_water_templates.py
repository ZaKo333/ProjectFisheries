#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 `templates/` 下的水质模板文件。

用法
=====
    # 1) 生成 / 覆盖全部模板 CSV 与 templates/README.md
    python scripts/build_water_templates.py

    # 2) 只校验磁盘上的文件与代码里的定义是否一致（CI / 提交前跑）
    python scripts/build_water_templates.py --check

    # 3) 只打印模板目录表，不写任何文件
    python scripts/build_water_templates.py --list

产物
=====
    templates/<key>.csv        每个模板一份，可直接上传 / 粘贴 / 用 Excel 打开
    templates/README.md        清单与用法（**整份生成**，不要手工编辑）

为什么模板文件要生成而不是手写
================================
模板有 12 个、每个 24 行，手写必然出现"代码里的定义"和"仓库里的 CSV"不一致，
而现场演示恰恰会拿这份 CSV 去上传。生成 + `--check` 让两者不可能漂移：
改数值只改 `src/fisheries/water_templates.py`，重跑本脚本即可。

`--check` 的判定标准是**逐字节相等**。所以模板 CSV 固定用 `\\n` 换行、
不带 BOM —— 这与仓库根的 `.gitattributes`（``* -text``）是配套的：
关掉换行符转换之后，谁在什么系统上 checkout 都得到同一份字节。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 `python scripts/build_water_templates.py` 而无需先设 PYTHONPATH。
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.water_templates import (  # noqa: E402
    TEMPLATES_DIR,
    all_templates,
    catalog_markdown,
    catalog_rows,
    render_csv,
)

_README = "README.md"


def _render_all(root: Path) -> dict[Path, str]:
    """算出"此刻应该写在磁盘上的全部内容"。写盘与校验共用这一份。"""
    payload: dict[Path, str] = {}
    for template in all_templates():
        payload[root / template.csv_filename] = render_csv(template)
    payload[root / _README] = catalog_markdown()
    return payload


def _write(payload: dict[Path, str]) -> int:
    written = 0
    for path, text in payload.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" —— 不做换行翻译，让文件里的 \n 原样落盘。
        path.write_text(text, encoding="utf-8", newline="")
        print(f"  [写出] {path.name:<26} {len(text.encode('utf-8')):>6} B")
        written += 1
    return written


def _check(payload: dict[Path, str]) -> int:
    """逐字节比对。返回不一致的条数。"""
    bad = 0
    for path, text in payload.items():
        expected = text.encode("utf-8")
        if not path.exists():
            print(f"  [缺失] {path.name}（应存在，{len(expected)} B）")
            bad += 1
            continue
        actual = path.read_bytes()
        if actual == expected:
            print(f"  [一致] {path.name:<26} {len(actual):>6} B")
            continue
        bad += 1
        note = f"磁盘 {len(actual)} B / 期望 {len(expected)} B"
        if actual == expected.replace(b"\n", b"\r\n"):
            note += "；只有换行符不同（应为 LF，实际是 CRLF）"
        elif actual.replace(b"\r\n", b"\n") == expected:
            note += "；只有换行符不同（应为 LF，实际含 CRLF）"
        print(f"  [不一致] {path.name}：{note}")
    return bad


def _list() -> None:
    rows = catalog_rows()
    print(f"{'key':<22} {'品种':<7} {'预期等级':<9} {'点数':<5} 名称")
    for row in rows:
        print(
            f"{row['key']:<22} {row['species']:<7} {row['expected_risk']:<9} "
            f"{row['n_points']:<5} {row['name']}"
        )
    print()
    print(f"共 {len(rows)} 个模板，目录：{TEMPLATES_DIR}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成（或校验）templates/ 下的水质模板文件。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验磁盘文件与代码定义是否逐字节一致，不写盘；不一致时退出码为 1",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只打印模板目录表，不做任何校验与写盘",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help=f"输出目录，默认 {TEMPLATES_DIR}",
    )
    args = parser.parse_args(argv)

    if args.list:
        _list()
        return 0

    root = Path(args.dir).resolve() if args.dir else TEMPLATES_DIR
    payload = _render_all(root)

    if args.check:
        print(f"校验水质模板（{len(payload)} 个文件）……")
        bad = _check(payload)
        print()
        if bad:
            print(
                f"✘ 有 {bad} 个文件与 src/fisheries/water_templates.py 不一致。\n"
                f"  改模板请改代码，然后重跑：python scripts/build_water_templates.py"
            )
            return 1
        print(f"✔ 全部 {len(payload)} 个文件与代码定义一致")
        return 0

    print(f"生成水质模板 → {root}")
    written = _write(payload)
    print()
    print(f"✔ 已写出 {written} 个文件（{len(all_templates())} 个模板 + 1 份 README）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
