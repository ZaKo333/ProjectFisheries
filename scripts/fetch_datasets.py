"""数据集导入与体检。

工程不再自带数据集，改用「按需导入 + 路径解析」：

    python scripts/fetch_datasets.py --list         看有哪些数据集
    python scripts/fetch_datasets.py --status       看哪些已经在本地
    python scripts/fetch_datasets.py --fetch aquaponics_pond
    python scripts/fetch_datasets.py --fetch all
    python scripts/fetch_datasets.py --where fish_disease

设计取舍见 ``src/fisheries/datasets.py`` 的模块文档。这里只负责把它做成命令行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as ds  # noqa: E402

#: 工程真正会读的数据集。``--status`` 会把「必需但缺失」的标红。
REQUIRED_BY_CODE = ("fish_disease", "aquaponics_pond")


def _rule(char: str = "-", width: int = 78) -> None:
    print(char * width)


def cmd_list() -> int:
    print()
    _rule("=")
    print("  数据集注册表")
    _rule("=")
    print()
    width = max(len(s.name) for s in ds.list_datasets()) + 2
    for spec in ds.list_datasets():
        mark = "★" if spec.name in REQUIRED_BY_CODE else " "
        print(f"  {mark} {spec.name:<{width}}{spec.title}")
        if spec.size_hint:
            print(f"      {'':<{width}}体积     {spec.size_hint}")
        if spec.kaggle_id:
            print(f"      {'':<{width}}kaggle   {spec.kaggle_id}")
        if spec.archive_url:
            print(f"      {'':<{width}}直链     {spec.archive_url}")
        if spec.used_by:
            print(f"      {'':<{width}}用于     {'、'.join(spec.used_by)}")
        if spec.note:
            for line in _wrap(spec.note, 60):
                print(f"      {'':<{width}}         {line}")
        print()
    print("  ★ = 代码当前真正依赖，缺了就跑不起来")
    print()
    print(f"  下载到本地：python scripts/fetch_datasets.py --fetch <名字>")
    print("  换存放位置：set FISHERIES_DATA_ROOT=<目录>（不必移动文件）")
    print()
    return 0


def _wrap(text: str, width: int) -> list[str]:
    """按显示宽度折行。中文按 2 列算，否则排版会歪。"""
    lines: list[str] = []
    cur = ""
    cur_w = 0
    for ch in text:
        w = 2 if ord(ch) > 0x2E7F else 1
        if cur_w + w > width and cur:
            lines.append(cur)
            cur, cur_w = "", 0
        cur += ch
        cur_w += w
    if cur:
        lines.append(cur)
    return lines or [""]


def cmd_status(*, with_size: bool, as_json: bool) -> int:
    items = ds.status_all(with_size=with_size)
    if as_json:
        payload = [
            {
                "name": it.spec.name,
                "title": it.spec.title,
                "available": it.available,
                "path": str(it.path) if it.path else None,
                "source": it.source,
                "size_bytes": it.size_bytes,
                "kaggle_id": it.spec.kaggle_id,
                "archive_url": it.spec.archive_url,
                "fetchable": it.spec.fetchable,
            }
            for it in items
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print()
    _rule("=")
    print("  数据集状态")
    _rule("=")
    print(f"  解析顺序：环境变量 {ds.DATA_ROOT_ENV} → 工程 data/datasets → "
          f"历史目录 数据集/ → kagglehub 缓存")
    print()
    width = max(len(it.spec.name) for it in items) + 2
    n_available = 0
    missing_required: list[str] = []
    for it in items:
        if it.available:
            n_available += 1
        elif it.spec.name in REQUIRED_BY_CODE:
            missing_required.append(it.spec.name)
        # 用 console.mark 而不是写死 ✔/✘：GBK 控制台下这两个字符编不出来，
        # 而"可用 / 缺失"是**承载信息**的，不能退化成 `?`。
        flag = _console.mark(it.available)
        print(f"  {flag} {it.spec.name:<{width}}{'可用' if it.available else '缺失'}")
        if it.available:
            print(f"      {'':<{width}}路径  {it.path}")
            print(f"      {'':<{width}}来源  {it.source}")
            if with_size:
                print(f"      {'':<{width}}体积  {it.size_text}")
        else:
            print(f"      {'':<{width}}下载  {it.spec.download_command()}")
    print()
    print(f"  可用 {n_available}/{len(items)}")

    total = 0
    if with_size:
        total = sum(it.size_bytes or 0 for it in items if it.available)
        print(f"  已导入总体积 {total / (1024 ** 3):,.2f} GB")
    print()

    if missing_required:
        print("  代码依赖但缺失的数据集：")
        for name in missing_required:
            spec = ds.DATASETS[name]
            print(f"    · {name} —— {spec.title}")
            print(f"      {spec.download_command()}")
        print()
        return 1
    return 0


def cmd_where(names: list[str], *, all_names: bool) -> int:
    if all_names:
        names = list(ds.DATASETS)
    rc = 0
    for name in names:
        path = ds.find(name)
        if path is None:
            print(f"# {name}: 缺失", file=sys.stderr)
            rc = 1
        else:
            print(path)
    return rc


def cmd_fetch(names: list[str], *, all_names: bool) -> int:
    if all_names:
        names = list(ds.DATASETS)
    rc = 0
    for name in names:
        if name not in ds.DATASETS:
            print(f"  未注册的数据集：{name}")
            rc = 1
            continue
        spec = ds.DATASETS[name]
        existing = ds.find(name)
        if existing is not None:
            print(f"  [已有] {name:<28} {existing}")
            continue
        print(f"  [下载] {name:<28} {spec.kaggle_id or spec.archive_url or '（无可下载来源）'}")
        try:
            got = ds.fetch(name)
        except ds.DatasetNotFoundError as exc:
            print(f"         失败：{exc}")
            rc = 1
        except Exception as exc:  # 网络/鉴权等，给一行摘要而不是整片栈
            print(f"         失败：{type(exc).__name__}: {exc}")
            rc = 1
        else:
            print(f"         完成：{got}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_datasets",
        description="数据集导入与体检（工程不再自带数据集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/fetch_datasets.py --list\n"
            "  python scripts/fetch_datasets.py --status\n"
            "  python scripts/fetch_datasets.py --fetch fish_disease\n"
            "  python scripts/fetch_datasets.py --fetch all\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="列出注册表里的全部数据集")
    group.add_argument("--status", action="store_true", help="检查本地有哪些（默认动作）")
    group.add_argument("--where", nargs="+", metavar="NAME", help="只打印路径，便于脚本调用")
    group.add_argument("--fetch", nargs="+", metavar="NAME", help="下载（可用 all）")
    parser.add_argument("--size", action="store_true", help="统计体积（慢，会遍历目录）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（配合 --status）")
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.where:
        return cmd_where(args.where, all_names="all" in args.where)
    if args.fetch:
        return cmd_fetch(args.fetch, all_names="all" in args.fetch)
    return cmd_status(with_size=args.size, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
