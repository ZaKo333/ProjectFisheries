"""数据集导入与体检。

工程不再自带数据集，改用「按需导入 + 路径解析」：

    python scripts/fetch_datasets.py --list         看有哪些数据集
    python scripts/fetch_datasets.py --status       看哪些已经在本地
    python scripts/fetch_datasets.py --fetch aquaponics_pond
    python scripts/fetch_datasets.py --fetch all
    python scripts/fetch_datasets.py --where fish_disease

模型权重（不进 git，但 md5 清单进 git）也走同一个入口：

    python scripts/fetch_datasets.py --weights              看权重齐不齐、md5 对不对
    python scripts/fetch_datasets.py --fetch-weights all    从镜像站下载并校验
    python scripts/fetch_datasets.py --write-checksums      重新生成 models/checksums.md5

设计取舍见 ``src/fisheries/datasets.py`` 与 ``src/fisheries/weights.py`` 的模块文档。
这里只负责把它做成命令行。
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
from fisheries import weights as wt  # noqa: E402

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
    print("  下载到本地：python scripts/fetch_datasets.py --fetch <名字>")
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
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权等，给一行摘要而不是整片栈
            print(f"         失败：{type(exc).__name__}: {exc}")
            rc = 1
        else:
            print(f"         完成：{got}")
    return rc


def _mb(size: int) -> str:
    return f"{size / 1048576:.1f} MB"


def cmd_weights(*, verify: bool, as_json: bool) -> int:
    """权重清单与健康度。缺「我们自己训的」权重时返回 1。"""
    rows = wt.status()
    if verify:
        for row in rows:
            ok, detail = wt.verify(row["name"]) if row["exists"] else (False, "文件不存在")
            row["verified"] = ok
            row["verify_detail"] = detail
    if as_json:
        print(json.dumps({"items": rows, "mirror": wt.mirror_base()}, ensure_ascii=False, indent=2))
        return 0 if all(r["exists"] for r in rows if r["pinned"]) else 1

    _rule("=")
    print("模型权重清单（不进 git；md5 清单 models/checksums.md5 进 git）")
    _rule("=")
    for row in rows:
        mark = _console.mark(row["exists"] and row.get("verified", True))
        size = _mb(row["bytes"]) if row["exists"] else "缺失"
        print(f"  {mark} {row['name']:<14} {row['label']}")
        print(f"      {row['file']:<34} {size:>10}")
        if row["exists"] and verify:
            print(f"      {row['verify_detail']}")
    _rule("-")
    mirror = wt.mirror_base()
    print(f"  镜像站：{mirror or '（未配置 FISHERIES_WEIGHTS_MIRROR）'}")
    print(f"  校验表：{wt.CHECKSUMS_FILE.relative_to(wt.PROJECT_ROOT)}")
    print("  下载：python scripts/fetch_datasets.py --fetch-weights all")
    return 0 if all(r["exists"] for r in rows if r["pinned"]) else 1


def cmd_fetch_weights(
    names: list[str], *, all_names: bool, force: bool, url: str | None, base: str | None
) -> int:
    targets = list(wt.WEIGHTS) if all_names else names
    rc = 0
    for name in targets:
        if name not in wt.WEIGHTS:
            print(f"  未注册的权重：{name}")
            rc = 1
            continue
        item = wt.spec(name)
        if not force and wt.find(name) is not None:
            ok, detail = wt.verify(name)
            if ok:
                print(f"  [已有] {name:<14} {item.file}（{detail}）")
                continue
            print(f"  [重下] {name:<14} 已有文件校验不过：{detail}")
        print(f"  [下载] {name:<14} {item.label}（{_mb(item.size) if item.size else '未知大小'}）")
        try:
            got = wt.fetch(name, url=url, base=base, force=force)
        except wt.WeightError as exc:
            lines = str(exc).splitlines() or [str(exc)]
            print(f"         失败：{lines[0]}")
            for extra in lines[1:]:
                print(f"               {extra}")
            rc = 1
        else:
            ok, detail = wt.verify(name)
            print(f"         完成：{got}")
            print(f"         校验：{detail}")
            if not ok:
                rc = 1
    return rc


def _stale_weight_lines() -> list[str]:
    """本地文件与注册表哈希不符的条目，附「应改成什么」的一行。"""
    stale: list[str] = []
    for name in sorted(wt.WEIGHTS):
        item = wt.WEIGHTS[name]
        if not item.pinned or not item.path.is_file():
            continue
        ok, detail = wt.verify(name)
        if ok:
            continue
        stale.append(
            f"    {name}: {detail}\n"
            f"      应更新为 md5={wt.md5_of(item.path)} "
            f"sha256={wt.sha256_of(item.path)} size={item.path.stat().st_size}"
        )
    return stale


def cmd_write_checksums(*, dry_run: bool) -> int:
    # 先核一遍本地权重。重训过的机器上注册表还是旧哈希，此时若照旧写盘，
    # 校验单会和真实文件对不上——那比不写更糟，所以只报新值、不落盘。
    stale = _stale_weight_lines()
    if stale:
        print("  权重与注册表不一致（多半是刚重训过），先改 src/fisheries/weights.py 的")
        print("  WEIGHTS[...] 再重跑这条命令：")
        print("\n".join(stale))
        return 1
    text = wt.render_checksums()
    if dry_run:
        print(text, end="")
        return 0
    target = wt.CHECKSUMS_FILE
    old = target.read_text(encoding="utf-8") if target.is_file() else ""
    if old == text:
        print(f"  未变化：{target.relative_to(wt.PROJECT_ROOT)}")
        return 0
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"  已写入：{target.relative_to(wt.PROJECT_ROOT)}")
    return 0


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
            "  python scripts/fetch_datasets.py --weights\n"
            "  python scripts/fetch_datasets.py --fetch-weights all\n"
            "  python scripts/fetch_datasets.py --write-checksums\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="列出注册表里的全部数据集")
    group.add_argument("--status", action="store_true", help="检查本地有哪些（默认动作）")
    group.add_argument("--where", nargs="+", metavar="NAME", help="只打印路径，便于脚本调用")
    group.add_argument("--fetch", nargs="+", metavar="NAME", help="下载（可用 all）")
    group.add_argument("--weights", action="store_true", help="列出模型权重、缺哪些、md5 对不对")
    group.add_argument(
        "--fetch-weights",
        nargs="+",
        metavar="NAME",
        help="下载模型权重并校验 md5/sha256（可用 all）",
    )
    group.add_argument(
        "--write-checksums",
        action="store_true",
        help="按当前权重重新生成 models/checksums.md5",
    )
    parser.add_argument("--size", action="store_true", help="统计体积（慢，会遍历目录）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出（配合 --status/--weights）")
    parser.add_argument("--verify", action="store_true", help="核对权重 md5 与 sha256（较慢）")
    parser.add_argument("--force", action="store_true", help="权重已存在也重下")
    parser.add_argument("--weight-url", metavar="URL", help="只从这一个地址下权重（跳过镜像推导）")
    parser.add_argument("--weight-mirror", metavar="BASE", help="临时指定权重镜像站根地址")
    parser.add_argument("--dry-run", action="store_true", help="配合 --write-checksums，只打印不落盘")
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list()
    if args.where:
        return cmd_where(args.where, all_names="all" in args.where)
    if args.fetch:
        return cmd_fetch(args.fetch, all_names="all" in args.fetch)
    if args.weights:
        return cmd_weights(verify=args.verify or args.json, as_json=args.json)
    if args.fetch_weights:
        return cmd_fetch_weights(
            args.fetch_weights,
            all_names="all" in args.fetch_weights,
            force=args.force,
            url=args.weight_url,
            base=args.weight_mirror,
        )
    if args.write_checksums:
        return cmd_write_checksums(dry_run=args.dry_run)
    return cmd_status(with_size=args.size, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
