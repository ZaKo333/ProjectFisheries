"""量一下：三份图像集之间、以及各源内部 Train/Valid/Test 之间，到底重叠多少。

一次性排查用。结论会决定 build_yolo_dataset.py 该不该信数据源自带的切分。
独立可跑：不依赖调用方的 cwd 或 PYTHONPATH。
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fisheries import datasets as ds  # noqa: E402

CLASSES = (
    "Healthy Fish",
    "Bacterial gill disease",
    "Bacterial diseases - Aeromoniasis",
    "Fungal diseases Saprolegniasis",
)
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    d = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            d.update(block)
    return d.hexdigest()


def images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES
    )


def scan(name: str) -> dict[tuple[str, str], set[str]]:
    """返回 {(split, class): {md5, ...}}；平铺源 split 记为 'flat'。"""
    root = ds.find(name)
    out: dict[tuple[str, str], set[str]] = {}
    if root is None:
        return out
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    splits = {p.name: p for p in subdirs}
    if {"Train", "Valid"} & set(splits):
        for split_name, folder in splits.items():
            if split_name == "Train_Augmented":
                continue
            for cls in CLASSES:
                key = (split_name, cls)
                out[key] = {md5(p) for p in images(folder / cls)}
    else:
        for cls in CLASSES:
            out[("flat", cls)] = {md5(p) for p in images(splits.get(cls, Path("__none__")))}
    return out


def main() -> int:
    data = {name: scan(name) for name in (
        "fish_disease", "freshwater_fish_disease_asia", "sb_fish_disease"
    )}

    print("=" * 78)
    print("每源每切分的「文件数 / 唯一 md5 数」")
    print("=" * 78)
    for src, table in data.items():
        print(f"\n{src}")
        for (split, cls), digests in sorted(table.items()):
            root = ds.find(src)
            n_files = len(images(root / cls)) if split == "flat" else len(images(root / split / cls))
            print(f"  {split:<18}{cls:<38}{n_files:>5} / {len(digests):>5}")

    print()
    print("=" * 78)
    print("同一源内部：Train 与 Valid / Test 的重叠")
    print("=" * 78)
    for src, table in data.items():
        splits = {s for (s, _) in table}
        if "flat" in splits or not {"Train", "Valid", "Test"} <= splits:
            print(f"\n{src}: 只有 {sorted(splits)}，跳过")
            continue
        print(f"\n{src}")
        for cls in CLASSES:
            tr = table[("Train", cls)]
            va = table[("Valid", cls)]
            te = table[("Test", cls)]
            print(
                f"  {cls:<38} Train {len(tr):>4}  Valid {len(va):>4}  Test {len(te):>4}"
                f"  | T∩V {len(tr & va):>4}  T∩Te {len(tr & te):>4}  V∩Te {len(va & te):>4}"
            )

    print()
    print("=" * 78)
    print("跨源重叠（按类）")
    print("=" * 78)
    names = list(data)
    for cls in CLASSES:
        unions = {}
        for src in names:
            acc: set[str] = set()
            for (split, c), digests in data[src].items():
                if c == cls:
                    acc |= digests
            unions[src] = acc
        a, b, c = (unions[n] for n in names)
        print(f"  {cls:<38} {[len(u) for u in (a, b, c)]}  并集={len(a | b | c)}")
        print(
            f"      fish∩fresh {len(a & b):>5}   fish∩sb {len(a & c):>5}"
            f"   fresh∩sb {len(b & c):>5}   三者交 {len(a & b & c):>5}"
        )

    print()
    print("=" * 78)
    print("合并去重后的净池（4 类）")
    print("=" * 78)
    total = 0
    for cls in CLASSES:
        acc: set[str] = set()
        for src in names:
            for (split, c), digests in data[src].items():
                if c == cls:
                    acc |= digests
        total += len(acc)
        print(f"  {cls:<38} {len(acc):>5}")
    print(f"  {'合计':<38} {total:>5}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
