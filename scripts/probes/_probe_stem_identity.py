"""用「文件名主干」而不是 md5 做身份识别，量出真正的图片张数。

起因：同一张照片以 `X.jpeg` / `X.jpg` / `X.png` 三种扩展名各存一份，
字节不同所以 md5 不同，但它们是**同一张图**。按 md5 去重会漏掉这种重复，
把它们分到训练集和测试集两侧就是泄漏。

一次性排查用。独立可跑：不依赖调用方的 cwd 或 PYTHONPATH。
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
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


def files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)


def scan(name: str) -> dict[tuple[str, str], list[Path]]:
    root = ds.find(name)
    out: dict[tuple[str, str], list[Path]] = {}
    if root is None:
        return out
    subs = {p.name: p for p in root.iterdir() if p.is_dir()}
    if "Train" in subs and "Valid" in subs:
        for split in ("Train", "Valid", "Test"):
            for cls in CLASSES:
                out[(split, cls)] = files(subs[split] / cls)
    else:
        for cls in CLASSES:
            out[("flat", cls)] = files(subs.get(cls, Path("__none__")))
    return out


def main() -> int:
    sources = ("fish_disease", "freshwater_fish_disease_asia", "sb_fish_disease")
    data = {n: scan(n) for n in sources}

    print("=" * 80)
    print("每源每切分：文件数 / 唯一文件名主干数 / 扩展名分布")
    print("=" * 80)
    for src, table in data.items():
        print(f"\n{src}")
        for (split, cls), paths in sorted(table.items()):
            stems = {p.stem for p in paths}
            ext = Counter(p.suffix.lower() for p in paths)
            print(
                f"  {split:<6}{cls:<38} file={len(paths):>4} stems={len(stems):>4}  {dict(ext)}"
            )

    print()
    print("=" * 80)
    print("同一主干多份文件的情况（例：X.jpg + X.png）")
    print("=" * 80)
    for src, table in data.items():
        print(f"\n{src}")
        for (split, cls), paths in sorted(table.items()):
            by_stem: dict[str, list[str]] = defaultdict(list)
            for p in paths:
                by_stem[p.stem].append(p.suffix.lower())
            multi = {s: e for s, e in by_stem.items() if len(e) > 1}
            if multi:
                sample = list(multi.items())[:2]
                print(
                    f"  {split:<6}{cls:<38} 主干 {len(by_stem)}，其中 {len(multi)} 个主干有多份"
                )
                for stem, exts in sample:
                    print(f"        {stem} -> {exts}")

    print()
    print("=" * 80)
    print("各源「唯一主干」集合，以及合并后的净池")
    print("=" * 80)
    per_src_class: dict[tuple[str, str], set[str]] = {}
    for src, table in data.items():
        for (_, cls), paths in table.items():
            per_src_class.setdefault((src, cls), set()).update(p.stem for p in paths)

    grand = 0
    for cls in CLASSES:
        sets = {src: per_src_class.get((src, cls), set()) for src in sources}
        union = set().union(*sets.values()) if sets else set()
        grand += len(union)
        print(f"\n  {cls}")
        for src in sources:
            print(f"      {src:<32} stems={len(sets[src]):>4}")
        print(
            f"      fish∩fresh={len(sets[sources[0]] & sets[sources[1]]):>4}"
            f"  fish∩sb={len(sets[sources[0]] & sets[sources[2]]):>4}"
            f"  **合并净池={len(union)}**"
        )
    print(f"\n  四类净池合计 = {grand}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
