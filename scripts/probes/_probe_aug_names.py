"""看 Train_Augmented 的文件名能不能回溯到原始图，以及原图的尺寸分布。

一次性排查用：决定「预增强的 21358 张能不能进训练集」以及 imgsz 该取多少。
独立可跑：不依赖调用方的 cwd 或 PYTHONPATH。
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fisheries import datasets as ds  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def jpeg_or_png_size(path: Path) -> tuple[int, int] | None:
    """不装 Pillow 也能拿到宽高：读文件头。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":
                with path.open("rb") as f2:
                    f2.seek(2)
                    while True:
                        marker = f2.read(2)
                        if len(marker) < 2 or marker[0] != 0xFF:
                            return None
                        code = marker[1]
                        if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
                            continue
                        length = struct.unpack(">H", f2.read(2))[0]
                        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                            data = f2.read(5)
                            h, w = struct.unpack(">HH", data[1:5])
                            return int(w), int(h)
                        f2.seek(length - 2, 1)
    except Exception:  # noqa: BLE001 - 排查脚本，读不出来就算了
        return None
    return None


def main() -> int:
    root = ds.find("fish_disease")
    if root is None:
        print("fish_disease 未找到")
        return 1

    for cls in ("Healthy Fish", "Bacterial gill disease"):
        print("=" * 78)
        print(f"Train_Augmented / {cls}")
        print("=" * 78)
        aug = sorted((root / "Train_Augmented" / cls).iterdir())
        train = sorted((root / "Train" / cls).iterdir())
        print(f"  文件数 {len(aug)}；Train 原始 {len(train)}")
        print("  Train 前 5 个文件名：")
        for p in train[:5]:
            print(f"    {p.name}")
        print("  Train_Augmented 前 12 个文件名：")
        for p in aug[:12]:
            print(f"    {p.name}")
        stems_train = {p.stem for p in train}
        stems_aug = {p.stem for p in aug}
        # 增强图的名字若以原图 stem 开头，就能回溯
        startswith = sum(
            1 for s in stems_aug if any(s.startswith(t) for t in stems_train)
        )
        print(f"  能以某个 Train stem 开头匹配上的：{startswith} / {len(stems_aug)}")

    print()
    print("=" * 78)
    print("原图尺寸分布（Train，四类各取 40 张）")
    print("=" * 78)
    for cls in (
        "Healthy Fish",
        "Bacterial gill disease",
        "Bacterial diseases - Aeromoniasis",
        "Fungal diseases Saprolegniasis",
    ):
        sizes = Counter()
        files = sorted((root / "Train" / cls).iterdir())[:40]
        for p in files:
            wh = jpeg_or_png_size(p)
            sizes[wh] += 1
        top = ", ".join(f"{k[0]}x{k[1]}×{v}" for k, v in sizes.most_common(5) if k)
        print(f"  {cls:<38} {top}")

    print()
    print("=" * 78)
    print("Train_Augmented 尺寸分布（四类各取 40 张）")
    print("=" * 78)
    for cls in (
        "Healthy Fish",
        "Bacterial gill disease",
        "Bacterial diseases - Aeromoniasis",
        "Fungal diseases Saprolegniasis",
    ):
        sizes = Counter()
        files = sorted((root / "Train_Augmented" / cls).iterdir())[:40]
        for p in files:
            sizes[jpeg_or_png_size(p)] += 1
        top = ", ".join(f"{k[0]}x{k[1]}×{v}" for k, v in sizes.most_common(5) if k)
        print(f"  {cls:<38} {top}")

    print()
    print("=" * 78)
    print("Train / Test 扩展名分布（Healthy Fish）")
    print("=" * 78)
    for split in ("Train", "Valid", "Test"):
        c = Counter(p.suffix.lower() for p in (root / split / "Healthy Fish").iterdir())
        print(f"  {split:<8} {dict(c)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
