"""训练鱼种分类器：从图片判断 鲤科 / 罗非鱼 / 虾蟹，产出 ``models/yolov8-species.pt``。

为什么单独一个脚本
------------------
:mod:`fisheries.tools.fish_disease` 用的是按**病害**标注的图集；品种识别需要
按**鱼种**标注的图集，两者的类别体系完全不同，不能共用一份训练产物。

数据从哪来
----------
默认读 :data:`fisheries.datasets` 注册表里的 ``fish_species_images``
（Kaggle: ``sripaadsrinivasan/fish-species-image-data``）。也可以 ``--data-root``
直接指定任意"每个子目录一个类别"的目录树（允许外面多套一层壳目录，
脚本会自己往下钻）。

    python scripts/train_yolo_species.py --check                 # 只看数据够不够、类别怎么映射
    python scripts/train_yolo_species.py                         # 建布局 + 训练 + 导出权重
    python scripts/train_yolo_species.py --epochs 60 --batch 32  # 常规调参

类别映射
--------
底层数据集的类别名五花八门（``Nile Tilapia`` / ``Grass_Carp`` / ``虾`` …），
统一交给 :func:`fisheries.tools.fish_species.map_raw_species` 收敛到三个阈值档案。
**收敛不了的类别默认中止训练**并打印出来：硬塞进某一档，训出来的模型会把
鲶鱼当成鲤科，而鲤科的氨氮警戒线比虾蟹宽一倍——这种错在演示里看不出来，
在生产里是事故。确认那些类别应当丢弃时用 ``--drop-unmapped``。

产物
----
``models/yolov8-species.pt``，类别名就是 ``鲤科``/``罗非鱼``/``虾蟹``。
放好之后 :func:`fisheries.tools.fish_species.fish_species_detect` 立刻生效，
接口与前端不需要重启配置——它们每次调用都重新探测权重。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

# 复用 train_yolo_fish 的预训练权重下载逻辑（镜像前缀、缓存位置都在那边）
import train_yolo_fish as tyf

from fisheries.tools.fish_species import (
    SPECIES_LABELS,
    map_raw_species,
)

CLS_MODEL = tyf.CLS_MODEL
RUNS_DIR = tyf.RUNS_DIR
WEIGHTS_OUT = PROJECT_ROOT / "models" / "yolov8-species.pt"
DATASET_OUT = PROJECT_ROOT / "data" / "derived" / "species_cls"
MANIFEST_OUT = PROJECT_ROOT / "data" / "derived" / "species_dataset_manifest.json"

#: 认作图片的扩展名。
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#: 向下钻透的最大层数（``root/<数据集名>/<类别>/<图片>`` 是两层壳）。
MAX_DEPTH = 4


# --------------------------------------------------------------------------
# 一、数据发现
# --------------------------------------------------------------------------


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def _images_in(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if _is_image(p)]


def default_data_root() -> Path:
    """注册表里的鱼种数据集路径；注册表不认识时退回约定目录。"""
    hit = None
    try:
        from fisheries import datasets as _datasets

        hit = _datasets.default_path("fish_species_images")
    except Exception:  # noqa: BLE001 —— 注册表不可用不该挡住 --data-root
        hit = None
    if hit is not None:
        return Path(hit)
    return PROJECT_ROOT / "data" / "datasets" / "fish-species-image-data"


def discover_class_root(root: Path) -> Path | None:
    """找到「直接子目录就是类别」的那一层。

    数据集压缩包常见多套一层壳（``fish-species-image-data/fish-species-image-data/``），
    所以不能写死层级。判定标准：当前目录的直接子目录里，至少两个各自包含图片。
    """
    if not root.is_dir():
        return None
    current = root
    for _ in range(MAX_DEPTH):
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        informative = [d for d in subdirs if _images_in(d)]
        if len(informative) >= 2:
            return current
        # 只有一层壳时继续往下钻；否则这棵树不像「按类别分目录」
        if len(subdirs) == 1:
            current = subdirs[0]
            continue
        return None
    return None


def collect_classes(class_root: Path) -> dict[str, list[Path]]:
    """返回 {类别目录名: 图片列表}，按图片数降序。"""
    result: dict[str, list[Path]] = {}
    for directory in sorted(class_root.iterdir()):
        if not directory.is_dir():
            continue
        images = _images_in(directory)
        if images:
            result[directory.name] = images
    return dict(sorted(result.items(), key=lambda kv: len(kv[1]), reverse=True))


def map_classes(raw: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """把原始类别收敛到三个品种档案。返回 (已映射, 未映射)。"""
    mapped: dict[str, list[Path]] = {}
    unmapped: dict[str, list[Path]] = {}
    for name, images in raw.items():
        species = map_raw_species(name)
        if species is None:
            unmapped[name] = images
        else:
            mapped.setdefault(species, []).extend(images)
    return mapped, unmapped


def print_report(root: Path, class_root: Path | None,
                 mapped: dict[str, list[Path]], unmapped: dict[str, list[Path]]) -> None:
    print()
    print("=" * 66)
    print("  鱼种数据集体检")
    print("=" * 66)
    print(f"  数据根目录 : {root}{'' if root.is_dir() else '   ✗ 不存在'}")
    print(f"  类别层     : {class_root if class_root else '✗ 没找到「子目录即类别」的层级'}")
    print()

    if mapped:
        print("  映射到品种档案：")
        for species in SPECIES_LABELS:
            images = mapped.get(species, [])
            mark = "✓" if images else "✗"
            print(f"    {mark} {species:6s} {len(images):6d} 张")
    print()

    if unmapped:
        print("  ⚠ 以下类别无法归入三个品种档案（默认中止训练）：")
        for name, images in unmapped.items():
            print(f"      {name}  ({len(images)} 张)")
        print("    确认这些类别应当丢弃时，加 --drop-unmapped 继续。")
        print()

    missing = [s for s in SPECIES_LABELS if not mapped.get(s)]
    if missing:
        print(f"  ⚠ 缺少品种：{', '.join(missing)} —— 模型无法判断这些品种，")
        print("    推理时它们恒为低置信度，上层会退回默认档位（鲤科）。")
        print()


# --------------------------------------------------------------------------
# 二、构建 YOLO-cls 布局
# --------------------------------------------------------------------------


def build_layout(
    mapped: dict[str, list[Path]],
    val_ratio: float,
    seed: int,
    out_dir: Path | None = None,
) -> dict[str, dict[str, int]]:
    """把映射好的图片按品种写进 ``train/`` ``val/`` 两层目录。

    重名文件用序号前缀消歧（不同类别目录下常有同名 ``1.jpg``）。
    ``out_dir`` 默认 :data:`DATASET_OUT`；测试传临时目录以免动到工程产物。
    """
    target_root = out_dir or DATASET_OUT
    rnd = random.Random(seed)
    if target_root.exists():
        shutil.rmtree(target_root)

    counts: dict[str, dict[str, int]] = {}
    for species in SPECIES_LABELS:
        images = list(mapped.get(species, []))
        if not images:
            continue
        rnd.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
        split = {"val": images[:n_val], "train": images[n_val:]}
        if not split["train"]:  # 图片太少时至少留一张给 train
            split["train"] = split["val"][:1]
            split["val"] = split["val"][1:]

        counts[species] = {}
        for part, files in split.items():
            target = target_root / part / species
            target.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(files):
                shutil.copy2(src, target / f"{index:05d}{src.suffix.lower()}")
            counts[species][part] = len(files)
    return counts


def write_manifest(class_root: Path, counts: dict[str, dict[str, int]]) -> None:
    payload = {
        "source": str(class_root),
        "layout": str(DATASET_OUT),
        "labels": list(SPECIES_LABELS),
        "counts": counts,
        "weights": str(WEIGHTS_OUT),
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  ✓ 数据集清单 → {MANIFEST_OUT}")


# --------------------------------------------------------------------------
# 三、训练与验证
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    _console.install()  # ultralytics 导入时会把 stdout 改成 UTF-8，抢回来
    pretrained = tyf.ensure_pretrained(CLS_MODEL)

    print()
    print(f"  分类训练：{CLS_MODEL} → {WEIGHTS_OUT.name}")
    print(f"  数据：{DATASET_OUT}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print()

    model = YOLO(str(pretrained))
    model.train(
        data=str(DATASET_OUT),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name="species",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        patience=args.patience,
        plots=True,
        verbose=True,
    )

    best = RUNS_DIR / "species" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练结束但没有找到 best.pt：{best}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, WEIGHTS_OUT)
    print(f"  ✓ 鱼种权重 → {WEIGHTS_OUT}")
    return WEIGHTS_OUT


def verify(weights: Path) -> bool:
    """拿验证集里的一张图跑一遍，确认权重真的能被 fish_species_detect 用起来。"""
    from fisheries.tools.fish_species import fish_species_detect

    samples = sorted((DATASET_OUT / "val").rglob("*")) if (DATASET_OUT / "val").is_dir() else []
    samples = [p for p in samples if _is_image(p)]
    if not samples:
        print("  ⚠ 验证集为空，跳过推理自检")
        return True

    print()
    print("  推理自检（取验证集若干张）：")
    ok = True
    for image in samples[:5]:
        out = fish_species_detect(str(image), weights_path=str(weights))
        expect = image.parent.name
        got = out.get("species")
        mark = "✓" if got == expect else "·"
        print(
            f"    {mark} {image.parent.name:6s} → {got or '未识别':6s} "
            f"conf={out.get('confidence')} detected={out.get('detected')}"
        )
        if out.get("status") != "ok":
            ok = False
    return ok


# --------------------------------------------------------------------------
# 四、入口
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="训练鱼种分类器，输出 models/yolov8-species.pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=None,
                        help="按类别分目录的原始数据根目录（默认取数据集注册表）")
    parser.add_argument("--check", action="store_true", help="只做数据体检，不训练")
    parser.add_argument("--drop-unmapped", action="store_true",
                        help="丢弃无法归入三个品种的类别（默认遇到就中止）")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda device，如 0；默认自动")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.data_root) if args.data_root else default_data_root()
    class_root = discover_class_root(root)
    raw = collect_classes(class_root) if class_root else {}
    mapped, unmapped = map_classes(raw)

    print_report(root, class_root, mapped, unmapped)

    if args.check:
        return 0 if class_root and mapped else 1

    if not class_root or not raw:
        print("✗ 没找到可训练的数据。请把鱼种数据集放到：")
        print(f"    {default_data_root()}")
        print("  或用 --data-root 指定已有目录。")
        return 1

    if unmapped and not args.drop_unmapped:
        print("✗ 有类别无法归入品种档案，已中止。确认后加 --drop-unmapped。")
        return 1

    if not mapped:
        print("✗ 一个品种都没映射上，检查数据集的目录名。")
        return 1

    counts = build_layout(mapped, args.val_ratio, args.seed)
    print("  布局：")
    for species, parts in counts.items():
        print(f"    {species:6s} train={parts.get('train', 0):6d} val={parts.get('val', 0):6d}")
    write_manifest(class_root, counts)

    weights = train(args)
    verify(weights)
    print()
    print(f"完成。权重：{weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
