"""把本机几份鱼病图像集拼成 YOLO 能直接吃的 4 分类训练集。

    python scripts/build_yolo_dataset.py             # 构建分类布局
    python scripts/build_yolo_dataset.py --report    # 只统计，不写文件
    python scripts/build_yolo_dataset.py --with-det  # 额外产出弱监督检测布局
    python scripts/build_yolo_dataset.py --force     # 已存在也重建

## 为什么需要这个脚本

上游给我们的从来不是「一个干净的数据集」，而是三份互相重叠、切分方式不同、
命名不一致的图像目录。实测（见 ``scripts/probes/_probe_image_overlap.py`` 与
``scripts/probes/_probe_stem_identity.py``）有三个必须处理的事实：

**一、三份数据其实是同一批照片。** 逐类比对文件名主干：

    Healthy Fish                      fish_disease 226 张，SB-FishDisease 158 张全在 fish_disease 里
    Bacterial gill disease            fish_disease 228 张，SB 36 张全被包含
    Bacterial diseases - Aeromoniasis fish_disease 241 张，SB 40 张全被包含
    Fungal diseases Saprolegniasis    fish_disease 236 张，SB 33 张全被包含

``freshwater_fish_disease_asia`` 的图与 ``fish_disease`` 逐 md5 相同（250/250 全重叠）。
**所以全库真正不同的照片只有 931 张**，SB 一份都没多贡献。

**二、上游自带的 Train/Valid/Test 切分是漏的。** ``fish_disease`` 里
``Train∩Valid = 50``、``Train∩Test = 50``（每类）——验证集和测试集的图**同时也在训练集里**。
直接沿用这套切分会得到虚高的指标，因此本脚本**丢弃上游切分、自己重新分层切分**。

**三、同一张照片以多种扩展名各存一份。** 例如 ``Healthy Fish (1).jpeg`` /
``.jpg`` / ``.png`` 三份，字节不同所以 md5 不同。**身份识别因此按「文件名主干」
而不是 md5**，否则同一张图会一份进训练集、一份进测试集。

还有一条限制：数据集里**没有任何检测框标注**（三份数据全盘扫过，没有 .txt/.xml/.json
标注文件），所以能训的是**分类**。``--with-det`` 产出的检测布局是每张图一个全图框的
弱监督口径，只为了让产品链路拿到 ``(class, confidence, box)`` 与 mAP@0.5 数字，
**不代表病灶被框住了**。

## 4 类映射的出处

不是拍脑袋定的。Ultralytics 平台那份同源数据的清单里带了官方中英文类名对照
（``freshwater-fish-disease-aquaculture-in-south-asia.ndjson``）：

    {"0":"红体病","1":"细菌性出血病","2":"烂鳃病","3":"水霉病",
     "4":"健康","5":"寄生虫病","6":"白尾病"}

与目录名的对应是稳定的。用户要的四类是其中的 ``1/2/3/4``——**细菌性出血取的是
``Aeromoniasis`` 那一支**，不是 ``Bacterial Red disease``（那是「红体病」）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as ds  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 默认输出目录。放在 ``data/derived`` 而不是 ``data/datasets``——
#: 后者是「导入的上游数据」的搜索位置，这里是我们自己派生的训练集。
DEFAULT_OUT = PROJECT_ROOT / "data" / "derived" / "fish_disease_4cls"
DEFAULT_DET_OUT = PROJECT_ROOT / "data" / "derived" / "fish_disease_4cls_det"


# --------------------------------------------------------------------------
# 类别定义
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassSpec:
    """一个目标类别：英文键、中文名、以及它在各数据源里的目录名。"""

    key: str
    zh: str
    source_dir: str
    note: str = ""


#: 训练用的 4 个类别。**顺序即 YOLO 的类索引，改动必须重新训练。**
CLASSES: tuple[ClassSpec, ...] = (
    ClassSpec("healthy", "健康鱼", "Healthy Fish"),
    ClassSpec("gill_rot", "烂鳃", "Bacterial gill disease"),
    ClassSpec("bacterial_hemorrhage", "细菌性出血", "Bacterial diseases - Aeromoniasis"),
    ClassSpec("saprolegniasis", "水霉", "Fungal diseases Saprolegniasis"),
)

#: 被舍弃的类别与理由（写进报告，免得以后有人问「怎么少了三类」）。
DROPPED: dict[str, str] = {
    "Bacterial Red disease": "红体病，用户指定的 4 类里没有它",
    "Parasitic diseases": "寄生虫病，属虫害而非细菌/真菌病害",
    "Viral diseases White tail disease": "白尾病，病毒性，且样本最少",
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: 同一主干多份文件时，优先取哪个扩展名（越靠前越优先），再比文件大小。
_EXT_PREFERENCE = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: 数据源扫描顺序。**先扫到的源占住同名主干**，所以把结构最规范的放最前。
SOURCES: tuple[str, ...] = (
    "fish_disease",
    "freshwater_fish_disease_asia",
    "sb_fish_disease",
)

#: 上游切分目录名 → 规范切分名。只在「记录这张图上游属于哪个集合」时用，
#: 最终切分一律由本脚本重新分层，见 :func:`split_pool`。
_UPSTREAM_SPLITS: dict[str, str] = {}
for _canonical, _aliases in (
    ("train", ("Train", "train")),
    ("val", ("Valid", "Val", "valid", "val")),
    ("test", ("Test", "test")),
):
    for _alias in _aliases:
        _UPSTREAM_SPLITS[_alias] = _canonical


# --------------------------------------------------------------------------
# 素材收集：身份 = (类别, 文件名主干)
# --------------------------------------------------------------------------


@dataclass
class Photo:
    """一张**不同的照片**。同一主干的多份文件已经收敛成这里的一份。"""

    identity: str  # "<class_key>/<stem>"
    class_key: str
    stem: str
    path: Path
    md5: str
    source: str  # 数据源注册名
    upstream_split: str | None  # 上游把它放在 train/val/test 哪一个（可能没有）
    variants: int  # 这张照片一共有几份文件（含被丢弃的扩展名）
    split: str = ""  # 本脚本重新分配的切分


@dataclass
class BuildReport:
    photos: list[Photo] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    contributed: dict[str, int] = field(default_factory=dict)
    variants_dropped: int = 0
    cross_source_collisions: int = 0
    missing_sources: list[str] = field(default_factory=list)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _collect_source(root: Path, source: str) -> list[tuple[Path, str, str | None]]:
    """扫一个数据源，返回 ``(图片, 类别目录名, 上游切分名或 None)``。

    兼容两种布局：``<root>/Train/<类>/``（已切分）与 ``<root>/<类>/``（平铺）。
    """
    out: list[tuple[Path, str, str | None]] = []

    # 先看有没有上游切分目录
    found_split = False
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        canonical = _UPSTREAM_SPLITS.get(child.name)
        if canonical is None:
            continue
        found_split = True
        for class_dir in sorted(p for p in child.iterdir() if p.is_dir()):
            out.extend((image, class_dir.name, canonical) for image in _iter_images(class_dir))

    if found_split:
        return out

    # 平铺布局
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        out.extend((image, class_dir.name, None) for image in _iter_images(class_dir))
    return out


def _pick_best(paths: list[Path]) -> Path:
    """同一主干有多份文件时挑一份：先按扩展名偏好，再按字节数取大。"""

    def sort_key(path: Path) -> tuple[int, int, str]:
        suffix = path.suffix.lower()
        rank = _EXT_PREFERENCE.index(suffix) if suffix in _EXT_PREFERENCE else len(_EXT_PREFERENCE)
        return (rank, -path.stat().st_size, path.name)

    return min(paths, key=sort_key)


def collect() -> BuildReport:
    """把所有源扫成一张「不同照片」表，按主干去重。"""
    report = BuildReport()
    by_dir = {c.source_dir: c for c in CLASSES}

    # 先按 (类别, 主干) 归组，组内再挑一份文件
    groups: dict[tuple[str, str], list[tuple[Path, str, str | None]]] = {}
    for source in SOURCES:
        root = ds.find(source)
        if root is None:
            report.missing_sources.append(source)
            continue
        for image, class_dir, upstream in _collect_source(root, source):
            spec = by_dir.get(class_dir)
            if spec is None:
                continue  # 不在 4 类里，整批跳过
            groups.setdefault((spec.key, image.stem), []).append((image, source, upstream))

    for (class_key, stem), members in sorted(groups.items()):
        # 数据源顺序优先：SOURCES 里靠前的源占住这个主干
        members.sort(key=lambda m: SOURCES.index(m[1]))
        chosen, source, upstream = members[0]
        if len({m[1] for m in members}) > 1:
            report.cross_source_collisions += 1
        report.variants_dropped += len(members) - 1
        report.photos.append(
            Photo(
                identity=f"{class_key}/{stem}",
                class_key=class_key,
                stem=stem,
                path=chosen,
                md5=_md5(chosen),
                source=source,
                upstream_split=upstream,
                variants=len(members),
            )
        )
        report.contributed[source] = report.contributed.get(source, 0) + 1

    split_pool(report.photos)
    report.counts = _tally(report.photos)
    return report


def split_pool(photos: list[Photo], *, val_ratio: float = 0.2, test_ratio: float = 0.2) -> None:
    """给整池照片做**确定性**分层切分，就地写回 ``photo.split``。

    为什么不用上游切分：实测上游的 Valid/Test 是 Train 的子集（每类
    ``Train∩Valid=50``、``Train∩Test=50``），沿用会让测试图出现在训练里。

    为什么按 sha1(主干) 排序而不是直接按主干排序：主干形如 ``Healthy Fish (37)``，
    字典序会带上原始编号的排序信息；先哈希再排序等价于一次固定种子的随机打乱，
    **同一次输入永远得到同一次切分**，今天和明天的测试集是同一个，指标才能纵向比较。
    """
    by_class: dict[str, list[Photo]] = {}
    for photo in photos:
        by_class.setdefault(photo.class_key, []).append(photo)

    for class_photos in by_class.values():
        class_photos.sort(key=lambda p: hashlib.sha1(p.stem.encode("utf-8")).hexdigest())
        n = len(class_photos)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        for index, photo in enumerate(class_photos):
            if index < n_test:
                photo.split = "test"
            elif index < n_test + n_val:
                photo.split = "val"
            else:
                photo.split = "train"


def _tally(photos: list[Photo]) -> dict[str, dict[str, int]]:
    counts = {c.key: {"train": 0, "val": 0, "test": 0} for c in CLASSES}
    for photo in photos:
        counts[photo.class_key][photo.split] += 1
    return counts


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------


def write_classification(report: BuildReport, out: Path, *, force: bool) -> None:
    """写 ``<out>/{train,val,test}/<class>/<img>`` —— YOLOv8 分类布局。"""
    if out.exists():
        if not force:
            raise SystemExit(f"输出目录已存在：{out}\n要重建就加 --force")
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (out / split / cls.key).mkdir(parents=True, exist_ok=True)

    used: dict[Path, str] = {}
    for photo in report.photos:
        target_dir = out / photo.split / photo.class_key
        name = f"{photo.stem}{photo.path.suffix.lower()}"
        target = target_dir / name
        if target.exists():
            # 主干相同但类别目录不同才会撞名（正常不会）；补哈希后缀，不覆盖。
            name = f"{photo.stem}-{photo.md5[:8]}{photo.path.suffix.lower()}"
            target = target_dir / name
        used[target] = photo.identity
        shutil.copy2(photo.path, target)

    _write_info(report, out, layout="classification")


def write_detection(report: BuildReport, out: Path, *, force: bool) -> None:
    """写一份**弱监督**检测布局：每张图一个覆盖全图的框。

    这不是真的目标检测标注——数据集里根本没有框。它给出的 ``mAP@0.5``
    只反映「图上有没有这个病 / 属于哪一类」，**不能解释成「病灶被框住了」**。
    存在的意义：产品链路要求检测器输出 ``(class, confidence, box)``，
    且用户点名要 mAP@0.5 口径，那就给出口径并标注清楚。
    """
    if out.exists():
        if not force:
            raise SystemExit(f"输出目录已存在：{out}\n要重建就加 --force")
        shutil.rmtree(out)
    class_index = {c.key: i for i, c in enumerate(CLASSES)}

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for photo in report.photos:
        stem = f"{photo.class_key}_{photo.md5[:12]}"
        shutil.copy2(photo.path, out / "images" / photo.split / f"{stem}.jpg")
        index = class_index[photo.class_key]
        # YOLO 框格式：class cx cy w h，全部归一化。全图框 = (0.5, 0.5, 1.0, 1.0)。
        (out / "labels" / photo.split / f"{stem}.txt").write_text(
            f"{index} 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8"
        )

    names = "\n".join(f"  {i}: {c.key}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(
        "# 弱监督检测数据集：每张图一个全图框，只为产出 mAP@0.5 口径。\n"
        "# 框不是病灶位置，别当成真的检测标注用。\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    _write_info(report, out, layout="detection-weak")


def _write_info(report: BuildReport, out: Path, *, layout: str) -> None:
    payload = {
        "layout": layout,
        "identity_rule": "文件名主干（不含扩展名）——同一张照片的多扩展名副本视为同一张",
        "split_rule": "本脚本自建：按 sha1(主干) 排序后 60/20/20 分层；不使用上游切分",
        "classes": [
            {"index": i, "key": c.key, "zh": c.zh, "source_dir": c.source_dir}
            for i, c in enumerate(CLASSES)
        ],
        "dropped_classes": DROPPED,
        "counts": report.counts,
        "totals": {
            split: sum(v[split] for v in report.counts.values())
            for split in ("train", "val", "test")
        },
        "distinct_photos": len(report.photos),
        "variants_dropped": report.variants_dropped,
        "cross_source_collisions": report.cross_source_collisions,
        "contributed_by_source": report.contributed,
        "missing_sources": report.missing_sources,
        "manifest": [
            {
                "identity": p.identity,
                "class": p.class_key,
                "split": p.split,
                "source": p.source,
                "upstream_split": p.upstream_split,
                "variants": p.variants,
                "md5": p.md5,
                "source_path": str(p.path),
            }
            for p in report.photos
        ],
    }
    (out / "dataset_info.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def print_report(report: BuildReport) -> None:
    width = max(len(c.key) for c in CLASSES) + 2
    print()
    print("=" * 76)
    print("  4 类训练集构建结果")
    print("=" * 76)
    print(f"  {'类别':<{width}}{'中文':<12}{'train':>7}{'val':>7}{'test':>7}{'合计':>7}")
    print("  " + "-" * 72)
    for cls in CLASSES:
        row = report.counts[cls.key]
        print(
            f"  {cls.key:<{width}}{cls.zh:<12}"
            f"{row['train']:>7}{row['val']:>7}{row['test']:>7}{sum(row.values()):>7}"
        )
    totals = {
        split: sum(v[split] for v in report.counts.values())
        for split in ("train", "val", "test")
    }
    print("  " + "-" * 72)
    print(
        f"  {'合计':<{width}}{'':<12}"
        f"{totals['train']:>7}{totals['val']:>7}{totals['test']:>7}"
        f"{sum(totals.values()):>7}"
    )
    print()
    print(f"  不同照片总数（按主干去重）：{len(report.photos)}")
    print(f"  因「同一张图多个扩展名」丢弃的副本：{report.variants_dropped}")
    print(f"  跨源同名主干（同图被多个源收录）：{report.cross_source_collisions}")
    print(f"  各源最终贡献的照片数：{report.contributed}")
    if report.missing_sources:
        print(f"  缺失的数据源：{report.missing_sources}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_yolo_dataset",
        description="把三份鱼病图像集拼成 YOLO 4 分类训练集",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="分类布局输出目录")
    parser.add_argument("--det-out", type=Path, default=DEFAULT_DET_OUT, help="检测布局输出目录")
    parser.add_argument("--with-det", action="store_true", help="额外产出弱监督检测布局")
    parser.add_argument("--report", action="store_true", help="只统计，不写任何文件")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时先删再建")
    args = parser.parse_args(argv)

    report = collect()
    print_report(report)

    if args.report:
        return 0
    if not report.photos:
        print("  没有任何照片入选，检查数据集是否已导入。", file=sys.stderr)
        return 1

    write_classification(report, args.out, force=args.force)
    print(f"  分类布局 → {args.out}")
    if args.with_det:
        write_detection(report, args.det_out, force=args.force)
        print(f"  检测布局 → {args.det_out}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
