"""把「鱼体 + 病灶」实例标注集清洗成 YOLO-seg 能直接吃的训练集。

输入是 `新建文件夹/test/{images,labels}` 那一份数据。它有几处必须先洗干净的地方，
否则训练出来的模型会学错：

1. **两种标注格式混在一起**。404 行是实例多边形（`class x1 y1 x2 y2 …`，6~41 个点），
   30 行却是轴对齐框（`class cx cy w h`）。这两者字段数不同，直接喂 YOLO 会把框的
   4 个数当成 2 个点。本脚本把 `cx cy w h` 展开成矩形多边形，统一按实例多边形处理。
   判据是字段数：``4`` 个数 → 框；``≥6`` 且为偶数 → 多边形。
2. **坐标可能越界**。浮点标注偶尔给出 -0.0001 或 1.0002，YOLO 会直接断言失败。
   这里做 ``clip`` 到 [0, 1]，裁剪后面积归零的实例丢弃。
3. **有完全重复的照片**（同一张图两个文件名）。重复图若一个进 train 一个进 val，
   验证指标就是假的。这里用 sha256 找完全重复、用 dHash 找近似重复，
   **同一簇永不跨 split**。
4. **小样本必须分层**。按「这张图有没有病灶」分层抽样，避免 val 里全是无病灶图。

清洗只做不改变语义的事：不猜类别、不补框、不合成图。所有丢弃与修复都记进
``info.json``，报告里逐条打印。

用法（在工程根目录）::

    python scripts/build_lesion_dataset.py --dry-run     # 只体检，不落盘
    python scripts/build_lesion_dataset.py               # 清洗 + 落盘
    python scripts/build_lesion_dataset.py --val-ratio 0.25 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

#: 原始标注集的默认位置：工程上级目录的 `新建文件夹/test`。
DEFAULT_SRC = PROJECT_ROOT.parent / "新建文件夹" / "test"

#: 清洗后的输出目录（YOLO 结构：images/{train,val} + labels/{train,val}）。
DEFAULT_OUT = PROJECT_ROOT / "data" / "derived" / "fish_lesion_seg"

#: 类别顺序即 data.yaml 里 `names` 的下标，**不可随意调整**。
CLASSES: tuple[str, ...] = ("fish", "lesion")

#: 类别语义（写进 info.json，给后来人一个不需要猜的说明）。
CLASS_MEANING: dict[str, str] = {
    "fish": "整条鱼体（用于「图里到底有没有鱼」与鱼体定位）",
    "lesion": "病灶/溃烂区域（出血、溃烂、水霉等外观异常的统一框）",
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

#: 坐标容差：超过这个量级才认为真的越界，避免把 1e-7 的浮点噪声当问题。
COORD_EPS = 1e-3

#: 多边形面积下限（归一化坐标系下 1e-5 约等于 640×640 图上的 4 个像素）。
MIN_AREA = 1e-5

#: dHash 近重复判定阈值（64 bit 里的汉明距离）。
DHASH_THRESHOLD = 4

SEED = 42


# --------------------------------------------------------------------------- #
# 一、单行标注的解析与归一化
# --------------------------------------------------------------------------- #

@dataclass
class Instance:
    """一个标注实例，坐标一律是归一化的 (x, y) 顶点序列。"""

    cls: int
    points: list[tuple[float, float]]
    kind: str  # "poly" | "bbox" | "line"

    @property
    def label(self) -> str:
        return CLASSES[self.cls] if 0 <= self.cls < len(CLASSES) else f"class{self.cls}"

    def bbox(self) -> tuple[float, float, float, float]:
        """返回 (cx, cy, w, h)。"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0

    def area(self) -> float:
        """鞋带公式求多边形面积（绝对值）。顶点少于 3 个时按 0 处理。"""
        pts = self.points
        if len(pts) < 3:
            return 0.0
        s = 0.0
        for i, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0

    def to_label_line(self) -> str:
        """序列化回 YOLO-seg 的一行：``cls x1 y1 x2 y2 …``（6 位小数）。"""
        coords = " ".join(f"{v:.6f}" for p in self.points for v in p)
        return f"{self.cls} {coords}"


def _rect_points(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    """把 `cx cy w h` 展开成 4 角点。"""
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def clip_points(points: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], bool]:
    """把坐标裁剪进 [0, 1]，返回 (新坐标, 是否发生过裁剪)。"""
    out: list[tuple[float, float]] = []
    clipped = False
    for x, y in points:
        nx = min(1.0, max(0.0, x))
        ny = min(1.0, max(0.0, y))
        if abs(nx - x) > 1e-9 or abs(ny - y) > 1e-9:
            clipped = True
        out.append((nx, ny))
    return out, clipped


def parse_line(line: str, *, n_classes: int = len(CLASSES)) -> tuple[Instance | None, str]:
    """解析一行标注。

    Returns:
        (实例, 说明)。解析失败时实例为 ``None``，说明是失败原因。
    """
    parts = line.split()
    if len(parts) < 2:
        return None, "字段太少"
    try:
        cls = int(float(parts[0]))
    except ValueError:
        return None, "类别不是数字"
    if cls < 0 or cls >= n_classes:
        return None, f"类别 {cls} 超出 0..{n_classes - 1}"
    try:
        nums = [float(v) for v in parts[1:]]
    except ValueError:
        return None, "坐标不是数字"

    if len(nums) == 4:
        cx, cy, w, h = nums
        if w <= 0 or h <= 0:
            return None, "框宽高非正"
        inst = Instance(cls=cls, points=_rect_points(cx, cy, w, h), kind="bbox")
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        pts = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        inst = Instance(cls=cls, points=pts, kind="poly")
    elif len(nums) == 2:
        return None, "只有 2 个点（连线，无面积）"
    else:
        return None, f"坐标个数 {len(nums)} 无法解释（应为 4 或 ≥6 的偶数）"

    inst.points, was_clipped = clip_points(inst.points)
    if was_clipped:
        return inst, "坐标越界已裁剪"
    if inst.area() < MIN_AREA:
        return None, "面积过小"
    return inst, ""


# --------------------------------------------------------------------------- #
# 二、文件级体检
# --------------------------------------------------------------------------- #

def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash_of(path: Path, size: int = 8) -> int | None:
    """差分感知哈希：缩到 (size+1)×size 灰度图，比较相邻像素亮度。"""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow 是 ultralytics 的硬依赖
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
            # 灰度图的 tobytes() 就是一维像素序列，等价于 getdata() 但不会触发弃用告警
            pixels = list(small.tobytes())
    except Exception:  # noqa: BLE001 —— 打不开就说明图片本身有问题，交给上层判损坏
        return None
    bits = 0
    idx = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits |= int(pixels[base + col] > pixels[base + col + 1]) << idx
            idx += 1
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def check_image(path: Path) -> str | None:
    """校验图片可解码；返回错误说明或 None。"""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 三、清洗主体
# --------------------------------------------------------------------------- #

@dataclass
class Photo:
    """一张通过解析的照片及其标注。"""

    stem: str
    image: Path
    sha256: str
    dhash: int | None
    instances: list[Instance]
    cluster: int = -1
    split: str = ""

    @property
    def has_lesion(self) -> bool:
        return any(inst.cls == CLASSES.index("lesion") for inst in self.instances)


@dataclass
class CleanReport:
    """清洗过程的全部计数，最终写进 info.json。"""

    src: Path
    images_found: int = 0
    labels_found: int = 0
    kept: int = 0
    dropped_corrupt: list[tuple[str, str]] = field(default_factory=list)
    dropped_no_valid_instance: list[str] = field(default_factory=list)
    dropped_duplicate: list[tuple[str, str]] = field(default_factory=list)
    unpaired_images: list[str] = field(default_factory=list)
    unpaired_labels: list[str] = field(default_factory=list)
    line_issues: Counter = field(default_factory=Counter)
    kind_counter: Counter = field(default_factory=Counter)
    class_counter: Counter = field(default_factory=Counter)
    clipped_instances: int = 0
    lesion_outside_fish: int = 0
    overlap_with_existing: list[str] = field(default_factory=list)
    splits: Counter = field(default_factory=Counter)
    lesion_split: dict[str, dict[str, int]] = field(default_factory=dict)


def _existing_hashes() -> set[str]:
    """现有 data/datasets 下图片的 sha256 集合，用来提示跨任务重复。"""
    root = PROJECT_ROOT / "data" / "datasets"
    if not root.is_dir():
        return set()
    out: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                out.add(sha256_of(path))
            except OSError:
                continue
    return out


def clean(src: Path, out: Path, *, val_ratio: float, seed: int, check_overlap: bool) -> tuple[list[Photo], CleanReport]:
    """扫描源目录，解析并清洗，返回保留下来的照片与统计。"""
    report = CleanReport(src=src)
    image_dir, label_dir = src / "images", src / "labels"

    images = {p.stem: p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in sorted(label_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".txt"}
    report.images_found = len(images)
    report.labels_found = len(labels)
    report.unpaired_images = sorted(set(images) - set(labels))
    report.unpaired_labels = sorted(set(labels) - set(images))

    photos: list[Photo] = []
    for stem in sorted(set(images) & set(labels)):
        image, label = images[stem], labels[stem]
        err = check_image(image)
        if err:
            report.dropped_corrupt.append((stem, err))
            continue
        instances: list[Instance] = []
        for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            inst, note = parse_line(line)
            if note:
                report.line_issues[note] += 1
            if note == "坐标越界已裁剪":
                report.clipped_instances += 1
            if inst is None:
                continue
            instances.append(inst)
            report.kind_counter[inst.kind] += 1
            report.class_counter[inst.label] += 1
        if not instances:
            report.dropped_no_valid_instance.append(stem)
            continue
        photos.append(
            Photo(
                stem=stem,
                image=image,
                sha256=sha256_of(image),
                dhash=dhash_of(image),
                instances=instances,
            )
        )

    # ---- 病灶是否落在鱼体内（只统计，不丢：小病灶可能压在边缘） ----
    for photo in photos:
        fish = [i for i in photo.instances if i.cls == CLASSES.index("fish")]
        lesions = [i for i in photo.instances if i.cls == CLASSES.index("lesion")]
        if not fish or not lesions:
            continue
        fx0 = min(min(p[0] for p in f.points) for f in fish)
        fx1 = max(max(p[0] for p in f.points) for f in fish)
        fy0 = min(min(p[1] for p in f.points) for f in fish)
        fy1 = max(max(p[1] for p in f.points) for f in fish)
        for les in lesions:
            cx, cy, _, _ = les.bbox()
            if not (fx0 - 0.02 <= cx <= fx1 + 0.02 and fy0 - 0.02 <= cy <= fy1 + 0.02):
                report.lesion_outside_fish += 1

    if check_overlap:
        existing = _existing_hashes()
        report.overlap_with_existing = [p.stem for p in photos if p.sha256 in existing]

    # ---- 去重：先按 sha256 分组，再用 dHash 把近似图并进同一簇 ----
    photos = _dedupe(photos, report)

    # ---- 分层划分：同一簇不跨 split，按「有无病灶」分层 ----
    _assign_clusters(photos)
    _split(photos, val_ratio=val_ratio, seed=seed, report=report)
    report.kept = len(photos)
    return photos, report


def _dedupe(photos: list[Photo], report: CleanReport) -> list[Photo]:
    """完全重复按 sha256 去掉；近似重复按 dHash 汉明距离并簇。"""
    by_hash: dict[str, Photo] = {}
    kept: list[Photo] = []
    for photo in photos:
        first = by_hash.get(photo.sha256)
        if first is not None:
            report.dropped_duplicate.append((photo.stem, f"与 {first.stem} 完全重复"))
            continue
        by_hash[photo.sha256] = photo
        kept.append(photo)

    # 近似重复：只在同一来源内部比较，O(n²) 对 200 张完全够用
    closured: list[Photo] = []
    for photo in kept:
        if photo.dhash is None:
            closured.append(photo)
            continue
        hit = next(
            (k for k in closured if k.dhash is not None and hamming(k.dhash, photo.dhash) <= DHASH_THRESHOLD),
            None,
        )
        if hit is not None:
            report.dropped_duplicate.append(
                (photo.stem, f"与 {hit.stem} 近似重复 dHash={hamming(hit.dhash, photo.dhash)}")
            )
            continue
        closured.append(photo)
    return closured


def _assign_clusters(photos: list[Photo]) -> None:
    """按 dHash 距离做并查集，给每张图一个簇号（用于防止跨 split 泄漏）。"""
    n = len(photos)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            a, b = photos[i].dhash, photos[j].dhash
            if a is not None and b is not None and hamming(a, b) <= DHASH_THRESHOLD:
                union(i, j)
    # 簇号重整为连续整数
    mapping: dict[int, int] = {}
    for i, photo in enumerate(photos):
        root = find(i)
        if root not in mapping:
            mapping[root] = len(mapping)
        photo.cluster = mapping[root]


def _split(photos: list[Photo], *, val_ratio: float, seed: int, report: CleanReport) -> None:
    """按「是否有病灶」分层，且同一簇整体进入同一 split。"""
    rng = random.Random(seed)
    groups: dict[tuple[bool, int], list[Photo]] = {}
    for photo in photos:
        groups.setdefault((photo.has_lesion, photo.cluster), []).append(photo)

    for (_has_lesion, _cluster), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for photo in members:
            photo.split = "val" if rng.random() < val_ratio else "train"

    # 兜底：若某个 stratum 全落进同一侧，把最大簇里的一部分挪过去
    for has_lesion in (True, False):
        train = [p for p in photos if p.has_lesion == has_lesion and p.split == "train"]
        val = [p for p in photos if p.has_lesion == has_lesion and p.split == "val"]
        if train and not val and len(train) >= 5:
            train.sort(key=lambda p: p.stem)
            for photo in train[-max(1, len(train) // 5):]:
                photo.split = "val"

    for photo in photos:
        report.splits[photo.split] += 1
    for split in ("train", "val"):
        subset = [p for p in photos if p.split == split]
        report.lesion_split[split] = {
            "images": len(subset),
            "with_lesion": sum(1 for p in subset if p.has_lesion),
            "fish_boxes": sum(sum(1 for i in p.instances if i.cls == 0) for p in subset),
            "lesion_boxes": sum(sum(1 for i in p.instances if i.cls == 1) for p in subset),
        }


# --------------------------------------------------------------------------- #
# 四、落盘
# --------------------------------------------------------------------------- #

def write_dataset(photos: list[Photo], out: Path, report: CleanReport, *, force: bool) -> None:
    if out.exists():
        if not force:
            raise SystemExit(f"输出目录已存在：{out}\n加 --force 覆盖，或换一个 --out。")
        shutil.rmtree(out)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for photo in photos:
        dst_img = out / "images" / photo.split / f"{photo.stem}{photo.image.suffix.lower()}"
        shutil.copy2(photo.image, dst_img)
        dst_lbl = out / "labels" / photo.split / f"{photo.stem}.txt"
        dst_lbl.write_text(
            "\n".join(inst.to_label_line() for inst in photo.instances) + "\n",
            encoding="utf-8",
        )

    (out / "data.yaml").write_text(
        "# 由 scripts/build_lesion_dataset.py 生成，勿手改\n"
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES)),
        encoding="utf-8",
    )


def write_info(photos: list[Photo], out: Path, report: CleanReport, *, val_ratio: float, seed: int) -> None:
    info = {
        "source": str(report.src),
        "classes": list(CLASSES),
        "class_meaning": CLASS_MEANING,
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {
            "images_found": report.images_found,
            "labels_found": report.labels_found,
            "kept": report.kept,
            "dropped_corrupt": len(report.dropped_corrupt),
            "dropped_no_valid_instance": len(report.dropped_no_valid_instance),
            "dropped_duplicate": len(report.dropped_duplicate),
            "clipped_instances": report.clipped_instances,
            "lesion_outside_fish": report.lesion_outside_fish,
        },
        "instance_kinds": dict(report.kind_counter),
        "instances_by_class": dict(report.class_counter),
        "line_issues": dict(report.line_issues),
        "splits": report.lesion_split,
        "dropped": {
            "corrupt": [{"stem": s, "error": e} for s, e in report.dropped_corrupt],
            "duplicate": [{"stem": s, "why": w} for s, w in report.dropped_duplicate],
            "no_valid_instance": report.dropped_no_valid_instance,
            "unpaired_images": report.unpaired_images,
            "unpaired_labels": report.unpaired_labels,
        },
        "overlap_with_existing_datasets": report.overlap_with_existing,
        "files": [
            {
                "stem": p.stem,
                "split": p.split,
                "sha256": p.sha256,
                "instances": [{"cls": i.cls, "kind": i.kind, "area": round(i.area(), 6)} for i in p.instances],
            }
            for p in photos
        ],
    }
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 五、报告
# --------------------------------------------------------------------------- #

def print_report(photos: list[Photo], report: CleanReport, out: Path) -> None:
    c = _console
    print()
    print("=" * 52)
    print("  病灶数据集清洗报告")
    print("=" * 52)
    print(f"  源目录      : {report.src}")
    print(f"  输出目录    : {out}")
    print(f"  发现        : {report.images_found} 图 / {report.labels_found} 标签")
    print(f"  保留        : {c.mark(True)} {report.kept} 张")
    print()
    print("  【标注格式】")
    for kind, n in report.kind_counter.most_common():
        print(f"    {kind:<6} {n} 个实例")
    for label, n in report.class_counter.most_common():
        print(f"    {label:<6} {n} 个实例")
    if report.line_issues:
        print("    解析异常：")
        for note, n in report.line_issues.most_common():
            print(f"      {note} × {n}")
    print()
    print("  【丢弃与修复】")
    print(f"    完全/近似重复   : {len(report.dropped_duplicate)}")
    print(f"    损坏图片        : {len(report.dropped_corrupt)}")
    print(f"    无有效实例      : {len(report.dropped_no_valid_instance)}")
    print(f"    坐标裁剪        : {report.clipped_instances} 个实例")
    print(f"    病灶落在鱼体外  : {report.lesion_outside_fish} 个（仅统计，未丢弃）")
    for stem, why in report.dropped_duplicate:
        print(f"      - {stem}: {why}")
    for stem, err in report.dropped_corrupt:
        print(f"      - {stem}: {err}")
    if report.overlap_with_existing:
        print(f"    与现有数据集重复: {len(report.overlap_with_existing)} 张 "
              f"（{', '.join(report.overlap_with_existing[:6])}{' …' if len(report.overlap_with_existing) > 6 else ''}）")
    print()
    print("  【划分】")
    for split in ("train", "val"):
        s = report.lesion_split.get(split, {})
        print(
            f"    {split:<5} {s.get('images', 0):>3} 张  "
            f"含病灶 {s.get('with_lesion', 0):>3} 张  "
            f"鱼框 {s.get('fish_boxes', 0):>3}  病灶框 {s.get('lesion_boxes', 0):>3}"
        )
    print()


# --------------------------------------------------------------------------- #
# 六、命令行
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_lesion_dataset.py",
        description="清洗「鱼体 + 病灶」标注集并落成 YOLO-seg 结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help=f"原始目录（含 images/ 与 labels/），默认 {DEFAULT_SRC}")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"输出目录，默认 {DEFAULT_OUT}")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=SEED, help=f"划分随机种子，默认 {SEED}")
    parser.add_argument("--dry-run", action="store_true", help="只体检与统计，不写任何文件")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时覆盖")
    parser.add_argument("--no-overlap-check", action="store_true", help="跳过与现有数据集的重复检查（会快很多）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.src.is_dir():
        print(f"{_console.mark(False)} 源目录不存在：{args.src}")
        return 1
    if not 0.0 < args.val_ratio < 1.0:
        print(f"{_console.mark(False)} --val-ratio 必须在 (0, 1) 之间，收到 {args.val_ratio}")
        return 1

    photos, report = clean(
        args.src,
        args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        check_overlap=not args.no_overlap_check,
    )
    print_report(photos, report, args.out)

    if args.dry_run:
        print("  --dry-run：未写任何文件。")
        return 0
    if not photos:
        print(f"{_console.mark(False)} 没有任何可用样本，未落盘。")
        return 1

    write_dataset(photos, args.out, report, force=args.force)
    write_info(photos, args.out, report, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  {_console.mark(True)} 已写出 {args.out}")
    print(f"    data.yaml : {(args.out / 'data.yaml').as_posix()}")
    print(f"    info.json : {(args.out / 'info.json').as_posix()}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
