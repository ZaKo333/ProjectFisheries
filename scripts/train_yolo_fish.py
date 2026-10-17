"""训练鱼病 4 分类 YOLOv8 权重，并把指标写成评测报告能直接引用的 JSON/Markdown。

    python scripts/train_yolo_fish.py --report        # 只体检：数据集 + 依赖 + 权重缓存
    python scripts/train_yolo_fish.py                 # 训练分类模型（默认 60 epoch）
    python scripts/train_yolo_fish.py --epochs 120
    python scripts/train_yolo_fish.py --det           # 额外训一个弱监督检测器（为 mAP@0.5）

产物
----
    models/yolov8-fish.pt          分类权重（`fish_disease_detect` 的默认后端）
    models/yolov8-fish-det.pt      弱监督检测权重（仅 --det 时产出）
    data/derived/runs/cls/         训练过程日志、混淆矩阵、曲线图
    data/derived/metrics.json      全部指标，供评测报告引用

## 为什么是「分类」而不是「检测」

上游三份数据同盘扫过，**一个检测框标注都没有**（没有 .txt/.xml/.json）。
所以能真训的只有分类；`yolov8-fish.pt` 是 YOLOv8-cls 权重，这也正是
`src/fisheries/tools/fish_disease.py` 的 `_yolo_predict()` 要求的
（它拿到 `results.probs is None` 就会报「请确认使用的是 YOLOv8-cls 分类权重」）。

## mAP@0.5 是怎么算出来的（重要，别误读）

用户点名要 mAP@0.5，但**没有框就没有真正的定位指标**。本脚本给出的是
**弱监督全图框口径**下的 mAP@0.5：

1. 数据集的每张图被赋一个覆盖全图的框（`build_yolo_dataset.py --with-det`
   产出的正是这份布局，`(class, cx, cy, w, h) = (c, 0.5, 0.5, 1.0, 1.0)`）；
2. 分类器对每张图输出一个类别与置信度，等价于「在全图框上打出该类、得分 = 置信度」；
3. 预测框与真值框都是全图 → 只要类别对，IoU 恒为 1.0，即恒过 0.5 阈值；
4. 于是按置信度排序做标准的 PR 曲线积分，得到每类 AP，再取均值 = mAP@0.5。

**它衡量的是「分类排序质量」，不是「病灶被框住了」。** 报告里必须写明这一点。
真正的定位 mAP 需要人工/半自动标注病灶框，属于本次范围之外。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
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

import build_yolo_dataset as byd  # noqa: E402  （同目录脚本）

#: YOLOv8 预训练权重的官方发布地址。本机直连 github.com 会被重置，
#: 因此按顺序尝试镜像前缀；空串代表直连，放在最后兜底。
_ASSET_TAG = "v8.3.0"
_ASSET_BASE = f"https://github.com/ultralytics/assets/releases/download/{_ASSET_TAG}"
_MIRROR_PREFIXES: tuple[str, ...] = (
    "https://ghproxy.net/",
    "https://gh-proxy.com/",
    "",
)

CLS_MODEL = "yolov8n-cls.pt"
DET_MODEL = "yolov8n.pt"

WEIGHTS_OUT = PROJECT_ROOT / "models" / "yolov8-fish.pt"
DET_WEIGHTS_OUT = PROJECT_ROOT / "models" / "yolov8-fish-det.pt"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
RUNS_DIR = PROJECT_ROOT / "data" / "derived" / "runs"
METRICS_OUT = PROJECT_ROOT / "data" / "derived" / "metrics.json"


# --------------------------------------------------------------------------
# 依赖与权重自检
# --------------------------------------------------------------------------


def check_environment() -> tuple[bool, list[str]]:
    """检查训练栈是否就绪，返回 (是否可训练, 说明行)。"""
    lines: list[str] = []
    ok = True

    try:
        import torch

        cuda = torch.cuda.is_available()
        lines.append(f"torch      : {torch.__version__}  cuda_available={cuda}")
        if cuda:
            lines.append(f"gpu        : {torch.cuda.get_device_name(0)}")
            lines.append(f"capability : sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))}")
        else:
            lines.append("gpu        : 不可用，将退回 CPU 训练（会慢很多）")
    except ImportError:
        ok = False
        lines.append("torch      : ✗ 未安装 —— 先装 torch（见 docs/模型训练说明.md）")

    try:
        import ultralytics

        _console.install()  # ultralytics 导入时会把 stdout 改成 UTF-8，抢回来
        lines.append(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        ok = False
        lines.append("ultralytics: ✗ 未安装")

    cls_dir = byd.DEFAULT_OUT
    if cls_dir.is_dir():
        counts = {
            s: sum(1 for _ in (cls_dir / s).rglob("*") if _.is_file())
            for s in ("train", "val", "test")
        }
        lines.append(f"训练集     : {cls_dir}  train={counts['train']} val={counts['val']} test={counts['test']}")
        if counts["train"] == 0 or counts["val"] == 0:
            ok = False
            lines.append("训练集     : ✗ 缺少 train 或 val 子集，先跑 build_yolo_dataset.py")
    else:
        ok = False
        lines.append(f"训练集     : ✗ 不存在 {cls_dir}，先跑 build_yolo_dataset.py")

    have = _find_pretrained(CLS_MODEL)
    lines.append(
        f"预训练权重 : {have if have else '✗ 未缓存 ' + CLS_MODEL + '（首次训练会自动下载）'}"
    )
    return ok, lines


def _find_pretrained(name: str) -> Path | None:
    """在常见缓存位置找预训练权重。"""
    candidates = [
        PRETRAINED_DIR / name,
        PROJECT_ROOT / name,
        PROJECT_ROOT / "models" / name,
        Path.cwd() / name,
        Path.home() / "AppData" / "Roaming" / "Ultralytics" / name,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def ensure_pretrained(name: str) -> Path:
    """确保预训练权重在本地，返回其路径。

    直连 github.com 在本机会被重置，所以先试镜像前缀。ultralytics 自己也只会
    直连下载，因此这里替它把文件放到 ``models/pretrained/`` 再传绝对路径进去。
    """
    found = _find_pretrained(name)
    if found is not None:
        return found

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    target = PRETRAINED_DIR / name
    url_path = f"{_ASSET_BASE}/{name}"
    errors: list[str] = []

    for prefix in _MIRROR_PREFIXES:
        url = f"{prefix}{url_path}"
        label = prefix or "（直连）"
        print(f"  下载预训练权重：{name}  经由 {label}")
        result = subprocess.run(
            ["curl.exe", "-L", "--fail", "--retry", "3", "--retry-delay", "3",
             "--connect-timeout", "20", "-o", str(target), url],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            size_mb = target.stat().st_size / 1e6
            print(f"  ✓ {label} 成功，{size_mb:.1f} MB")
            return target
        errors.append(f"{label} → exit={result.returncode} {result.stderr.strip()[:120]}")
        target.unlink(missing_ok=True)

    raise SystemExit(
        f"下载 {name} 失败，已试过所有镜像：\n  " + "\n  ".join(errors)
        + f"\n请手动下载 {url_path} 放到 {PRETRAINED_DIR}"
    )


# --------------------------------------------------------------------------
# 指标计算
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    top1: float = 0.0
    macro_f1: float = 0.0
    mAP50: float = 0.0
    n: int = 0
    errors: list[dict[str, object]] = field(default_factory=list)


def _average_precision(pairs: list[tuple[float, int]], n_pos: int) -> float:
    """标准 AP：按置信度降序，用全点插值算 PR 曲线下面积。

    Args:
        pairs: ``(confidence, is_true_positive)``，已按需排序由本函数负责。
        n_pos: 该类在测试集里的真实样本数（GT 框数）。
    """
    if n_pos == 0:
        return float("nan")
    pairs = sorted(pairs, key=lambda kv: kv[0], reverse=True)
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, is_tp in pairs:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    # 全点插值：把每个 recall 点的 precision 取成其后所有点的最大值
    best = 0.0
    for i in range(len(precisions) - 1, -1, -1):
        best = max(best, precisions[i])
        precisions[i] = best
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls, strict=True):
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def evaluate(model, class_keys: list[str], test_root: Path) -> EvalResult:
    """在测试集上逐图推理，算出准确率、逐类指标、混淆矩阵与弱监督 mAP@0.5。"""
    result = EvalResult()
    for cls in class_keys:
        result.confusion[cls] = dict.fromkeys(class_keys, 0)

    truths: list[str] = []
    preds: list[str] = []
    confs: list[float] = []

    for cls in class_keys:
        folder = test_root / cls
        if not folder.is_dir():
            continue
        for image in sorted(p for p in folder.iterdir() if p.is_file()):
            output = model.predict(str(image), verbose=False)[0]
            probs = output.probs
            names = output.names
            scores = {names[i]: float(p) for i, p in enumerate(probs.data.tolist())}
            top = max(scores, key=lambda k: scores[k])
            truths.append(cls)
            preds.append(top)
            confs.append(scores[top])
            result.confusion[cls][top] = result.confusion[cls].get(top, 0) + 1
            if top != cls and len(result.errors) < 40:  # 报告里只留样例，别把 JSON 撑爆
                result.errors.append(
                    {"image": str(image), "true": cls, "pred": top, "conf": round(scores[top], 4)}
                )

    result.n = len(truths)
    if result.n == 0:
        return result

    result.top1 = sum(1 for t, p in zip(truths, preds, strict=True) if t == p) / result.n

    # 逐类 precision / recall / F1（按测试集里真实存在的类算）
    f1s: list[float] = []
    for cls in class_keys:
        tp = sum(1 for t, p in zip(truths, preds, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truths, preds, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truths, preds, strict=True) if t == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision == precision and recall == recall and (precision + recall) > 0
            else 0.0
        )
        result.per_class[cls] = {
            "support": tp + fn,
            "precision": round(precision, 4) if precision == precision else None,
            "recall": round(recall, 4) if recall == recall else None,
            "f1": round(f1, 4),
        }
        f1s.append(f1)
    result.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # 弱监督 mAP@0.5：全图框协议下，类别正确 ⟺ IoU=1.0 ≥ 0.5
    aps: list[float] = []
    for cls in class_keys:
        n_pos = sum(1 for t in truths if t == cls)
        pairs = [
            (conf, 1 if (p == cls and t == cls) else 0)
            for t, p, conf in zip(truths, preds, confs, strict=True)
            if p == cls
        ]
        ap = _average_precision(pairs, n_pos)
        if ap == ap:  # 非 NaN
            aps.append(ap)
    result.mAP50 = sum(aps) / len(aps) if aps else 0.0
    return result


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------


def train_classification(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    _console.install()  # ultralytics 导入时会把 stdout 改成 UTF-8，抢回来
    dataset = byd.DEFAULT_OUT
    pretrained = ensure_pretrained(CLS_MODEL)

    print()
    print(f"  分类训练：{CLS_MODEL} → {WEIGHTS_OUT.name}")
    print(f"  数据：{dataset}")
    print(f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print()

    model = YOLO(str(pretrained))
    model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name="cls",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        patience=args.patience,
        plots=True,
        verbose=True,
    )

    best = RUNS_DIR / "cls" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"训练结束但没有找到 best.pt：{best}")

    WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, WEIGHTS_OUT)
    print(f"  ✓ 分类权重 → {WEIGHTS_OUT}")
    return WEIGHTS_OUT


def train_detection(args: argparse.Namespace) -> Path | None:
    """训一个弱监督检测器，专门为产出 ultralytics 口径的 mAP@0.5。"""
    from ultralytics import YOLO

    _console.install()  # ultralytics 导入时会把 stdout 改成 UTF-8，抢回来
    det_dir = byd.DEFAULT_DET_OUT
    if not (det_dir / "data.yaml").is_file():
        print(f"  检测布局不存在，先构建：{det_dir}")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_yolo_dataset.py"),
             "--with-det", "--force"],
            check=True,
        )
    pretrained = ensure_pretrained(DET_MODEL)

    print()
    print(f"  弱监督检测训练：{DET_MODEL} → {DET_WEIGHTS_OUT.name}")
    print("  ⚠ 框是整图，mAP@0.5 只反映分类排序质量，不代表病灶定位。")
    print()

    model = YOLO(str(pretrained))
    model.train(
        data=str(det_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name="det",
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        patience=args.patience,
        plots=True,
        verbose=True,
    )
    best = RUNS_DIR / "det" / "weights" / "best.pt"
    if not best.is_file():
        return None
    DET_WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, DET_WEIGHTS_OUT)
    print(f"  ✓ 检测权重 → {DET_WEIGHTS_OUT}")
    return DET_WEIGHTS_OUT


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------


def write_metrics(result: EvalResult, args: argparse.Namespace, weights: Path) -> None:
    class_keys = [c.key for c in byd.CLASSES]
    zh = {c.key: c.zh for c in byd.CLASSES}
    payload = {
        "model": weights.name,
        "task": "classify",
        "classes": [{"key": c.key, "zh": c.zh} for c in byd.CLASSES],
        "dataset": str(byd.DEFAULT_OUT),
        "identity_rule": "文件名主干去重后的 931 张不同照片；自建 60/20/20 分层切分",
        "hyperparams": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(args.device),
            "seed": args.seed,
        },
        "test_size": result.n,
        "top1_accuracy": round(result.top1, 4),
        "macro_f1": round(result.macro_f1, 4),
        "mAP50_weak_fullframe": round(result.mAP50, 4),
        "mAP50_caveat": (
            "弱监督全图框口径：每张图真值框与预测框都是整图，类别正确即 IoU=1.0≥0.5，"
            "因此该值等价于按置信度排序的分类 AP 均值，**不衡量病灶定位**。"
            "数据集没有任何检测框标注，真正的定位 mAP 无法计算。"
        ),
        "per_class": result.per_class,
        "confusion": result.confusion,
        "misclassified_samples": result.errors,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("=" * 76)
    print("  测试集评估结果")
    print("=" * 76)
    print(f"  测试样本数      : {result.n}")
    print(f"  Top-1 准确率    : {result.top1:.4f}")
    print(f"  宏平均 F1       : {result.macro_f1:.4f}")
    print(f"  mAP@0.5(弱口径) : {result.mAP50:.4f}")
    print()
    print(f"  {'类别':<24}{'样本':>6}{'精确率':>10}{'召回率':>10}{'F1':>10}")
    print("  " + "-" * 60)
    for cls in class_keys:
        row = result.per_class.get(cls)
        if not row:
            continue
        precision = f"{row['precision']:.4f}" if row["precision"] is not None else "—"
        recall = f"{row['recall']:.4f}" if row["recall"] is not None else "—"
        print(
            f"  {cls + '/' + zh[cls]:<24}{row['support']:>6}{precision:>10}{recall:>10}{row['f1']:>10.4f}"
        )
    print()
    print(f"  指标 JSON → {METRICS_OUT}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="train_yolo_fish", description="训练鱼病 4 分类 YOLOv8 权重"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda device，如 0；默认自动")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", action="store_true", help="只做环境与数据体检")
    parser.add_argument("--det", action="store_true", help="额外训弱监督检测器以产出 mAP@0.5")
    parser.add_argument("--skip-build", action="store_true", help="不自动重建数据集")
    args = parser.parse_args(argv)

    # 没指定设备就自动挑
    if args.device is None:
        try:
            import torch

            args.device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print()
    print("=" * 76)
    print("  环境体检")
    print("=" * 76)
    ok, lines = check_environment()
    for line in lines:
        print(f"  {line}")
    print()

    if args.report:
        print(f"  可训练：{'是' if ok else '否'}")
        print()
        return 0 if ok else 1

    if not byd.DEFAULT_OUT.is_dir() and not args.skip_build:
        print("  训练集不存在，先构建：")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_yolo_dataset.py")],
            check=True,
        )

    if not ok:
        print("  环境不完整，无法训练。", file=sys.stderr)
        return 1

    weights = train_classification(args)

    if args.det:
        train_detection(args)

    # 用训练好的权重在测试集上评估
    print()
    print("  在测试集上评估…")
    from ultralytics import YOLO

    _console.install()  # ultralytics 导入时会把 stdout 改成 UTF-8，抢回来
    model = YOLO(str(weights))
    result = evaluate(model, [c.key for c in byd.CLASSES], byd.DEFAULT_OUT / "test")
    write_metrics(result, args, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
