"""在清洗后的「鱼体 + 病灶」数据集上训练 YOLOv8-seg，并输出逐类 mAP。

这份数据集与 `build_yolo_dataset.py` 造出来的 4 分类集**不是一回事**：

- 4 分类集是「整张图属于哪一类」，框是弱监督的全图框，mAP 恒等 1.0，测不出定位能力；
- 这份是**真实实例多边形**（鱼体一个多边形、病灶一个多边形），mAP 是真的定位指标。

因此这里的指标可以和 README §12.4 里那句「不衡量病灶定位能力」对照着看。

产物：
- ``models/yolov8-fish-lesion-seg.pt``（best 权重）
- ``data/derived/lesion_metrics.json``（逐类 mAP、混淆信息、训练参数）

用法（在工程根目录）::

    python scripts/train_lesion_seg.py --epochs 80
    python scripts/train_lesion_seg.py --epochs 5 --imgsz 320     # 冒烟
    python scripts/train_lesion_seg.py --eval-only                # 只评估现有权重
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
for _p in (str(_SRC), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fisheries import console as _console  # noqa: E402

_console.install()

DEFAULT_DATA = PROJECT_ROOT / "data" / "derived" / "fish_lesion_seg" / "data.yaml"
WEIGHTS_OUT = PROJECT_ROOT / "models" / "yolov8-fish-lesion-seg.pt"
RUNS_DIR = PROJECT_ROOT / "data" / "derived" / "runs"
METRICS_OUT = PROJECT_ROOT / "data" / "derived" / "lesion_metrics.json"

SEG_MODEL = "yolov8n-seg.pt"

#: 与 train_yolo_fish.py 保持同一组镜像前缀：本机直连 github.com 会被重置。
_ASSET_TAG = "v8.3.0"
_ASSET_BASE = f"https://github.com/ultralytics/assets/releases/download/{_ASSET_TAG}"
_MIRROR_PREFIXES: tuple[str, ...] = ("https://ghproxy.net/", "https://gh-proxy.com/", "")


def _find_pretrained(name: str) -> Path | None:
    """在 models/pretrained 下找同名权重（ultralytics 也会认工程根目录）。"""
    for base in (PROJECT_ROOT / "models" / "pretrained", PROJECT_ROOT):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def ensure_pretrained(name: str) -> Path:
    """返回本地权重路径；缺失时按镜像顺序下载到 models/pretrained。"""
    found = _find_pretrained(name)
    if found:
        return found
    import urllib.request

    dest = PROJECT_ROOT / "models" / "pretrained" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for prefix in _MIRROR_PREFIXES:
        url = f"{prefix}{_ASSET_BASE}/{name}"
        try:
            print(f"  下载 {url}")
            with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            if dest.stat().st_size > 1_000_000:
                return dest
            dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 —— 逐个镜像试，全失败再报
            last = exc
            print(f"    失败：{type(exc).__name__}: {exc}")
            dest.unlink(missing_ok=True)
    raise SystemExit(f"无法下载 {name}：{last}\n请手动放到 {dest}")


def pick_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def train(args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    weights = ensure_pretrained(args.model)
    print(f"  预训练权重 : {weights}")
    print(f"  数据集     : {args.data}")
    print(f"  设备       : {args.device}    imgsz={args.imgsz}  epochs={args.epochs}  batch={args.batch}")
    print()

    model = YOLO(str(weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        seed=42,
        plots=False,
        val=True,
        verbose=True,
    )
    best = Path(model.trainer.best)
    return best


def evaluate(weights: Path, data: Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=imgsz, device=device, plots=False, verbose=False)
    names = metrics.names if hasattr(metrics, "names") else {}
    box = getattr(metrics, "box", None)
    seg = getattr(metrics, "seg", None)

    def _per_class(m) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        if m is None:
            return out
        try:
            p, r = m.p.tolist(), m.r.tolist()
            ap50, ap = m.ap50.tolist(), m.ap.tolist()
        except AttributeError:
            return out
        for i, name in names.items():
            if i < len(ap50):
                out[str(name)] = {
                    "precision": round(float(p[i]), 4),
                    "recall": round(float(r[i]), 4),
                    "mAP50": round(float(ap50[i]), 4),
                    "mAP50-95": round(float(ap[i]), 4),
                }
        return out

    result = {
        "weights": str(weights),
        "data": str(data),
        "imgsz": imgsz,
        "box": {
            "mAP50": round(float(box.map50), 4) if box is not None else None,
            "mAP50-95": round(float(box.map), 4) if box is not None else None,
            "per_class": _per_class(box),
        },
        "seg": {
            "mAP50": round(float(seg.map50), 4) if seg is not None else None,
            "mAP50-95": round(float(seg.map), 4) if seg is not None else None,
            "per_class": _per_class(seg),
        },
    }
    return result


def print_metrics(metrics: dict) -> None:
    print()
    print("=" * 52)
    print("  定位指标（真值是多边形，不是弱监督全图框）")
    print("=" * 52)
    for key, title in (("box", "检测 mAP（框）"), ("seg", "分割 mAP（掩膜）")):
        block = metrics[key]
        if block["mAP50"] is None:
            print(f"  {title:<14}: 该任务无输出")
            continue
        print(f"  {title:<14}: mAP50={block['mAP50']:.4f}   mAP50-95={block['mAP50-95']:.4f}")
        for name, row in block["per_class"].items():
            print(
                f"      {name:<8} P={row['precision']:.3f}  R={row['recall']:.3f}  "
                f"AP50={row['mAP50']:.3f}  AP50-95={row['mAP50-95']:.3f}"
            )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_lesion_seg.py",
        description="训练鱼体+病灶实例分割模型",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help=f"data.yaml 路径，默认 {DEFAULT_DATA}")
    parser.add_argument("--model", default=SEG_MODEL, help=f"预训练权重名，默认 {SEG_MODEL}")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数，默认 80")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长，默认 640")
    parser.add_argument("--batch", type=int, default=16, help="批大小，默认 16")
    parser.add_argument("--device", default="auto", help="训练设备：auto / 0 / cpu")
    parser.add_argument("--name", default="fish_lesion_seg", help="runs 下的实验名")
    parser.add_argument("--eval-only", action="store_true", help="不训练，只评估现有 models 权重")
    parser.add_argument("--no-save", action="store_true", help="不把 best.pt 复制到 models/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.device = pick_device() if args.device == "auto" else args.device

    if not args.data.is_file():
        print(f"{_console.mark(False)} data.yaml 不存在：{args.data}")
        print("  先跑：python scripts/build_lesion_dataset.py")
        return 1

    if args.eval_only:
        if not WEIGHTS_OUT.is_file():
            print(f"{_console.mark(False)} 权重不存在：{WEIGHTS_OUT}")
            return 1
        metrics = evaluate(WEIGHTS_OUT, args.data, args.imgsz, args.device)
    else:
        best = train(args)
        print(f"\n  训练完成，best = {best}")
        if not args.no_save:
            WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, WEIGHTS_OUT)
            print(f"  {_console.mark(True)} 已复制到 {WEIGHTS_OUT}")
        metrics = evaluate(WEIGHTS_OUT if not args.no_save else best, args.data, args.imgsz, args.device)

    metrics["epochs"] = args.epochs
    metrics["batch"] = args.batch
    metrics["pretrained"] = args.model
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print_metrics(metrics)
    print(f"  指标已写出：{METRICS_OUT}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
