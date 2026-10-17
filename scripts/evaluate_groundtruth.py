"""跑 M8 的**人工标签评测**：拿人判的等级当真值，量引擎判得准不准。

    python scripts/evaluate_groundtruth.py                 # 读 data/eval_groundtruth/
    python scripts/evaluate_groundtruth.py --min-labels 50
    python scripts/evaluate_groundtruth.py --out %TEMP%/gt --dry-run

与 `scripts/evaluate_water_rules.py` 的区别，一句话：
**那个量的是「两套阈值实现是否一致」，这个量的是「引擎判得对不对」。**

前者 71.9% 的标签是拿同一套阈值现算的，自己跟自己比，只能证明代码没写错；
真正的准确率必须有一条独立的人工真值，就存在 `data/eval_groundtruth/labels.csv`。

人工标注不足 `--min-labels`（默认 20）条时**不出任何指标**，退出码 2，
状态记 `pending_labels`——宁可显示待标注，也不拿几条样本算个漂亮数字。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fisheries import console as _console  # noqa: E402 - 先 sys.path 再导入

_console.install()

from fisheries.evaluation.groundtruth import (  # noqa: E402 - 同上
    DEFAULT_DIR,
    MIN_LABELS,
    evaluate,
    read_sample,
)

DEFAULT_JSON = PROJECT_ROOT / "docs" / "_m8_groundtruth.json"
RULES_JSON = PROJECT_ROOT / "docs" / "_m8_metrics.json"


def _consistency_number() -> dict | None:
    """读一眼第一组（规则一致性）的头部数字，用来在输出里对照。"""
    if not RULES_JSON.exists():
        return None
    try:
        payload = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    primary = payload.get("primary") or {}
    macro = primary.get("macro") or {}
    return {
        "n": macro.get("n"),
        "accuracy": macro.get("accuracy"),
        "kappa": macro.get("kappa"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用人工标注的真值集评测 M8（独立真值，不是规则一致性）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="人工标注集目录")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="指标写到这里（相对路径按仓库根）")
    parser.add_argument("--species", default="鲤科", help="判级用的鱼种档位")
    parser.add_argument("--horizon-hours", type=int, default=72, help="引擎的趋势外推视野")
    parser.add_argument("--min-labels", type=int, default=MIN_LABELS, help="少于这么多条不出指标")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    directory: Path = args.dir if args.dir.is_absolute() else (PROJECT_ROOT / args.dir)
    dest: Path = args.json if args.json.is_absolute() else (PROJECT_ROOT / args.json)

    print("=" * 60)
    print("M8 人工标签评测（独立真值）")
    print("=" * 60)
    print(f"  标注目录   {directory}")
    print(f"  最少条数   {args.min_labels}")
    print()

    try:
        rows = read_sample(directory)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print("  先跑：python scripts/make_groundtruth_sample.py", file=sys.stderr)
        return 1

    result = evaluate(
        rows, species=args.species, horizon_hours=args.horizon_hours, min_labels=args.min_labels
    )

    print(f"抽样 {result['n_sampled']} 条，已标注 {result['n_labeled']} 条"
          f"（还缺 {result['n_missing']} 条）")

    if result["status"] == "pending_labels":
        print()
        print(_console.mark(False) + f" {result['reason']}")
        print()
        print("这一栏现在是空的——这是设计如此，不是 bug：")
        print("  人工标签必须由人看过读数再填，脚本不替人做判断。")
        if not args.dry_run:
            dest.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  状态已写进 {dest}")
        return 2

    consistency = _consistency_number()
    print()
    print(result["markdown"])
    print()
    print(f"  人工标签准确率        {result['accuracy']:.4f}  ({result['n_scored']} 条已判)")
    print(f"  Cohen's κ            {result['kappa']:.4f}")
    print(f"  宏平均 P / R / F1    {result['macro_precision']:.4f} / "
          f"{result['macro_recall']:.4f} / {result['macro_f1']:.4f}")
    if result["high_risk_recall"] is not None:
        print(f"  高危召回             {result['high_risk_recall']:.4f}"
              f"（漏判 {result['high_risk_missed']} 条）")
    if result["false_alarm_rate"] is not None:
        print(f"  正常误报为高危        {result['false_alarm_rate']:.4f}")
    if result["adjacent_agreement"] is not None:
        print(f"  相邻等级一致率        {result['adjacent_agreement']:.4f}")
    if result["annotators"]:
        print(f"  标注人               {'、'.join(result['annotators'])}")
    if consistency:
        print()
        print("对照（不是同一个东西）：")
        print(f"  规则一致性准确率      {consistency['accuracy']:.4f}"
              f"（{consistency['n']} 条，已标标签 = 离线阈值脚本输出）")
        print("  ↑ 它证明「代码没写错、两套实现一致」，不能当准确率引用。")

    if not args.dry_run:
        dest.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print(f"指标已写入 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
