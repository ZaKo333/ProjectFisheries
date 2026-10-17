"""抽一批窗口，给人来标等级——M8 的「人工标签准确率」就靠它。

    python scripts/make_groundtruth_sample.py                # 抽 100 条到 data/eval_groundtruth/
    python scripts/make_groundtruth_sample.py --n 60 --dry-run
    python scripts/make_groundtruth_sample.py --out %TEMP%/gt --force

产出三件：

- `labels.csv`——抽样清单。**`human_level` / `annotator` / `note` 三列留空等人填**；
- `windows.json`——每条样本的原始读数，评测时原样喂给引擎；
- `manifest.json`——抽样参数（条数、随机种子、窗长、抽自哪些文件、生成时间）。

## 标注怎么做

打开 `labels.csv`，逐条看同一窗口在 `windows.json` 里的读数，按下面的口径判级：

- **正常**——五项指标都在「适宜」区间，且没有恶化趋势；
- **预警**——有一项进入「不适」区间，或虽在适宜区间但明显在恶化；
- **高危**——有一项进入「危险」区间（溶氧 < 1.5、氨氮 > 1.0、亚硝酸盐 > 0.5、
  pH 越出 5.5–9.5 等），或已明显出现浮头/死亡的临界状态。

`free_ammonia_mg_l` 是末点按 Emerson 公式折算出的**非离子氨**，参考用：
超过 0.02 mg/L 即已越 GB 11607-1989 的限值，应判高危。

判完在 `annotator` 填自己的名字（用于报告里写「由谁标注」），`note` 可写理由。
**不要留空又填名字**，也不要把三个等级写成别的词——脚本会报错。

填好之后跑 `python scripts/evaluate_groundtruth.py`。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fisheries import console as _console  # noqa: E402 - 先 sys.path 再导入

_console.install()

from fisheries.evaluation.groundtruth import (  # noqa: E402 - 同上
    DEFAULT_DIR,
    DEFAULT_N,
    DEFAULT_SEED,
    HUMAN_COLUMNS,
    MIN_LABELS,
    check_isolation,
    read_sample,
    sample_windows,
    write_sample,
)
from fisheries.evaluation.ponds import AGGREGATE_MINUTES, load_ponds  # noqa: E402 - 同上

DEFAULT_README = """# M8 人工标签评测集

这个目录是**独立真值集**：抽样脚本只负责挑样本，等级必须由人来判。
它不参与任何训练，也不和 `data/derived/`、`data/datasets/` 共用文件。

## 文件

| 文件 | 作用 | 由谁写 |
|---|---|---|
| `labels.csv` | 抽样清单 + 人工标注列 | 脚本生成清单，**人填三列** |
| `windows.json` | 每条样本的原始读数 | 脚本生成，不要改 |
| `manifest.json` | 抽样参数与条数 | 脚本生成 |

## 判级口径

按窗口**末点**的五项指标判：

- **正常**：五项都在「适宜」区间（溶氧 ≥ 5、氨氮 ≤ 0.2、亚硝酸盐 ≤ 0.1、pH 6.5–8.5、水温 20–30）；
- **预警**：有一项进入「不适」区间，或虽在适宜区间但窗口内明显恶化；
- **高危**：有一项进入「危险」区间（溶氧 < 1.5、氨氮 > 1.0、亚硝酸盐 > 0.5、pH 越出 5.5–9.5）。

`free_ammonia_mg_l` 是按 Emerson 公式由总氨氮 + pH + 水温折算的非离子氨，
**超过 0.02 mg/L 即已越 GB 11607-1989 限值，应判高危**（哪怕总氨氮还没到 1.0）。

## 两条硬规则

1. **不要编标签**。填不出来就留空，指标会显示「待标注」，这比一个假数字诚实。
2. 标完在 `annotator` 填名字。报告会写明这批评测由谁标注、标了多少条。

至少 `{min_labels}` 条才会出指标；建议 50–100 条，且覆盖低氧、氨氮偏高、
正常三类各若干。抽样脚本已按溶氧分层，请勿自行删掉难判的样本。
"""


def _filled(path: Path) -> int:
    """数一下 `labels.csv` 里已经填了多少条人工等级。"""
    if not path.exists():
        return 0
    try:
        rows = read_sample(path.parent)
    except (ValueError, FileNotFoundError, KeyError):
        return 0
    return sum(1 for row in rows if row.labeled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="抽一批窗口供人工标注（M8 人工标签评测的抽样步骤）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="抽多少条")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（同数据同种子结果一致）")
    parser.add_argument("--window-hours", type=float, default=3.0, help="每个样本的窗口长度（小时）")
    parser.add_argument(
        "--stride-hours", type=float, default=None, help="窗口步长，默认等于窗长（不重叠）"
    )
    parser.add_argument(
        "--min-gap-hours", type=float, default=24.0, help="同一口塘两条样本的最小间隔，避免高度相似"
    )
    parser.add_argument("--aggregate-minutes", type=int, default=AGGREGATE_MINUTES, help="聚合粒度（分钟）")
    parser.add_argument("--limit-files", type=int, default=None, help="只读前 N 个数据文件（调试用）")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--force", action="store_true", help="覆盖已有清单（已填标注时仍会拒绝）")
    parser.add_argument("--dry-run", action="store_true", help="只打印抽样结果，不写文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.n < MIN_LABELS:
        print(
            f"⚠ 只抽 {args.n} 条，少于出指标所需的 {MIN_LABELS} 条；"
            f"标注完也只会显示「待标注」。建议 --n {max(50, MIN_LABELS)} 以上。"
        )

    out: Path = args.out if args.out.is_absolute() else (PROJECT_ROOT / args.out)
    check_isolation(out)

    print("=" * 60)
    print("M8 人工标签评测集 · 抽样")
    print("=" * 60)
    print(f"  抽样条数   {args.n}")
    print(f"  随机种子   {args.seed}")
    print(f"  窗口长度   {args.window_hours} 小时（步长 {args.stride_hours or args.window_hours}）")
    print(f"  同塘间隔   ≥ {args.min_gap_hours} 小时")
    print(f"  输出目录   {out}")
    print()

    by_pond, stats = load_ponds(
        None, aggregate_minutes=args.aggregate_minutes, limit_files=args.limit_files
    )
    if not by_pond:
        print("✗ 没有读到任何池塘数据，先跑 python scripts/fetch_datasets.py pull", file=sys.stderr)
        return 1

    print(
        f"读入 {stats.get('n_files', len(by_pond))} 个文件 / "
        f"{stats.get('n_samples', 0)} 个聚合点 / {len(by_pond)} 口塘"
    )
    rows = sample_windows(
        by_pond,
        n=args.n,
        seed=args.seed,
        window_hours=args.window_hours,
        stride_hours=args.stride_hours,
        min_gap_hours=args.min_gap_hours,
    )
    if not rows:
        print("✗ 抽不到样本（窗口数不足或间隔限制太严），试试降低 --min-gap-hours", file=sys.stderr)
        return 1

    per_pond = Counter(row.pond_id for row in rows)
    oxygen = Counter(
        "缺测"
        if row.values.get("dissolved_oxygen") is None
        else "低氧(<3)"
        if row.values["dissolved_oxygen"] < 3
        else "偏低(<5)"
        if row.values["dissolved_oxygen"] < 5
        else "充足(≥5)"
        for row in rows
    )
    print(f"实抽 {len(rows)} 条：")
    print("  按塘：" + "、".join(f"{k} {v} 条" for k, v in sorted(per_pond.items())))
    print("  按溶氧：" + "、".join(f"{k} {v} 条" for k, v in oxygen.most_common()))

    if args.dry_run:
        print()
        print("（--dry-run：未写文件）")
        for row in rows[:5]:
            print(
                f"  {row.window_id} {row.pond_id} {row.started_at:%Y-%m-%d %H:%M} "
                f"DO={row.values.get('dissolved_oxygen')} "
                f"NH3-N={row.values.get('ammonia_n')} "
                f"pH={row.values.get('ph')}"
            )
        return 0

    filled = _filled(out / "labels.csv")
    if filled and not args.force:
        print()
        print(
            f"✗ {out / 'labels.csv'} 已有 {filled} 条人工标注，拒绝覆盖"
            "（覆盖会丢掉别人刚标的等级）。",
            file=sys.stderr,
        )
        print("  要重新抽，先把整个目录移走；--force 只覆盖空清单。", file=sys.stderr)
        return 1

    paths = write_sample(
        rows,
        out,
        meta={
            "n_requested": args.n,
            "seed": args.seed,
            "window_hours": args.window_hours,
            "stride_hours": args.stride_hours or args.window_hours,
            "min_gap_hours": args.min_gap_hours,
            "aggregate_minutes": args.aggregate_minutes,
            "ponds": sorted(per_pond),
            "source": "data/datasets/wq_ponds（load_ponds 默认目录）",
        },
    )

    readme = out / "README.md"
    if not readme.exists():
        readme.write_text(
            DEFAULT_README.format(min_labels=MIN_LABELS), encoding="utf-8", newline="\n"
        )

    print()
    print(_console.mark(True) + f" 已写出 {paths['labels']}")
    print(f"              {paths['windows']}")
    print(f"              {paths['manifest']}")
    print()
    print(f"下一步：打开 {paths['labels']}，把 {len(rows)} 行的")
    print(f"  {', '.join(HUMAN_COLUMNS)} 三列填上（等级填 正常 / 预警 / 高危）。")
    print("填完跑：python scripts/evaluate_groundtruth.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
