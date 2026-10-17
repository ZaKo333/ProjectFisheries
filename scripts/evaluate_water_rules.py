#!/usr/bin/env python
"""M8 外部验证：在水产养殖池塘的实测数据上评测水质规则引擎。

跑两组验证，产出一份 Markdown 报告与一份机器可读的 JSON：

1. **阈值定级**（`runner`）：按行业阈值把窗口末点打成 正常/预警/高危，
   再和规则引擎的评分分档对照。
   ⚠ **这组的标签不是独立真值**——它是我们用 `water_risk.py` 的同一套阈值
   现算出来的。所以它量的是「两套阈值口径（最坏情况 vs 加权评分）的分歧」，
   **不是**引擎的准确率。报告里必须保留这条警告。
2. **溶氧趋势外推**（`do_trend`）：用过去的溶氧序列预测若干小时后的溶氧。
   真值是**真实发生的未来读数**，所以这一组是独立的。

用法::

    python scripts/evaluate_water_rules.py
    python scripts/evaluate_water_rules.py --limit-files 2 --limit-windows 200   # 快速冒烟
    python scripts/evaluate_water_rules.py --window-hours 1 --no-sensitivity

纪律：**不许把负结果包装成正面结论。** 如果引擎在某一组上输了，
结论段必须直说输了、输多少、以及为什么。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 让脚本可以直接跑（不必先设 PYTHONPATH）。
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 控制台编码兜底——详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import paths as _paths  # noqa: E402
from fisheries.evaluation import (  # noqa: E402
    AGGREGATE_MINUTES,
    DEFAULT_HISTORY_POINTS,
    DEFAULT_LEAD_POINTS,
    HISTORY_CHOICES,
    RANGES,
    WINDOW_HOURS,
    describe,
    evaluate,
    evaluate_do_trend,
    growth_coherence,
    iter_windows,
    load_ponds,
    pond_outcomes,
    render_report,
    run_engine,
    summarize_samples,
)

DEFAULT_MD = PROJECT_ROOT / "docs" / "M8_评测报告.md"
DEFAULT_JSON = PROJECT_ROOT / "docs" / "_m8_metrics.json"
DEFAULT_VISION = PROJECT_ROOT / "data" / "derived" / "metrics.json"

#: 窗口长度敏感性对照——只有这三个窗长会被扫。
SENSITIVITY_WINDOWS: tuple[float, ...] = (1.0, 3.0, 6.0)


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def load_vision_metrics(path: Path) -> dict | None:
    """读图像模态（病害识别）的留出集指标；没有就返回 None。

    这是**另一件东西**，不属于第 4/5 节那两组验证。读不到不算失败：
    报告第 8 节会写「未运行」，第 1~7 节的结论不受影响。
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _collect_windows(by_pond, window_hours: float):
    """按「不重叠窗口」切样本。步长恒等于窗长——样本之间必须互相独立。"""
    return list(iter_windows(by_pond, window_hours=window_hours, stride_hours=window_hours))


def _growth_by_pond(totals: dict) -> dict[str, float | None]:
    return {row["pond_id"]: row.get("growth_pct") for row in totals["per_file"]}


def _covered_samples(windows) -> int:
    return sum(w.n_points for w in windows)


# --------------------------------------------------------------------------- #
# 结论与局限（由实测数字生成，不写死）
# --------------------------------------------------------------------------- #


def build_conclusion(primary, do_trend, coherence) -> str:
    """一句话结论——按实测数字拼，赢就说赢、输就说输。"""
    macro = primary.confusion().macro()
    parts = [
        f"第一组（阈值定级，**标签不独立**）准确率 {macro.accuracy:.1%}、"
        f"κ {macro.kappa:.3f}、漏报率 {primary.late_prediction_rate():.1%}——"
        "这是两套阈值口径的分歧，不能当引擎准确率引用。"
    ]
    if do_trend is not None and do_trend.overall.n:
        overall = do_trend.overall
        gap = overall.improvement_pct
        if gap > 1.0:
            verdict = f"引擎优于持久性基线 {gap:.1f}%"
        elif gap < -1.0:
            verdict = f"引擎**劣于**持久性基线 {abs(gap):.1f}%"
        else:
            verdict = f"引擎与持久性基线基本打平（{gap:+.1f}%）"
        parts.append(
            f"第二组（溶氧趋势外推，真值独立）在 {overall.n} 个样本上{verdict}"
            f"（MAE {overall.engine_mae:.3f} vs {overall.persistence_mae:.3f}），"
            f"方向判对率 {overall.direction_accuracy:.1%}。"
        )
    else:
        parts.append("第二组（溶氧趋势外推）本次未运行。")
    if coherence is not None and coherence.n_pairs:
        # 别把 `coherence.verdict()` 整句抄进来——它自带一句「在 N 个池塘上…」，
        # 和这里的前缀叠起来会读成两句半。
        if coherence.rho < 0:
            reading = "负相关，标签与塘的实际产出方向一致"
        elif coherence.rho > 0:
            reading = "正相关，标签与塘的实际产出方向相反——这条要单独解释"
        else:
            reading = "无相关"
        parts.append(
            f"合理性检查：{coherence.n_pairs} 口塘的「高危占比 vs 体重增长」"
            f"Spearman ρ = {coherence.rho:+.3f}（{reading}）。"
        )
    return "".join(parts)


def build_limitations(primary, totals: dict, summary: dict, skipped: int, do_trend) -> list[str]:
    """本报告的限制——这部分不许删，删了报告就变成广告了。"""
    notes = [
        "**第一组的标签不独立。** 标签按 `water_risk.py` 的同一套阈值算出，"
        "与引擎同源；它衡量两套口径的分歧，不是与事实的符合度。",
        "**亚硝酸盐是盲区。** 这份数据只有硝酸盐（`nitrate`），引擎的一级指标 "
        "`nitrite_n` 填不进去，权重被重新归一化。真实塘口亚硝酸盐超标时本评测看不到。",
        "**氨氮与硝酸盐的量纲是推断的。** 数据标着 `(g/ml)`，但按 g/ml 读会得出"
        "「鱼在致死浓度里长胖几十倍」的矛盾，故按 µg/L 处理（÷1000）；"
        "唯一标 `(mg/L)` 的 IoTPond8 不换算。这条推断若有误，第一组的氨氮分档会整体偏移。",
        "**窗口末点取真值。** 引擎看到的就是窗口里的点，两边比的是同一件事；"
        "但这也意味着评测没有考察引擎对未来风险的提前量。",
        "**「正常」占多数会抬高准确率。** 标签分布见第 1 节，"
        "读准确率时必须连 κ 与逐类 P/R/F1 一起看。",
    ]
    if skipped:
        notes.append(
            f"**有 {skipped:,} 个聚合点没进入任何窗口**"
            f"（不足一个窗长、或点太少），它们不参与第一组评测。"
        )
    drop = int(totals.get("n_out_of_range", 0))
    if drop:
        # 量程从 `ponds.RANGES` 现取，别再在文案里手写一遍——
        # 之前这里写的是「溶氧 >25」，而代码里早就是 20 了。
        notes.append(
            f"**清洗掉了 {drop:,} 个越界/哨兵读数**（温度低于 "
            f"{RANGES['water_temp'][0]:g} ℃、pH 不在 "
            f"{RANGES['ph'][0]:g}～{RANGES['ph'][1]:g}、溶氧超过 "
            f"{RANGES['dissolved_oxygen'][1]:g} mg/L、氨氮换算后超过 "
            f"{RANGES['ammonia_n'][1]:g} mg/L 等）。"
            "这些不是真实超标，是传感器缺测编码或单位错误。"
        )
    if do_trend is not None:
        notes.append(
            "**第二组只覆盖引擎的预测成分。** 溶氧的线性外推是引擎唯一的预测项，"
            "这一组的结果不能外推到阈值定级或整个引擎。"
        )
    return notes


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_water_rules.py",
        description="在 aquaponics_pond（IoTpond*.csv）上评测水质规则引擎。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="池塘 CSV 所在目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_MD, help="Markdown 报告输出路径")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="机器可读指标输出路径")
    parser.add_argument(
        "--vision-metrics",
        type=Path,
        default=DEFAULT_VISION,
        help="图像模态留出集指标（data/derived/metrics.json）",
    )
    parser.add_argument("--species", default="鲤科", help="引擎鱼种档（鲤科/罗非鱼/虾蟹）")
    parser.add_argument("--horizon-hours", type=int, default=72, help="引擎预测视界（小时）")
    parser.add_argument("--window-hours", type=float, default=WINDOW_HOURS, help="窗口长度（小时）")
    parser.add_argument(
        "--aggregate-minutes", type=int, default=AGGREGATE_MINUTES, help="原始读数聚合粒度（分钟）"
    )
    parser.add_argument("--limit-files", type=int, default=None, help="只读前 N 个池塘 CSV")
    parser.add_argument(
        "--limit-windows", type=int, default=None, help="每组评测最多调用引擎的次数（冒烟用）"
    )
    parser.add_argument(
        "--history-points",
        type=int,
        choices=HISTORY_CHOICES,
        default=DEFAULT_HISTORY_POINTS,
        help="第二组的回看点数",
    )
    parser.add_argument(
        "--lead-points", type=int, default=DEFAULT_LEAD_POINTS, help="第二组的预测步数"
    )
    parser.add_argument("--no-do-trend", action="store_true", help="跳过第二组（溶氧趋势）")
    parser.add_argument("--no-sensitivity", action="store_true", help="跳过窗长敏感性对照")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)

    paths = None
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"[错误] 数据目录不存在：{data_dir}")
            return 1
        paths = sorted(
            (p for p in data_dir.glob("*.csv") if p.is_file()),
            key=lambda p: (len(p.name), p.name.lower()),
        )
        if not paths:
            print(f"[错误] {data_dir} 里没有 CSV")
            return 1

    print("读取池塘数据……")
    by_pond, totals = load_ponds(
        paths, aggregate_minutes=args.aggregate_minutes, limit_files=args.limit_files
    )
    # 数据一进来就把本机绝对路径压成占位符。报告与 JSON 两条出口都在下游，
    # 在这里收口，两条都不会漏（占位符定义见 fisheries.paths）。
    totals["per_file"] = _paths.stabilize_tree(totals["per_file"])
    if not by_pond:
        print("[错误] 没有读到任何数据")
        return 1
    summary = summarize_samples(by_pond)
    print(
        f"  {totals['n_files']} 个文件 / {totals['n_lines']:,} 行 → "
        f"{summary['n_samples']:,} 个聚合点（{summary['n_ponds']} 口塘）"
    )
    print(
        f"  清洗：缺测丢弃 {totals['n_dropped_missing_required']:,}、"
        f"越界/哨兵 {totals['n_out_of_range']:,}、时间戳不可解析 {totals['n_bad_timestamp']:,}"
    )
    print(f"  标签分布：{summary['label_counts']}")

    growth = _growth_by_pond(totals)
    coherence = growth_coherence(pond_outcomes(by_pond, growth))

    # ---- 第一组：阈值定级 -------------------------------------------------
    primary_windows = _collect_windows(by_pond, args.window_hours)
    if not primary_windows:
        print("[错误] 没有切出任何窗口（数据跨度不足一个窗长？）")
        return 1
    skipped = max(0, summary["n_samples"] - _covered_samples(primary_windows))
    print(
        f"\n[第一组] {len(primary_windows):,} 个 {args.window_hours:g} 小时窗口，"
        f"调用引擎（上限 {args.limit_windows}）……"
    )
    samples, engine_stats = run_engine(
        primary_windows,
        species=args.species,
        horizon_hours=args.horizon_hours,
        limit=args.limit_windows,
    )
    primary = evaluate(
        samples,
        name="阈值定级（主口径）",
        mode="trend",
        window_hours=args.window_hours,
        engine_stats=engine_stats,
    )
    macro = primary.confusion().macro()
    print(
        f"  可用 {len(primary.usable):,} / 失败 {engine_stats['n_failed']}，"
        f"{engine_stats['ms_per_call']:.1f} ms/次"
    )
    print(
        f"  准确率 {macro.accuracy:.1%}　κ {macro.kappa:.3f}　"
        f"ρ {primary.score_spearman():.3f}　漏报 {primary.late_prediction_rate():.1%}"
    )

    # ---- 窗长敏感性 -------------------------------------------------------
    sensitivity = []
    if not args.no_sensitivity:
        for hours in SENSITIVITY_WINDOWS:
            if abs(hours - args.window_hours) < 1e-9:
                continue
            windows = _collect_windows(by_pond, hours)
            if not windows:
                continue
            print(f"[敏感性] 窗长 {hours:g} 小时，{len(windows):,} 个窗口……")
            rows, stats = run_engine(
                windows,
                species=args.species,
                horizon_hours=args.horizon_hours,
                limit=args.limit_windows,
            )
            sensitivity.append(
                evaluate(
                    rows,
                    name=f"窗长 {hours:g} 小时",
                    mode="trend",
                    window_hours=hours,
                    engine_stats=stats,
                )
            )

    # ---- 第二组：溶氧趋势外推 ---------------------------------------------
    do_trend = None
    if not args.no_do_trend:
        print(
            f"\n[第二组] 溶氧趋势外推：回看 {args.history_points} 点 → "
            f"预测 {args.lead_points} 步之后……"
        )
        do_trend = evaluate_do_trend(
            by_pond,
            step_minutes=args.aggregate_minutes,
            lead_points=args.lead_points,
            history_points=args.history_points,
        )
        overall = do_trend.overall
        if overall.n:
            print(
                f"  {overall.n:,} 个样本：引擎 MAE {overall.engine_mae:.3f} vs "
                f"持久性 {overall.persistence_mae:.3f}（{overall.improvement_pct:+.1f}%），"
                f"方向 {overall.direction_accuracy:.1%}"
            )

    # ---- 图像模态（附录） -------------------------------------------------
    vision = _paths.stabilize_tree(load_vision_metrics(Path(args.vision_metrics)))
    print(f"\n[附录] 图像模态指标：{'已载入' if vision else '未运行（文件不存在）'}")

    # ---- 出报告 -----------------------------------------------------------
    generated_at = started.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    report = render_report(
        data_stats=totals,
        label_distribution=summary,
        primary=primary,
        sensitivity=sensitivity,
        skipped=skipped,
        generated_at=generated_at,
        species=args.species,
        horizon_hours=args.horizon_hours,
        coherence=coherence,
        do_trend=do_trend,
        vision_metrics=vision,
        conclusion=build_conclusion(primary, do_trend, coherence),
        limitations=build_limitations(primary, totals, summary, skipped, do_trend),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {out}（{len(report):,} 字符）")

    payload = {
        "generated_at": generated_at,
        "metric_kind": "rule_consistency",
        "metric_label": "规则一致性准确率（引擎输出 vs 离线阈值脚本，不是与事实相符的准确率）",
        "protocol": {
            "species": args.species,
            "horizon_hours": args.horizon_hours,
            "window_hours": args.window_hours,
            "stride_hours": args.window_hours,
            "aggregate_minutes": args.aggregate_minutes,
            "limit_files": args.limit_files,
            "limit_windows": args.limit_windows,
            "history_points": args.history_points,
            "lead_points": args.lead_points,
        },
        "data": {k: v for k, v in totals.items() if k != "per_file"},
        "data_per_file": totals["per_file"],
        "summary": summary,
        "coherence": {
            "n_ponds": len(coherence.ponds),
            "n_pairs": coherence.n_pairs,
            "rho": coherence.rho,
            "verdict": coherence.verdict(),
        },
        "skipped_samples": skipped,
        "primary": primary.to_dict(),
        "sensitivity": [run.to_dict() for run in sensitivity],
        "do_trend": (
            None
            if do_trend is None
            else {
                "overall": do_trend.overall.__dict__,
                "by_history": {
                    str(k): v.__dict__ for k, v in sorted(do_trend.by_history.items())
                },
                "step_minutes": do_trend.step_minutes,
                "lead_hours": do_trend.lead_hours,
                "n_series": do_trend.n_series,
                "n_ponds": do_trend.n_ponds,
                "verdict": do_trend.verdict(),
                "sources": list(do_trend.sources),
            }
        ),
        "vision": vision,
        "distributions": {
            "engine_score": describe([s.risk_score for s in primary.usable if s.ok]),
        },
    }
    dest = Path(args.json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已写入 {dest}")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n完成，用时 {elapsed:.1f} 秒。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
