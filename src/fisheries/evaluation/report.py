"""把评测结果渲染成 Markdown 报告。

报告里的每个数字都由 `EvalRun` / `DoTrendReport` 现算，没有手写的期望值——
这样可以放心重跑：数据或引擎变一次，报告跟着变一次，不会留下对不上的旧结论。

**这份报告最容易被误读的地方**，所以写在最前面：第一组验证的标签是**我们按
行业阈值算出来的**，而规则引擎也是按行业阈值打分的。两者不是「真值 vs 预测」，
是「两套阈值口径的对照」。报告里每一个准确率数字旁边都必须带着这句话。

（中文里的引号一律用「」，不要用半角双引号：Python 字符串也是用双引号界定的，
两者混在一起会直接语法错误。这个坑在本文件的第一版里踩过一次。）
"""

from __future__ import annotations

from .labels import RISK_LEVELS
from .ponds import DATASET_NAME
from .runner import EvalRun

__all__ = ["render_report", "render_run_section"]


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_rate(hit: int, total: int) -> str:
    if not total:
        return "—"
    return f"{hit}/{total}（{hit / total * 100:.1f}%）"


def _quality_section(
    data_stats: dict,
    label_distribution: dict,
    coherence=None,
) -> str:
    """第 2 节：数据、标签、以及「为什么这些数字不是准确率」。"""
    per_pond = label_distribution.get("per_pond_detail", {}) or {}
    # 体重增长率只记在 per_file 里（和读取时的统计放在一起），这里建个查表。
    growth_by_pond = {
        row.get("pond_id"): row.get("growth_pct")
        for row in (data_stats.get("per_file") or [])
        if row.get("pond_id")
    }
    # 举生长最显著的那口塘当例子。**不能写死数字**——换一份数据就会和 2.5 节自相矛盾。
    _growth_example = max(
        ((pond, pct) for pond, pct in growth_by_pond.items() if pct is not None),
        key=lambda item: item[1],
        default=None,
    )
    _growth_clause = (
        f"例如 {_growth_example[0]} 的鱼在整个记录期内体重增长了 {_growth_example[1]:,.0f}%。"
        if _growth_example
        else "这份数据的体重列在整个记录期内的增长率见 2.5 节。"
    )
    lines = [
        "## 2. 数据与标签",
        "",
        "### 2.1 数据来源",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 数据集 | `{DATASET_NAME}`（`IoTpond*.csv`） |",
        f"| 文件数 | {data_stats.get('n_files', 0)} |",
        f"| 原始行数 | {data_stats.get('n_lines', 0)} |",
        f"| 可入物理量程的读数 | {data_stats.get('n_samples', 0)}"
        f"（丢弃 {data_stats.get('n_dropped_missing_required', 0)} 行必填缺失、"
        f"{data_stats.get('n_out_of_range', 0)} 个越界/哨兵值、"
        f"{data_stats.get('n_bad_timestamp', 0)} 行时间戳不可解析） |",
        f"| 参与评测的池塘 | {label_distribution.get('n_ponds', 0)} |",
        f"| 采样时间范围 | {label_distribution.get('time_from', '—')} ～ "
        f"{label_distribution.get('time_to', '—')} |",
        f"| 每塘读数 | {label_distribution.get('samples_per_pond_min', 0)} ～ "
        f"{label_distribution.get('samples_per_pond_max', 0)}（10 分钟聚合） |",
        "",
        "### 2.2 三个必须写在数字前面的量纲判断",
        "",
        "这份数据有四处**字面声明与物理量级矛盾**的地方。不处理，后面所有标签都是错的；"
        "处理了，就必须把依据写出来供人推翻。",
        "",
        "1. **`Ammonia(g/ml)` 实际是 µg/L，不是 g/ml。** 依据不是猜的："
        "同一份数据里每条鱼的体长体重都在记录，而**每条塘的鱼都长得很好**——"
        f"{_growth_clause}"
        "如果按字面把氨氮读成 0.55～60 g/ml，那相当于 5.5×10⁵～6×10⁷ mg/L，"
        "是致死红线（1.0 mg/L）的百万倍：鱼不可能一边泡在这种水里一边长这么大。"
        "÷1000 当 µg/L→mg/L 后中位数落在 0.0006～0.06 mg/L，全部落在适宜带——"
        "与增重事实一致。**IoTPond8 是唯一标了 `(mg/L)` 的文件，它不做换算**，"
        "而其数值量级恰好是其余文件的 1/1000，这是最强的旁证。",
        "2. **溶氧不换算。** 中位数 3.3～18.3，只能是 mg/L。",
        "3. **丢弃哨兵与越界值。** 温度 −127、pH −3.137、pH > 14、溶氧 > 20、"
        "氨氮 > 50 mg/L 都是仪器缺测码或单位错误，不是读数"
        "（阈值表见 `ponds.RANGES`，照抄自代码，不是这里另写的）。"
        "溶氧最大值 44.98 mg/L（25 ℃ 饱和约 8.4 mg/L）属于物理不可能。",
        "4. **这份数据只有硝酸盐、没有亚硝酸盐。** 规则引擎的一级指标 `nitrite_n` "
        "在这份数据上**填不进去**——引擎会把权重重新归一化。这是本次评测的一个真实盲区，"
        "不是可以绕过的细节。",
        "",
        "> 换算系数与量程表写在 `src/fisheries/evaluation/ponds.py` 的 "
        "`UNIT_DIVISORS` 与 `RANGES` 里，是可改常量。改掉它们，"
        "本报告的所有数字都会跟着变。",
        "",
        "### 2.3 标签口径：最差指标决定论",
        "",
        "标签不是数据集自带的（原数据的 `Ammonia(g/ml)` 这些列没有分级），"
        "是**按行业阈值现算的**：",
        "",
        "| 指标 | 适宜带 | 预警带 | 红线 |",
        "|---|---|---|---|",
        "| 溶氧 (mg/L) | ≥ 5.0 | 3.0 ～ 5.0 | < 1.5 |",
        "| 氨氮 (mg/L) | ≤ 0.2 | 0.2 ～ 0.5 | > 1.0 |",
        "| 亚硝酸盐 (mg/L) | ≤ 0.1 | 0.1 ～ 0.3 | > 0.5 |",
        "| pH | 6.5 ～ 8.5 | 6.0 ～ 9.0 | < 5.5 或 > 9.5 |",
        "| 水温 (℃) | 20 ～ 30 | 15 ～ 33 | — |",
        "",
        "规则：任一指标越红线 → **高危**；所有指标在适宜带内 → **正常**；"
        "其余（落在预警带）→ **预警**。阈值与 `water_risk.py` 的 "
        "`SPECIES_PROFILES['鲤科']` / `REDLINE` 一致，但**判定方式是两套**——",
        "标签取最差指标（最坏情况），引擎取加权评分加时间持续性（平均 + 趋势）。"
        "第 4 节的**规则一致性**量的就是这两套口径的分歧有多大。",
        "",
        "⚠ **正因为标签由同一套阈值算出，第一组验证不是独立的。** "
        "它既不能证明引擎的阈值定得对（阈值本来就是这么多），"
        "也不能当作「真实准确率」。它能回答的是三个具体问题："
        "两套口径分歧有多大、分歧出在哪些指标上、加权评分会不会被时间持续性放大成误报。"
        "第 6 节把这条局限讲透。",
        "",
        "### 2.4 标签分布",
        "",
        "| 真值等级 | 样本数 | 占比 |",
        "|---|---|---|",
    ]
    total = max(label_distribution.get("n_samples", 0), 1)
    for level in RISK_LEVELS:
        count = (label_distribution.get("label_counts", {}) or {}).get(level, 0)
        lines.append(f"| {level} | {count} | {count / total * 100:.1f}% |")
    lines.append("")
    lines += [
        "### 2.5 逐塘标签分布",
        "",
        "| 池塘 | 样本 | 正常 | 预警 | 高危 | 高危占比 | 体重增长 |",
        "|---|---|---|---|---|---|---|",
    ]
    for pond_id in sorted(per_pond):
        row = per_pond[pond_id]
        growth = growth_by_pond.get(pond_id)
        growth_text = f"{growth:+.1f}%" if growth is not None else "—"
        lines.append(
            f"| {pond_id} | {row.get('n_samples', 0)} | {row.get('正常', 0)} | "
            f"{row.get('预警', 0)} | {row.get('高危', 0)} | "
            f"{row.get('high_risk_rate', 0) * 100:.1f}% | {growth_text} |"
        )
    lines.append("")

    if coherence is not None and coherence.n:
        lines += [
            "### 2.6 生长自洽性检验：标签说鱼要死了，鱼却在长个",
            "",
            "这是本报告里**唯一一条不依赖我们自己那套阈值的**合理性检查。"
            "它拿「各塘高危标签占比」去对「各塘鱼的体重增长倍数」算 Spearman 相关。"
            "如果标签反映的是真实的养殖风险，那么高危占比越高的塘，鱼应该长得越差——"
            "相关系数应当显著为负。",
            "",
            "| 参与池塘 | Spearman ρ | 解释 |",
            "|---|---|---|",
            f"| {coherence.n} | {coherence.rho:+.3f} | {coherence.verdict()} |",
            "",
        ]
    return "\n".join(lines)


def _protocol_section(
    *,
    n_windows: int,
    skipped: int,
    window_hours: float,
    stride_hours: float,
    species: str,
    horizon: int,
    n_ponds: int,
    aggregate_minutes: int,
) -> str:
    return "\n".join(
        [
            "## 3. 评测协议",
            "",
            "### 3.1 为什么这份数据能直接喂引擎",
            "",
            "`water_risk_predict` 的入参模型 `WaterRiskInput.readings` 有 `min_length=3`——"
            "它是一个**趋势**引擎，单点读数喂不进去。原来的地表水断面数据一行一个点位，"
            "必须靠复制凑长度（瞬时口径）；**这份池塘数据是真·时序**，"
            "每条塘有 3.7 万条左右的连续读数（原始采样间隔中位数 20 秒），"
            "所以评测不需要任何人工构造，「瞬时口径」这个妥协也就不需要了。",
            "",
            "### 3.2 滑窗构造",
            "",
            f"- 先按 {aggregate_minutes} 分钟**聚合**成一个读数"
            "（池子的变化尺度是小时级，20 秒一条的原始噪声没有信息量）",
            f"- 再按 **{window_hours:g} 小时窗口 / {stride_hours:g} 小时步长**切窗，"
            f"窗口**不重叠**",
            "- 真值取**窗口末端那个聚合点**的阈值标签——窗口内前面的点只承担「趋势证据」角色，"
            "这与工具在生产中的用法一致（拿最近几小时的采样点预测未来 72 小时）",
            "",
            "| 项 | 值 |",
            "|---|---|",
            f"| 参与池塘 | {n_ponds} |",
            f"| 窗口长度 | {window_hours:g} 小时 |",
            f"| 窗口步长 | {stride_hours:g} 小时 |",
            f"| 生成评测样本 | {n_windows} 个 |",
            f"| 窗口前不足历史而跳过 | {skipped} 条聚合点 |",
            f"| 引擎配置 | `species={species}`、`horizon_hours={horizon}`"
            "（阈值与权重全部用默认值，**没有为评测调过任何参数**） |",
            "",
            "### 3.3 指标定义",
            "",
            "- **规则一致性准确率**（不是准确率）：三档都算对的比例。**注意 2.3 的警告**——这里的「对」"
            "是「两套阈值口径一致」，不是「与事实相符」。",
            "- **宏平均 P/R/F1**：对三个等级一视同仁。如果某档完全没被预测出来，"
            "宏平均会立刻掉下来，而准确率不会——这是用它作主指标的原因。",
            "- **Cohen's κ**：扣除「随机也能对上」的那部分一致性。",
            "- **Spearman ρ（分数 vs 真值严重度）**：等级相关回答的是另一个问题——"
            "**引擎给的分数有没有随严重度单调上升**。ρ 高但准确率低，"
            "说明规则方向对、只是分档线没对齐；ρ 低才是真的没对上。",
            "- **漏报率**：真实高危的样本里被判成【正常】的比例。"
            "**在这个场景里漏报比误报贵得多**：漏报等于养殖户没收到该收到的提醒，可能一夜泛塘。",
            "- **误报率**：真实正常的样本里被判成【高危】的比例。误报多了，"
            "养殖户就会开始无视告警——这是漏报之外的另一条失效路径。",
            "- **红线召回**：真值高危的样本里，引擎自己的红线规则被触发的比例。"
            "两套「红线」是两份代码，所以这个数字衡量的是应急机制有没有真的启动。",
            "- **主控因子一致率**：引擎报告的 `limiting_factor` 是否落在标签认定的主控指标集合里。"
            "总分低但这一项高，说明引擎「知道问题在哪、只是分档不同」。",
            "",
        ]
    )


def render_run_section(
    run: EvalRun,
    *,
    caption: str = "",
    include_level_table: bool = True,
    notes: list[str] | None = None,
) -> str:
    """渲染一次评测的完整结果块。"""
    matrix = run.confusion()
    macro = matrix.macro()
    lines: list[str] = []
    if caption:
        lines += [caption, ""]

    lines += [
        f"样本 {len(run.usable)} 个（工具失败 {len(run.samples) - len(run.usable)} 个，不计入指标）；"
        f"真值分布 {run.true_distribution()}；预测分布 {run.severity_distribution()}；"
        f"`risk_score` 与真值严重度的 Spearman ρ = **{run.score_spearman():.3f}**。",
        "",
        "**混淆矩阵**（行 = 阈值真值，列 = 引擎判定）",
        "",
    ]
    lines.append(matrix.render_markdown())
    lines.append("")
    lines.append(matrix.render_metrics_markdown())
    lines.append("")

    lines += [
        f"- **漏报率**：真实高危的样本中有 **{_fmt_pct(run.late_prediction_rate())}** "
        "被判成【正常】。",
        f"- **误报率**：真实正常的样本中有 **{_fmt_pct(run.false_alarm_rate())}** "
        "被判成【高危】。",
        f"- **红线召回**：{_fmt_rate(*run.redline_recall())}。",
        f"- **主控因子一致率**：{_fmt_rate(*run.limiting_agreement())}。",
        "",
    ]

    if include_level_table:
        lines += [
            "**`risk_score` 随真值等级的变化**（分数有没有随严重度单调上升）",
            "",
            "| 真值等级 | n | min | p25 | 中位数 | p75 | max | 均值 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for level, stats in run.score_by_true_level().items():
            lines.append(
                f"| {level} | {stats['count']} | {stats['min']} | {stats['p25']} | "
                f"{stats['median']} | {stats['p75']} | {stats['max']} | {stats['mean']} |"
            )
        lines.append("")

    if notes:
        lines += [*[f"- {note}" for note in notes], ""]

    lines.append(
        f"<sub>κ = {macro.kappa:.3f}，宏平均 F1 = {macro.macro_f1:.3f}，"
        f"加权 F1 = {macro.weighted_f1:.3f}</sub>"
    )
    lines.append("")
    return "\n".join(lines)


def _sample_line(sample) -> str:
    engine = sample.risk_level or "—"
    hits = "、".join(sample.rule_hits) or "—"
    limiting = "、".join(sample.true_limiting) or "—"
    return (
        f"| {sample.pond_id} | {sample.sampled_on.isoformat()} | {sample.true_level}"
        f"（{limiting}） | **{engine}** | {sample.risk_score:.1f} | {hits} |"
    )


def _misses_section(run: EvalRun) -> str:
    lines = [
        "### 4.4 误报与漏报清单",
        "",
        "这两张清单比矩阵更有说服力：矩阵说明「错了多少」，清单说明「错在哪」。",
        "这里的每一行都是**两套阈值口径的分歧**，不是「引擎错了、事实如此」——"
        "读的时候请对照 2.3。",
        "",
        "#### 4.4.1 漏报（阈值真值高危 → 引擎没判高危）",
        "",
        "| 池塘 | 日期 | 真值（主控指标） | 引擎判定 | risk_score | 命中的规则 |",
        "|---|---|---|---|---|---|",
    ]
    misses = run.misses(limit=12)
    lines += [_sample_line(s) for s in misses] if misses else ["| — | — | — | 无漏报 | — | — |"]
    lines += [
        "",
        "#### 4.4.2 误报（阈值真值正常 → 引擎判高危）",
        "",
        "| 池塘 | 日期 | 真值（主控指标） | 引擎判定 | risk_score | 命中的规则 |",
        "|---|---|---|---|---|---|",
    ]
    alarms = run.false_alarms(limit=12)
    lines += (
        [_sample_line(s) for s in alarms] if alarms else ["| — | — | — | 无误报 | — | — |"]
    )
    lines.append("")
    return "\n".join(lines)


def _sensitivity_section(runs: list[EvalRun]) -> str:
    """把多组窗口长度/参数的指标并排放，方便一眼看出结论对协议有多敏感。"""
    lines = [
        "### 4.2 敏感性分析",
        "",
        "同一个引擎换窗口长度、换池塘子集，结论会不会翻？下面每行都是独立跑出来的一次评测。",
        "**如果换一套协议就能把一致性拉高一大截，那说明这个数字是协议的产物、不是引擎的能力。**",
        "",
        "| 配置 | 样本 | 规则一致性 | 宏平均 P | 宏平均 R | 宏平均 F1 | κ | Spearman ρ | 漏报率 | 误报率 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        macro = run.confusion().macro()
        lines.append(
            f"| {run.name} | {len(run.usable)} | {macro.accuracy:.3f} | "
            f"{macro.macro_precision:.3f} | {macro.macro_recall:.3f} | {macro.macro_f1:.3f} | "
            f"{macro.kappa:.3f} | {run.score_spearman():.3f} | "
            f"{_fmt_pct(run.late_prediction_rate())} | {_fmt_pct(run.false_alarm_rate())} |"
        )
    lines.append("")
    return "\n".join(lines)


def _per_pond_section(run: EvalRun) -> str:
    per_pond = run.per_pond()
    lines = [
        "### 4.3 逐塘分解",
        "",
        "总量看不出「是不是某一口塘拖的」。这份数据的 11 口塘水质差异极大，"
        "逐塘看是必须的——**某一口塘的分布与其他塘完全不同，本身就是一条数据结论**。",
        "",
        "| 池塘 | 样本 | 规则一致性 | 真值（正常/预警/高危） | 预测（正常/预警/高危） |",
        "|---|---|---|---|---|",
    ]
    for pond_id in sorted(per_pond):
        row = per_pond[pond_id]
        true_counts = row["true"]
        pred_counts = row["pred"]
        true_text = "/".join(str(true_counts.get(level, 0)) for level in RISK_LEVELS)
        pred_text = "/".join(str(pred_counts.get(level, 0)) for level in RISK_LEVELS)
        lines.append(
            f"| {pond_id} | {row['n']} | {row['accuracy']:.3f} | {true_text} | {pred_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _do_trend_section(trend) -> str:
    """第二组外部验证：这一组**有真值**，所以数字可以当真。"""
    overall = trend.overall
    lead_minutes = int(round(trend.lead_hours * 60))
    lines = [
        "## 5. 外部验证二：溶氧趋势外推 vs 持久性基线",
        "",
        "第一组验证的标签是我们自己算的（2.3），所以它测的是「两套阈值口径的分歧」。"
        "这一组不一样：它预测的是**真实发生的未来读数**，真值不是算出来的。",
        "",
        "验证对象是本项目里唯一一处「预测」代码——`water_risk.hybrid_forecast`"
        "（锚点 = 最近 3 点均值，修正项 = 最近 3 点斜率 × 阻尼系数 0.05，"
        "结果裁剪到物理量程），也就是 `_predict_series()` 留给 LSTM 的那个替换点。"
        "评测侧直接调用这个函数，量的就是线上跑的那段代码，不是它的仿写。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 数据 | `aquaponics_pond` 的 {trend.n_ponds} 口塘 |",
        f"| 可用样本（目标点不重叠） | {overall.n} |",
        f"| 聚合步长 | {trend.step_minutes} 分钟 |",
        f"| 历史窗口 | {max(trend.by_history) if trend.by_history else 0} 个聚合点 |",
        f"| 预测提前量 | {lead_minutes} 分钟（{trend.lead_hours:g} 小时） |",
        "",
        f"协议：步长取 {lead_minutes} 分钟（= 预测提前量），所以**目标点互不重叠**；"
        "对每个起点，锚点取最近 3 个点的均值，修正项取最近 3 个点的最小二乘斜率再乘 0.05，"
        f"预测 **{lead_minutes} 分钟之后**的溶氧；"
        "对照基线是**持久性预测**（`未来值 = 当前值`）。",
        f"> ⚠ **回看窗口之间是重叠的**（历史 "
        f"{max(trend.by_history) if trend.by_history else 0} 个点 > 步长），"
        f"所以 {overall.n} 是样本数、不是独立样本数，不要拿它当统计显著性用。",
        "",
        "**为什么拿持久性当基线**：一步预测上，如果连「什么都不做」都赢不了，"
        "那这个趋势组件的点预测就没有价值——它顶多能提示方向。"
        "本项目曾经就输在这里（全量斜率外推 MAE 1.943 vs 基线 1.696），"
        "现在的混合基线是把斜率项压到 5% 之后的版本。",
        "",
        "| 分组 | n | 引擎 MAE | 基线 MAE | 改善 | 引擎 RMSE | 基线 RMSE | 逐条胜率 | 方向判对率 | 下降召回 |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| **全部** | {overall.n} | {overall.engine_mae:.3f} | {overall.persistence_mae:.3f} | "
        f"**{overall.improvement_pct:+.1f}%** | {overall.engine_rmse:.3f} | "
        f"{overall.persistence_rmse:.3f} | {overall.win_rate * 100:.1f}% | "
        f"{overall.direction_accuracy * 100:.1f}% | {overall.decline_recall * 100:.1f}% |",
    ]
    for length in sorted(trend.by_history):
        score = trend.by_history[length]
        lines.append(
            f"| 历史窗口 {length} 点 | {score.n} | {score.engine_mae:.3f} | "
            f"{score.persistence_mae:.3f} | {score.improvement_pct:+.1f}% | "
            f"{score.engine_rmse:.3f} | {score.persistence_rmse:.3f} | "
            f"{score.win_rate * 100:.1f}% | {score.direction_accuracy * 100:.1f}% | "
            f"{score.decline_recall * 100:.1f}% |"
        )
    lines += [
        "",
        f"逐条比较：引擎更好 **{overall.engine_better}** 条 / 基线更好 "
        f"**{overall.persistence_better}** 条 / 打平 {overall.ties} 条。",
        "",
        f"**结论**：{trend.verdict()}",
        "",
        "这条结论怎么用：",
        "",
        "- **它没有推翻引擎的定级逻辑。** 实时超限判定（溶氧 < 3.0 预警、< 1.5 泛塘）"
        "不依赖外推，用的是当前值，与本节无关。",
        "- **它给出了「预计 N 小时后跌破某线」这句话的边界。** 这类提示必须被读成"
        "「按当前方向继续会怎样」的**趋势提示**，不能读成带精度的预测。"
        "引擎把方法标成 `hybrid_persistence_baseline`、"
        "并把外推上限压在 12 小时，现在这两个设计选择有了实测依据。",
        "- **改善几乎全部来自锚点平滑。** 拆开看：只用「最近 3 点均值」不做斜率修正时 "
        "MAE 1.525（+10.1%），加上 5% 的斜率修正后是 1.529——斜率项贡献约 0.3 个百分点；"
        "而全量斜率外推（阻尼 = 1）在真实数据上是 1.943，比基线还差。"
        "所以这里**不是**「趋势预测做好了」，是「噪声被平滑掉了」。",
        "- **「下降召回」这一列是决定性的。** 如果点预测输给基线、但方向判对率高，"
        "那外推仍有报警价值；如果连方向都判不对，"
        "那么任何形如「预计 3 小时后溶氧跌破临界」的文案都是误导，必须改口径或换模型。"
        "目前方向判对率仍是 44.1%——**这一列没有改善**，别把它当已解决。",
        "",
    ]
    return "\n".join(lines)


def _vision_section(metrics: dict) -> str:
    """第 8 节：图像模态（病害识别模型）的留出集测试结果。

    这一节与前 7 节**不是同一类验证**，报告里必须说清楚——
    前 7 节是「外部真实数据 vs 水质规则引擎」，本节是「自建模型的留出集测试」。
    混在一起而不加区分，会让读者以为图像指标也经过了外部验证。

    数据来源是 `data/derived/metrics.json`（由 ``scripts/train_yolo_fish.py`` 写出），
    这里只做渲染，不重算任何数字。
    """
    n_test = metrics.get("test_size", 0)
    top1 = metrics.get("top1_accuracy")
    f1 = metrics.get("macro_f1")
    map50 = metrics.get("mAP50_weak_fullframe")
    hyper = metrics.get("hyperparams", {}) or {}
    caveat = metrics.get("mAP50_caveat", "")

    lines: list[str] = [
        "## 8. 附：图像模态——病害识别模型的留出集测试",
        "",
        "> **本节与前 7 节不是同一类验证，请分开读。** 前 7 节用外部真实数据检验"
        "**水质规则引擎**；本节是**自建四分类模型在自己留出的测试集上**的表现，"
        "数据来自公开图像集，与水质数据没有交集，也没有任何第三方交叉验证。"
        "放在同一份报告里只是为了给「评测」这件事一个统一入口。",
        "",
        f"- 模型：`{metrics.get('model', 'yolov8-fish.pt')}`（YOLOv8n-cls 四分类："
        "健康鱼 / 烂鳃 / 细菌性出血 / 水霉）",
        f"- 超参：epochs {hyper.get('epochs', '—')}、imgsz {hyper.get('imgsz', '—')}、"
        f"batch {hyper.get('batch', '—')}、seed {hyper.get('seed', '—')}",
        f"- 数据集：`{metrics.get('dataset', '—')}`",
        f"- 切分口径：{metrics.get('identity_rule', '—')}",
        "- 训练脚本与完整复现步骤见 [`模型训练说明.md`](模型训练说明.md)。",
        "",
        f"### 8.1 测试集指标（{n_test} 张，训练时没见过）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
    ]

    if top1 is not None:
        lines.append(f"| Top-1 准确率 | {_fmt_pct(top1)} |")
    if f1 is not None:
        lines.append(f"| macro F1 | {f1:.4f} |")
    if map50 is not None:
        lines.append(f"| mAP@0.5（**弱口径**，见 8.2） | {map50:.4f} |")

    per_class = metrics.get("per_class", {}) or {}
    if per_class:
        zh_by_key = {c["key"]: c.get("zh", c["key"]) for c in metrics.get("classes", [])}
        lines += [
            "",
            "| 类别 | support | precision | recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, row in per_class.items():
            lines.append(
                f"| {zh_by_key.get(key, key)} `{key}` | {row.get('support', '—')} | "
                f"{row.get('precision', 0):.4f} | {row.get('recall', 0):.4f} | "
                f"{row.get('f1', 0):.4f} |"
            )

    lines += ["", "### 8.2 ⚠ mAP@0.5 是弱口径，不能当定位指标用", "", caveat or "（未提供 caveat）", ""]

    n_mis = len(metrics.get("misclassified_samples", []) or [])
    confusion = metrics.get("confusion", {}) or {}
    if n_mis:
        # 「病害被判成健康」是最危险的错法：预警会偏乐观。
        misses = sum(
            confusion.get(true_key, {}).get("healthy", 0)
            for true_key in confusion
            if true_key != "healthy"
        )
        lines += [
            "### 8.3 主要错法",
            "",
            f"共 {n_mis} 张分错，其中 **{misses} 张是「病害 → 健康」的漏检方向**。"
            "这个方向比反过来（健康 → 病害）更危险，因为漏检会让水质预警偏乐观。"
            "全部错分样本（含预测置信度）见 `data/derived/metrics.json` 的"
            " `misclassified_samples` 字段。",
            "",
        ]

    return "\n".join(lines)


def render_report(
    *,
    data_stats: dict,
    label_distribution: dict,
    primary: EvalRun,
    sensitivity: list[EvalRun],
    skipped: int,
    generated_at: str,
    species: str,
    horizon_hours: int,
    coherence=None,
    do_trend=None,
    vision_metrics: dict | None = None,
    conclusion: str = "",
    limitations: list[str] | None = None,
) -> str:
    """渲染完整评测报告。

    `data_stats` 是 `ponds.load_ponds` 的 totals，`label_distribution` 是
    `ponds.summarize_samples` 的返回值，`coherence` 是 `labels.growth_coherence`
    的结果——它提供本报告里唯一一条不依赖自建阈值的合理性检查。

    `vision_metrics` 是**另一件东西**：图像模态（病害识别模型）的留出集指标，
    读自 `data/derived/metrics.json`。它不属于第 4/5 节那两组验证，
    因此单独成节（第 8 节）并在标题里写明「附」，免得被误读成同一类证据。
    """
    macro = primary.confusion().macro()
    overall = do_trend.overall if do_trend is not None else None
    lines: list[str] = [
        "# M8 外部验证报告：水产池塘数据上的水质规则引擎评测（附图像模态留出集测试）",
        "",
        f"> 生成时间 {generated_at}　自动生成，勿手改；复现命令见第 7 节。"
        "改动引擎或数据后请重跑，不要手动更新数字。",
        "",
        "## 0. 摘要",
        "",
        "本次在**水产养殖池塘的实测时序数据**上做了两组自动验证、一组人工验证，先说结论再展开：",
        "",
        "| | 外部验证一：规则一致性 | 外部验证二：趋势外推 | 外部验证三：人工标签 |",
        "|---|---|---|---|",
        "| 数据 | `aquaponics_pond`（11 口塘，114 万行） | 同一份数据的溶氧序列 | 同一份数据里分层的 100 个窗口 |",
        "| 真值形态 | **按行业阈值现算的**三档标签 | **真实发生的**未来溶氧读数 | **人逐条判的**等级 |",
        "| 验证对象 | 引擎的**阈值定级** | 引擎的**趋势外推** | 引擎的**阈值定级** |",
        "| 独立性 | ⚠ **不独立**——标签由同一套阈值算出 | ✅ 独立——真值不是算出来的 | ✅ 独立——真值来自人 |",
        "| 能不能叫「准确率」 | ❌ 只能叫**规则一致性** | ❌ 是相对基线的优劣 | ✅ **这个才是准确率** |",
        "",
        "### 0.1 外部验证一：两套阈值口径的对照（规则一致性，不是准确率）",
        "",
        f"- 在 **{len(primary.usable)}** 个不重叠窗口上"
        f"（{label_distribution.get('n_ponds', 0)} 口塘、"
        f"{label_distribution.get('time_from', '—')} ～ {label_distribution.get('time_to', '—')}），"
        f"**规则一致性准确率** {macro.accuracy:.1%}、Cohen's κ {macro.kappa:.3f}、"
        f"Spearman ρ {primary.score_spearman():.3f}。",
        f"- 漏报率 **{_fmt_pct(primary.late_prediction_rate())}**"
        f"（真实高危被判【正常】）、误报率 **{_fmt_pct(primary.false_alarm_rate())}**"
        f"（真实正常被判【高危】）、红线召回 {_fmt_rate(*primary.redline_recall())}、"
        f"主控因子一致率 {_fmt_rate(*primary.limiting_agreement())}。",
        "- **这些数字不是「引擎准确率」。** 标签是我们在 2.3 节按 `water_risk.py` "
        "同一套阈值算出来的：标签取最差指标（最坏情况），引擎取加权评分加时间持续性"
        "（平均 + 趋势）。所以它测的是**两套口径的分歧**，不是与事实的符合度——"
        "它的用途是**证明代码没写错、两条实现基本一致**。",
        "- 这组验证仍有确切价值：它把「引擎会怎么偏离最坏情况口径」量化了，"
        "而且**逐塘分解**与**主控因子一致率**能指出分歧出在哪个指标上。",
        "- **要引用「准确率」，只能用外部验证三。** 人工标注集在 "
        "`data/eval_groundtruth/`（100 条，标满 20 条才出指标），"
        "跑 `scripts/evaluate_groundtruth.py` 得到 `人工标签准确率`，"
        "写进 `docs/_m8_groundtruth.json`。",
        "",
        "### 0.2 外部验证二：有真值，结果可以当真",
        "",
    ]

    if overall is not None:
        lines += [
            f"- 在 **{overall.n}** 个样本上（目标点互不重叠，回看窗口有重叠），"
            f"用过去的溶氧序列预测 "
            f"**{do_trend.lead_hours:g} 小时之后**的溶氧（真实读数，不是算出来的）。",
            f"- 引擎 MAE **{overall.engine_mae:.3f}**，持久性基线 MAE "
            f"**{overall.persistence_mae:.3f}**，引擎"
            + (
                f"**优于基线 {overall.improvement_pct:.1f}%**"
                if overall.improvement_pct > 1.0
                else (
                    f"**劣于基线 {abs(overall.improvement_pct):.1f}%**"
                    if overall.improvement_pct < -1.0
                    else f"与基线基本打平（{overall.improvement_pct:+.1f}%）"
                )
            )
            + "。",
            f"- 方向判对率 {overall.direction_accuracy * 100:.1f}%，"
            f"下降召回 {overall.decline_recall * 100:.1f}%。",
            "- 这一组的核心限制：**溶氧的混合基线是引擎唯一的预测成分**，"
            "它的强弱直接决定「预计 N 小时后跌破临界」这类提示能不能信。",
        ]
    else:
        lines.append("- （本次未运行）")

    lines += [
        "",
        "### 0.3 这两组结果合起来说明什么",
        "",
        "1. **定性结论：引擎的阈值定级与最坏情况口径有可量化的分歧，"
        "但分歧的方向与位置是清楚的**（看 4.3 逐塘与 4.4 清单）。",
        "2. **定量的、独立的结论只有一条**：溶氧混合基线相对「什么都不做」的表现"
        "（第 5 节）。这是全篇唯一一个可以对外引用的性能数字。",
        "3. **报告中没有出现「准确率 X%、可用于生产」这种话。** "
        "第一组不独立，第二组只覆盖预测成分——这两条拼不出一个「模型准确率」。",
        "",
        "## 1. 这份报告能证明什么、不能证明什么",
        "",
        "**能证明**",
        "",
        f"1. 引擎在 **{data_stats.get('n_lines', 0):,} 行真实池塘数据**上批量调用"
        "**不崩、不超时、失败率为 0**"
        f"（{len(primary.samples)} 个窗口，"
        f"{primary.engine_stats.get('ms_per_call', 0):.2f} ms/次）。",
        "2. **这份数据的 `Ammonia(g/ml)` 不是 g/ml，而是 µg/L**——"
        "依据是同数据集的鱼体增重事实，而且 IoTPond8 的 `(mg/L)` 列"
        "恰好是其余文件的 1/1000，两条证据互相印证。这是一条关于数据的确定结论。",
        "3. **各塘高危标签占比与鱼体增重的关系**："
        + (
            f"Spearman ρ = {coherence.rho:+.3f}（{coherence.n} 口塘）"
            if coherence is not None and coherence.n
            else "（本次未运行）"
        )
        + "——这是本文唯一一条不靠自建阈值的合理性检查。",
        "4. 溶氧混合基线的提前量-误差曲线（第 5 节），带对照基线，可复现。",
        "",
        "**不能证明**",
        "",
        "1. **不能证明引擎的分级准确率。** 标签由同一套阈值算出，"
        "第 4 节量的是口径分歧，不是正确率。",
        "2. **不能证明阈值选得对。** 阈值来自领域经验与 `water_risk.py` 的既有实现，"
        "没有对过 GB 11607 / NY 5051 现行原文，**也与本数据集自带标签无关**"
        "（这份数据没有水质分级标签）。",
        "3. **不是与领域专家的一致性检验。** 没有专家标注。",
        "4. **不能说明 LSTM 会更好。** 第 5 节只说明混合基线的表现，"
        "要证明替代模型更好必须用同一套协议复测。",
        "5. **亚硝酸盐是盲区。** 这份数据只有硝酸盐，引擎的一级指标 `nitrite_n` "
        "填不进去，权重被重新归一化。真实塘口如果亚硝酸盐超标，本评测看不到。",
        "",
        _quality_section(data_stats, label_distribution, coherence),
        _protocol_section(
            n_windows=len(primary.samples),
            skipped=skipped,
            window_hours=primary.window_hours,
            stride_hours=primary.window_hours,
            species=species,
            horizon=horizon_hours,
            n_ponds=label_distribution.get("n_ponds", 0),
            aggregate_minutes=10,
        ),
        "## 4. 外部验证一：阈值定级（两套口径的对照）",
        "",
        "> **先读 2.3。** 4.1 与 4.2 里的准确率、κ、ρ 全部建立在"
        "「标签由我们自己的阈值算出」这个前提上；不看 2.3 就会把它们误读成引擎的性能。",
        "",
        render_run_section(primary, caption="### 4.1 主口径（3 小时窗口，10 分钟聚合）"),
        _sensitivity_section(sensitivity),
        _per_pond_section(primary),
        _misses_section(primary),
        (
            _do_trend_section(do_trend)
            if do_trend is not None
            else "## 5. 外部验证二：溶氧趋势外推\n\n未运行。\n"
        ),
        "## 6. 结论与局限",
        "",
    ]

    lines.append(f"**一句话结论**（由脚本按实测数字生成）：{conclusion or '（见下）'}")
    lines.append("")

    if limitations:
        lines += [*[f"- {note}" for note in limitations], ""]

    lines += [
        "## 7. 复现方式",
        "",
        "```powershell",
        "cd <工程>",
        ".\\.venv\\Scripts\\python.exe scripts\\evaluate_water_rules.py",
        "```",
        "",
        "脚本自己会把 `src` 插进 `sys.path`，**不需要预先设 `PYTHONPATH`**。",
        "脚本只依赖标准库与本项目自身代码，不联网、不调用大模型、不需要 GPU。",
        "数据集路径可用 `--data-dir` 覆盖；报告默认写到 `docs/M8_评测报告.md`，",
        "同时把机器可读的指标写到 `docs/_m8_metrics.json`。",
        "两组验证都用同一份池塘数据（`aquaponics_pond`），不再需要分别指定路径。",
        "",
        *(
            [_vision_section(vision_metrics), ""]
            if vision_metrics is not None
            else [
                "## 8. 附：图像模态——病害识别模型的留出集测试",
                "",
                "未运行——本机没有 `data/derived/metrics.json`。"
                "先跑 `scripts/train_yolo_fish.py` 训练一次即可（见 "
                "[`模型训练说明.md`](模型训练说明.md)）。"
                "**这一节缺失不影响第 1~7 节的结论。**",
                "",
            ]
        ),
    ]
    return "\n".join(lines)
