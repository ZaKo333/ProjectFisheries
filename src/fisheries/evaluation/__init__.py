"""M8 外部验证：用真实数据集检验水质规则引擎。

这个子包只做**评测**，不参与线上决策链路——`fisheries.tools.water_risk`
不会被它反向依赖（只在 `runner.run_engine` 里黑盒调用公开入口）。

模块划分：

- `labels`：**阈值标签器**——按行业阈值把一行读数打成 正常 / 预警 / 高危
  （最差指标决定论），外加一条不依赖自建阈值的合理性检查
  `growth_coherence`（高危占比 vs 鱼体增重）
- `metrics`：混淆矩阵、准确率、宏平均 P/R/F1、Cohen's κ、Spearman、分位数
- `ponds`：读取 `aquaponics_pond`（11 个 IoTpond*.csv，114 万行），
  处理四种列名方言、单位换算与越界值，聚合成 10 分钟读数后切滑窗
- `runner`：调用规则引擎、把预测与阈值标签配成对、切片误报漏报
- `do_trend`：**有真值的那一组**——用池塘溶氧序列比较引擎的混合基线外推
  与持久性基线（真值是真实发生的未来读数，不是算出来的）
- `groundtruth`：**唯一有人工真值的一组**——从同一份数据分层抽样
  （`data/eval_groundtruth/`），由人逐条判等级，算「人工标签准确率」；
  不足 `MIN_LABELS` 条不出指标（宁可显示待标注）
- `report`：渲染成 Markdown 评测报告

## 这三组验证的独立性不一样，别混着引用

| | 第一组（`runner`） | 第二组（`do_trend`） | 第三组（`groundtruth`） |
|---|---|---|---|
| 真值 | **按同一套阈值现算的** | **真实发生的未来读数** | **人逐条判的等级** |
| 独立性 | ⚠ 不独立 | ✅ 独立 | ✅ 独立 |
| 量的是什么 | 两套阈值口径（最坏情况 vs 加权评分）的分歧 | 混合基线外推相对持久性基线的优劣 | 引擎判得**对不对** |
| 能叫「准确率」吗 | ❌ 只能叫**规则一致性** | ❌ 是相对基线的优劣 | ✅ **只有这个能** |

第一组的 71.9%**不是**「引擎准确率」——标签和引擎用的是同一套阈值依据，
它证明的是「代码没写错、两条实现基本一致」。这条警告在 `report.py`
与 `docs/M8_评测报告.md` 里都逐字写着；真正的准确率看第三组。
"""

from __future__ import annotations

from .do_trend import (
    DEFAULT_HISTORY_POINTS,
    DEFAULT_LEAD_POINTS,
    DEFAULT_STEP_MINUTES,
    HISTORY_CHOICES,
    DoTrendReport,
    PondDoSeries,
    ProjectionScore,
    evaluate_do_trend,
    iter_do_series,
    score_series,
)
from .groundtruth import (
    DEFAULT_DIR as GT_DEFAULT_DIR,
)
from .groundtruth import (
    DEFAULT_N as GT_DEFAULT_N,
)
from .groundtruth import (
    DEFAULT_SEED as GT_DEFAULT_SEED,
)
from .groundtruth import (
    MIN_LABELS as GT_MIN_LABELS,
)
from .groundtruth import (
    LabeledWindow,
    WindowSample,
    check_isolation,
    labeled_rows,
    read_sample,
    sample_windows,
    write_sample,
)
from .groundtruth import (
    evaluate as evaluate_groundtruth,
)
from .labels import (
    AQUA_BANDS,
    RISK_LEVELS,
    RISK_ORDINAL,
    CoherenceReport,
    IndicatorBand,
    PondOutcome,
    growth_coherence,
    indicator_severities,
    label_distribution,
    label_indicators,
    worst_indicators,
)
from .metrics import ClassReport, ConfusionMatrix, MacroReport, describe, spearman
from .ponds import (
    AGGREGATE_MINUTES,
    COLUMN_ALIASES,
    DATASET_NAME,
    DEFAULT_DIR,
    RANGES,
    REQUIRED_FIELDS,
    STRIDE_HOURS,
    UNIT_DIVISORS,
    WINDOW_HOURS,
    PondSample,
    PondWindow,
    iter_windows,
    load_pond,
    load_ponds,
    parse_timestamp,
    pond_outcomes,
    resolve_columns,
    summarize_samples,
)
from .report import render_report, render_run_section
from .runner import EvalRun, EvalSample, evaluate, run_engine

__all__ = [
    "AGGREGATE_MINUTES",
    "AQUA_BANDS",
    "COLUMN_ALIASES",
    "DATASET_NAME",
    "DEFAULT_DIR",
    "DEFAULT_HISTORY_POINTS",
    "DEFAULT_LEAD_POINTS",
    "DEFAULT_STEP_MINUTES",
    "GT_DEFAULT_DIR",
    "GT_DEFAULT_N",
    "GT_DEFAULT_SEED",
    "GT_MIN_LABELS",
    "HISTORY_CHOICES",
    "RANGES",
    "REQUIRED_FIELDS",
    "RISK_LEVELS",
    "RISK_ORDINAL",
    "STRIDE_HOURS",
    "UNIT_DIVISORS",
    "WINDOW_HOURS",
    "ClassReport",
    "CoherenceReport",
    "ConfusionMatrix",
    "DoTrendReport",
    "EvalRun",
    "EvalSample",
    "IndicatorBand",
    "LabeledWindow",
    "MacroReport",
    "PondDoSeries",
    "PondOutcome",
    "PondSample",
    "PondWindow",
    "ProjectionScore",
    "WindowSample",
    "check_isolation",
    "describe",
    "evaluate",
    "evaluate_do_trend",
    "evaluate_groundtruth",
    "growth_coherence",
    "indicator_severities",
    "iter_do_series",
    "iter_windows",
    "label_distribution",
    "label_indicators",
    "labeled_rows",
    "load_pond",
    "load_ponds",
    "parse_timestamp",
    "pond_outcomes",
    "read_sample",
    "render_report",
    "render_run_section",
    "resolve_columns",
    "run_engine",
    "sample_windows",
    "score_series",
    "spearman",
    "summarize_samples",
    "worst_indicators",
    "write_sample",
]
