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
- `do_trend`：**有真值的那一组**——用池塘溶氧序列比较引擎的线性外推
  与持久性基线（真值是真实发生的未来读数，不是算出来的）
- `report`：渲染成 Markdown 评测报告

## 这两组验证的独立性不一样，别混着引用

| | 第一组（`runner`） | 第二组（`do_trend`） |
|---|---|---|
| 真值 | **按同一套阈值现算的** | **真实发生的未来读数** |
| 独立性 | ⚠ 不独立 | ✅ 独立 |
| 量的是什么 | 两套阈值口径（最坏情况 vs 加权评分）的分歧 | 线性外推相对持久性基线的优劣 |

第一组的准确率**不是**「引擎准确率」——标签和引擎用的是同一套阈值依据。
这条警告在 `report.py` 与 `docs/M8_评测报告.md` 里都逐字写着。
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
    "ClassReport",
    "CoherenceReport",
    "ConfusionMatrix",
    "DATASET_NAME",
    "DEFAULT_DIR",
    "DEFAULT_HISTORY_POINTS",
    "DEFAULT_LEAD_POINTS",
    "DEFAULT_STEP_MINUTES",
    "DoTrendReport",
    "EvalRun",
    "EvalSample",
    "HISTORY_CHOICES",
    "IndicatorBand",
    "MacroReport",
    "PondDoSeries",
    "PondOutcome",
    "PondSample",
    "PondWindow",
    "ProjectionScore",
    "RANGES",
    "REQUIRED_FIELDS",
    "RISK_LEVELS",
    "RISK_ORDINAL",
    "STRIDE_HOURS",
    "UNIT_DIVISORS",
    "WINDOW_HOURS",
    "describe",
    "evaluate",
    "evaluate_do_trend",
    "growth_coherence",
    "indicator_severities",
    "iter_do_series",
    "iter_windows",
    "label_distribution",
    "label_indicators",
    "load_pond",
    "load_ponds",
    "parse_timestamp",
    "pond_outcomes",
    "render_report",
    "render_run_section",
    "resolve_columns",
    "run_engine",
    "score_series",
    "spearman",
    "summarize_samples",
    "worst_indicators",
]
