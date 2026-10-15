"""按水产养殖阈值把单条读数打成 正常 / 预警 / 高危，并检验标签与实际生长的自洽性。

## 为什么要自己造标签

原来的「第一组外部验证」用的是 `china_water_pollution_data.csv` 的
`Pollution_Level` 列。实测它与全部测量列统计独立（连数据集自带的
`Water_Quality_Index` 都只有 ρ +0.0x），那把尺子没有刻度——
真值不可用，意味着「分级准确率」这一项**根本没有测**。

换成 PondsData 之后，数据集本身**没有**人工标注，所以标签必须自己造。
本模块用的是池塘养殖通行阈值（与 `tools/water_risk.py` 的
`SPECIES_PROFILES["鲤科"]`、`ACTION_THRESHOLDS` 同源）：

| 指标 | 单位 | 适宜 | 预警线（超出即预警） | 红线（越过即高危） |
|---|---|---|---|---|
| 溶氧 | mg/L | ≥ 5.0 | < 3.0 | < 1.5（泛塘临界值） |
| 氨氮 | mg/L | ≤ 0.2 | > 0.5 | > 1.0 |
| 亚硝酸盐 | mg/L | ≤ 0.1 | > 0.3 | > 0.5 |
| pH | — | 6.5 – 8.5 | 6.0 – 9.0 之外 | < 5.5 或 > 9.5 |
| 水温 | ℃ | 20 – 30 | 15 – 33 之外 | — |

**判定方式刻意与引擎不同**：引擎是加权评分（`BASE_WEIGHTS` + 趋势外推 +
红线覆盖），本模块是**最差指标决定论**——任一指标越过红线就是高危。
两者因此可以真刀真枪地比，而不是拿引擎去比它自己的评分函数。

## 这份标签的局限（会写进报告）

标签与测量值同源：它不是独立真值，而是「按标准该怎么判」的形式化。
所以准确率衡量的是**引擎与标准的一致性**，不是引擎与现实的差距。
补上这最后一段的是同一份数据集里的鱼体长/体重，见 `growth_coherence()`：
标签若真的反映塘的健康状况，高危样本占比高的池塘，鱼应该长得更慢。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .metrics import spearman

__all__ = [
    "AQUA_BANDS",
    "RISK_LEVELS",
    "RISK_ORDINAL",
    "CoherenceReport",
    "IndicatorBand",
    "PondOutcome",
    "growth_coherence",
    "indicator_severities",
    "label_distribution",
    "label_indicators",
    "worst_indicators",
]

#: 三档风险等级，序号即严重度（用于等级单调性检验）。
RISK_LEVELS: tuple[str, ...] = ("正常", "预警", "高危")
RISK_ORDINAL: dict[str, int] = {level: i for i, level in enumerate(RISK_LEVELS)}


@dataclass(frozen=True)
class IndicatorBand:
    """一个养殖水质指标的分级线。

    区间用 ``None`` 表示该侧不限——溶氧只有下界（越低越坏），
    氨氮只有上界（越高越坏）。这样一个类就能同时表达两类指标。
    """

    field: str
    label: str
    unit: str
    #: 适宜区间（含端点）。
    optimum: tuple[float | None, float | None]
    #: 预警区间；落在适宜区间之外、这个区间之内 = 预警。
    watch: tuple[float | None, float | None]
    #: 红线；越过即高危。
    redline: tuple[float | None, float | None]

    @staticmethod
    def _outside(value: float, bounds: tuple[float | None, float | None]) -> bool:
        low, high = bounds
        if low is not None and value < low:
            return True
        if high is not None and value > high:
            return True
        return False

    def breaches_redline(self, value: float) -> bool:
        return self._outside(value, self.redline)

    def breaches_optimum(self, value: float) -> bool:
        return self._outside(value, self.optimum)

    def severity(self, value: float) -> int:
        """这条读数在这个指标上算第几档（0 正常 / 1 预警 / 2 高危）。"""
        if self.breaches_redline(value):
            return 2
        if self.breaches_optimum(value):
            return 1
        return 0


#: 分级表。字段名与引擎的 `WaterQualityPoint` 对齐，便于两边逐个指标对照。
AQUA_BANDS: tuple[IndicatorBand, ...] = (
    IndicatorBand(
        field="dissolved_oxygen",
        label="溶氧",
        unit="mg/L",
        optimum=(5.0, None),
        watch=(3.0, None),
        redline=(1.5, None),
    ),
    IndicatorBand(
        field="ammonia_n",
        label="氨氮",
        unit="mg/L",
        optimum=(None, 0.2),
        watch=(None, 0.5),
        redline=(None, 1.0),
    ),
    IndicatorBand(
        field="nitrite_n",
        label="亚硝酸盐",
        unit="mg/L",
        optimum=(None, 0.1),
        watch=(None, 0.3),
        redline=(None, 0.5),
    ),
    IndicatorBand(
        field="ph",
        label="pH",
        unit="",
        optimum=(6.5, 8.5),
        watch=(6.0, 9.0),
        redline=(5.5, 9.5),
    ),
    IndicatorBand(
        field="water_temp",
        label="水温",
        unit="℃",
        optimum=(20.0, 30.0),
        watch=(15.0, 33.0),
        redline=(None, None),
    ),
)

BANDS_BY_FIELD: dict[str, IndicatorBand] = {b.field: b for b in AQUA_BANDS}


def indicator_severities(values: Mapping[str, float | None]) -> dict[str, int]:
    """逐个指标给出严重度；缺测的指标不出现。"""
    out: dict[str, int] = {}
    for field, band in BANDS_BY_FIELD.items():
        value = values.get(field)
        if value is None:
            continue
        out[field] = band.severity(float(value))
    return out


def worst_indicators(values: Mapping[str, float | None]) -> tuple[str, ...]:
    """导致当前等级的那些指标（可能不止一个）。全部合格时返回空元组。"""
    severities = indicator_severities(values)
    if not severities:
        return ()
    top = max(severities.values())
    if top == 0:
        return ()
    return tuple(field for field, sev in severities.items() if sev == top)


def label_indicators(values: Mapping[str, float | None]) -> str:
    """按「最差指标决定论」给一条读数定级。

    全部指标都在适宜区间内 → 正常；有指标越红线 → 高危；其余越界 → 预警。

    一项都没测出来时返回【预警】而不是【正常】——**缺测不能当合格**。
    这份数据里温度/溶氧缺测是常态，静默判正常等于把「没看见」说成「没问题」。
    """
    severities = indicator_severities(values)
    if not severities:
        return "预警"
    return RISK_LEVELS[max(severities.values())]


def label_distribution(labels: Iterable[str]) -> dict[str, int]:
    """统计各等级条数。"""
    out = {level: 0 for level in RISK_LEVELS}
    for label in labels:
        if label in out:
            out[label] += 1
    return out


# ---------------------------------------------------------------------------
# 标签自洽性检验：标签 vs 鱼的实际生长
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PondOutcome:
    """一个池塘上「标签分布」与「实际产出」的对照。"""

    pond_id: str
    n_labeled: int
    high_risk_rate: float
    warning_rate: float
    do_median: float | None = None
    ammonia_median: float | None = None
    #: 体重从首段到末段的中位数增长（%）；没有体重列时为 ``None``。
    growth_pct: float | None = None


@dataclass(frozen=True)
class CoherenceReport:
    """高危占比与生长速度的关系。

    这是本次评测里**唯一一处真值与测量值不同源**的检验：
    标签由水质读数算出，长得「好不好」由鱼自己的体重体现。
    如果标签有效，高危占比高的池塘应该长得更慢，ρ 应为负。
    """

    ponds: tuple[PondOutcome, ...] = ()
    n_pairs: int = 0
    rho: float = 0.0

    @property
    def n(self) -> int:
        return self.n_pairs

    def verdict(self) -> str:
        if self.n_pairs < 4:
            return (
                f"只有 {self.n_pairs} 个池塘同时有标签与体重数据，"
                f"样本太少，这一项**不足以判定标签有效性**。"
            )
        level = "负" if self.rho < 0 else ("正" if self.rho > 0 else "零")
        if self.rho <= -0.5:
            strength = "标签与塘的实际产出方向一致"
        elif self.rho < 0:
            strength = "方向一致但很弱"
        else:
            strength = "**方向相反**——高危占比高的池塘反而长得更快"
        return (
            f"在 {self.n_pairs} 个池塘上，「高危样本占比」与「鱼体重增长」的 "
            f"Spearman ρ {self.rho:+.3f}（{level}相关）：{strength}。"
        )


def growth_coherence(outcomes: Sequence[PondOutcome]) -> CoherenceReport:
    """算「高危占比」与「体重增长」的 Spearman 相关。"""
    usable = [o for o in outcomes if o.growth_pct is not None and o.n_labeled > 0]
    if len(usable) < 2:
        return CoherenceReport(ponds=tuple(usable), n_pairs=len(usable), rho=0.0)
    rho = spearman(
        [o.high_risk_rate for o in usable],
        [float(o.growth_pct) for o in usable],
    )
    return CoherenceReport(ponds=tuple(usable), n_pairs=len(usable), rho=rho)
