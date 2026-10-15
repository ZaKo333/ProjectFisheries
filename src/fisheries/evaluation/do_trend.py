"""第二组外部验证：把引擎的溶氧趋势外推拿去和「持久性基线」比。

## 为什么这一组也换了数据集

原来这一组用的是 `Dissolved oxygen prediction in river water`
（147 + 63 行，`O2_1…O2_7` → `target`）。**换成河川数据是当初的妥协**：
当时手上唯一带回归真值的数据集只有它，而本项目的场景是**池塘养殖**——
河川的溶氧动力学（流动、复氧、点源污染）和静水塘不是一回事。

现在这份妥协不需要了。`aquaponics_pond` 是同一个项目要用的那份池塘数据，
它自带**真实时间戳**和连续溶氧序列，于是可以构造出比原协议更贴近生产的问题：

> 给定这口塘过去 `history_hours` 小时的溶氧，**1 小时后**的溶氧是多少？

原协议的「一个下标步」没有绝对时间单位，这里换成了真实的 10 分钟步长。

## 验证协议

对每个池塘，按 10 分钟聚合点滑动取样本：

- **历史**：过去 `history_points` 个点（默认 18 点 = 3 小时）的溶氧。
- **真值**：`lead_points` 个点之后（默认 6 点 = 1 小时）的溶氧。
- **引擎口径**：`末点 + slope_per_hour × lead_hours`，斜率由
  `water_risk._slope_per_hour` 最小二乘拟合——**直接复用引擎自己的函数**，
  保证量的是已交付的那段代码，不是复现品。这正是引擎 `_project` 在做的事。
- **基线口径**：持久性预测（persistence），即 `next = last`。
  这是时间序列预测的默认基线；一个趋势模型如果在一步预测上连「什么都不做」
  都赢不了，那它的价值就只在方向提示，不在点预测。

除了点误差，还单独统计**方向召回**：真值里溶氧真的往下走的那些样本，
引擎的斜率有没有给出「下降」的判断。点预测输了没关系，
但方向再错，这套外推就连提示价值都没有了。

**目标点**不重叠（步长 = `lead_points`），避免同一条读数被当成两个样本的真值；
但回看窗口之间有重叠，所以样本数不是独立样本数（`verdict()` 里会说明）。
`by_history` 按历史长度分组，用来看斜率对窗口长度有多敏感。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping, Sequence

from ..tools.water_risk import _slope_per_hour
from .ponds import PondSample

__all__ = [
    "DEFAULT_HISTORY_POINTS",
    "DEFAULT_LEAD_POINTS",
    "DEFAULT_STEP_MINUTES",
    "DoTrendReport",
    "PondDoSeries",
    "ProjectionScore",
    "engine_next",
    "evaluate_do_trend",
    "iter_do_series",
    "persistence_next",
    "score_series",
]

#: 聚合步长（分钟）。与 `ponds.AGGREGATE_MINUTES` 一致。
DEFAULT_STEP_MINUTES = 10
#: 历史窗口长度（聚合点数）。18 点 = 3 小时。
DEFAULT_HISTORY_POINTS = 18
#: 预测提前量（聚合点数）。6 点 = 1 小时。
DEFAULT_LEAD_POINTS = 6
#: 分组用的历史长度候选。
HISTORY_CHOICES: tuple[int, ...] = (6, 12, 18)


@dataclass(frozen=True)
class PondDoSeries:
    """一条待预测的池塘溶氧序列。"""

    series_id: str
    pond_id: str
    lags: tuple[float, ...]
    target: float | None = None
    origin: str = ""

    @property
    def n_lags(self) -> int:
        return len(self.lags)

    @property
    def last(self) -> float:
        return self.lags[-1]

    @property
    def usable(self) -> bool:
        return self.n_lags >= 2 and self.target is not None


def iter_do_series(
    by_pond: Mapping[str, Sequence[PondSample]],
    *,
    history_points: int = DEFAULT_HISTORY_POINTS,
    lead_points: int = DEFAULT_LEAD_POINTS,
    stride_points: int | None = None,
    limit_per_pond: int | None = None,
) -> Iterator[PondDoSeries]:
    """按池塘切出 `(历史溶氧, 1 小时后的溶氧)` 样本对。

    默认步长 = ``lead_points``：**目标点**互不重叠，同一条读数不会被当成两个样本的
    真值。注意回看窗口之间**是有重叠的**（``history_points`` 18 > 步长 6 时相邻样本
    共享 12 个历史点）——这不影响引擎与基线在同一批样本上的公平对比，
    但意味着有效独立样本数少于 ``oseries`` 的条数，不要拿它当统计显著性用。
    """
    if history_points < 2:
        raise ValueError("history_points 必须 >= 2（拟合斜率至少要两个点）")
    if lead_points < 1:
        raise ValueError("lead_points 必须 >= 1")
    stride = stride_points or lead_points

    for pond_id in sorted(by_pond):
        samples = list(by_pond[pond_id])
        emitted = 0
        for end in range(history_points - 1, len(samples) - lead_points, stride):
            chunk = samples[end - history_points + 1 : end + 1]
            lags = tuple(s.dissolved_oxygen for s in chunk)
            if len(lags) != history_points:
                continue
            target = samples[end + lead_points].dissolved_oxygen
            emitted += 1
            yield PondDoSeries(
                series_id=f"{pond_id}#{end}",
                pond_id=pond_id,
                lags=lags,
                target=target,
                origin=chunk[-1].sampled_at.isoformat(),
            )
            if limit_per_pond is not None and emitted >= limit_per_pond:
                break


def engine_next(
    item: PondDoSeries,
    *,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    lead_hours: float,
) -> float:
    """引擎的预测：`末点 + 最小二乘斜率 × 提前量`。

    刻意调用 `water_risk._slope_per_hour` 而不是在这里重写一遍最小二乘——
    否则量的是「我复现的算法」，不是「线上跑的那段代码」。
    """
    if item.n_lags < 2:
        raise ValueError("至少需要 2 个历史点才能拟合斜率")
    interval_hours = step_minutes / 60.0
    slope = _slope_per_hour(list(item.lags), interval_hours)
    return item.last + slope * lead_hours


def persistence_next(item: PondDoSeries) -> float:
    """持久性基线：下一步等于当前值。"""
    return item.last


@dataclass(frozen=True)
class ProjectionScore:
    """一组序列上的预测质量。"""

    n: int = 0
    engine_mae: float = 0.0
    persistence_mae: float = 0.0
    engine_rmse: float = 0.0
    persistence_rmse: float = 0.0
    engine_better: int = 0
    persistence_better: int = 0
    ties: int = 0
    direction_hits: int = 0
    direction_n: int = 0
    #: 真值里溶氧确实下降的样本数（用来看方向召回的分母）。
    decline_n: int = 0
    #: 其中引擎斜率也判断为下降的条数。
    decline_hits: int = 0

    @property
    def improvement_pct(self) -> float:
        """引擎相对持久性基线的 MAE 改善率（正数 = 更好）。"""
        if self.persistence_mae <= 0:
            return 0.0
        return (self.persistence_mae - self.engine_mae) / self.persistence_mae * 100.0

    @property
    def direction_accuracy(self) -> float:
        """方向（升 / 降 / 平）判对的比例。"""
        if self.direction_n <= 0:
            return 0.0
        return self.direction_hits / self.direction_n

    @property
    def decline_recall(self) -> float:
        """真值下降的样本里，引擎也判下降的比例。"""
        if self.decline_n <= 0:
            return 0.0
        return self.decline_hits / self.decline_n

    @property
    def win_rate(self) -> float:
        """逐条比较时引擎胜出的比例（不含平局）。"""
        decided = self.engine_better + self.persistence_better
        if decided <= 0:
            return 0.0
        return self.engine_better / decided


def _sign(value: float, *, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def score_series(
    series: Iterable[PondDoSeries],
    *,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    lead_hours: float,
) -> ProjectionScore:
    """在给定序列集合上同时算引擎与基线的误差。"""
    interval_hours = step_minutes / 60.0
    n = 0
    engine_abs = persistence_abs = 0.0
    engine_sq = persistence_sq = 0.0
    engine_better = persistence_better = ties = 0
    direction_hits = direction_n = 0
    decline_n = decline_hits = 0

    for item in series:
        if not item.usable:
            continue
        assert item.target is not None  # usable 已保证
        predicted = engine_next(item, step_minutes=step_minutes, lead_hours=lead_hours)
        baseline = persistence_next(item)
        actual = item.target

        e_err = abs(predicted - actual)
        p_err = abs(baseline - actual)
        n += 1
        engine_abs += e_err
        persistence_abs += p_err
        engine_sq += e_err**2
        persistence_sq += p_err**2

        if e_err < p_err - 1e-12:
            engine_better += 1
        elif p_err < e_err - 1e-12:
            persistence_better += 1
        else:
            ties += 1

        slope = _slope_per_hour(list(item.lags), interval_hours)
        truth_sign = _sign(actual - item.last)
        direction_n += 1
        if _sign(slope) == truth_sign:
            direction_hits += 1
        if truth_sign < 0:
            decline_n += 1
            if _sign(slope) < 0:
                decline_hits += 1

    if n == 0:
        return ProjectionScore()

    return ProjectionScore(
        n=n,
        engine_mae=engine_abs / n,
        persistence_mae=persistence_abs / n,
        engine_rmse=(engine_sq / n) ** 0.5,
        persistence_rmse=(persistence_sq / n) ** 0.5,
        engine_better=engine_better,
        persistence_better=persistence_better,
        ties=ties,
        direction_hits=direction_hits,
        direction_n=direction_n,
        decline_n=decline_n,
        decline_hits=decline_hits,
    )


@dataclass(frozen=True)
class DoTrendReport:
    """两组外部验证里「有回归真值」的那一组。"""

    overall: ProjectionScore = field(default_factory=ProjectionScore)
    by_history: dict[int, ProjectionScore] = field(default_factory=dict)
    step_minutes: int = DEFAULT_STEP_MINUTES
    lead_hours: float = 1.0
    n_series: int = 0
    n_ponds: int = 0
    sources: tuple[str, ...] = ()

    def verdict(self) -> str:
        """一句话结论，措辞跟着数字走。"""
        if self.overall.n == 0:
            return "没有可用的溶氧序列，无法比较。"
        gap = self.overall.improvement_pct
        head = (
            f"在 {self.n_ponds} 口塘、{self.overall.n} 个样本上（目标点互不重叠，"
            f"回看窗口之间有重叠，故有效独立样本数更少），"
            f"预测 **{self.lead_hours:g} 小时后的溶氧**"
        )
        numbers = (
            f"线性外推 MAE {self.overall.engine_mae:.3f} mg/L，"
            f"持久性基线 {self.overall.persistence_mae:.3f} mg/L"
        )
        if gap > 1.0:
            return f"{head}，{numbers}——外推优于基线 {gap:.1f}%。"
        if gap < -1.0:
            return (
                f"{head}，{numbers}——外推**差于**基线 {abs(gap):.1f}%，"
                f"点预测不该采信，只能当趋势提示用。"
            )
        return f"{head}，{numbers}——两者基本打平（相差 {abs(gap):.1f}%）。"


def evaluate_do_trend(
    by_pond: Mapping[str, Sequence[PondSample]],
    *,
    step_minutes: int = DEFAULT_STEP_MINUTES,
    lead_points: int = DEFAULT_LEAD_POINTS,
    history_points: int = DEFAULT_HISTORY_POINTS,
    stride_points: int | None = None,
    limit_per_pond: int | None = None,
) -> DoTrendReport:
    """在池塘溶氧序列上跑「引擎外推 vs 持久性基线」。

    `by_history` 用 `HISTORY_CHOICES` 里的长度各跑一遍，
    但只在不超过 `history_points` 的候选上——历史不够长的分组没法切样本。
    """
    lead_hours = lead_points * step_minutes / 60.0

    overall_series = list(
        iter_do_series(
            by_pond,
            history_points=history_points,
            lead_points=lead_points,
            stride_points=stride_points,
            limit_per_pond=limit_per_pond,
        )
    )
    by_history: dict[int, ProjectionScore] = {}
    for length in HISTORY_CHOICES:
        if length > history_points:
            continue
        subset = list(
            iter_do_series(
                by_pond,
                history_points=length,
                lead_points=lead_points,
                stride_points=stride_points,
                limit_per_pond=limit_per_pond,
            )
        )
        by_history[length] = score_series(
            subset, step_minutes=step_minutes, lead_hours=lead_hours
        )

    return DoTrendReport(
        overall=score_series(overall_series, step_minutes=step_minutes, lead_hours=lead_hours),
        by_history=by_history,
        step_minutes=step_minutes,
        lead_hours=lead_hours,
        n_series=len(overall_series),
        n_ponds=len(by_pond),
        sources=tuple(sorted(by_pond)),
    )
