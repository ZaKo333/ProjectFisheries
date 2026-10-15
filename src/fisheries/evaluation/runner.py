"""评测执行：把池塘滑窗送进 `water_risk_predict`，收集预测并与阈值标签比对。

设计要点：

- **只调用公开的 `water_risk_predict`**，不 import 它的内部评分函数。
  外部验证的意义就在于「黑盒调用生产入口」——绕过入口直接调内部函数，
  测到的就不是用户实际会拿到的结果了。
- 真值一类的东西**不在这里重算**：`PondWindow.true_level` 是数据层按行业阈值
  打好的标签（最差指标决定论），引擎是加权评分（BASE_WEIGHTS + 趋势外推）。
  两套判定方式不同，比出来的数才有意义。
- 工具返回 `status != "ok"` 的样本单独计数、不计入混淆矩阵。不能把「失败」
  悄悄算成「预测为正常」——那会把工具的健壮性问题伪装成准确率。
- 除了三档混淆矩阵，还单独统计两件事：**红线召回**（真值高危的样本里，
  引擎有没有报出红线命中）和**主控因子一致率**（引擎说的主控因子，
  是不是标签认定的那个指标）。这两个比总分更能说明引擎「哪里对、哪里错」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Sequence

from .labels import RISK_LEVELS, RISK_ORDINAL
from .metrics import ConfusionMatrix, describe, spearman
from .ponds import WINDOW_HOURS, PondWindow

__all__ = ["EvalRun", "EvalSample", "evaluate", "run_engine"]


@dataclass
class EvalSample:
    """一个样本的预测结果。"""

    pond_id: str
    sampled_on: date
    #: 真值：数据层按行业阈值打的标签（正常 / 预警 / 高危）。
    true_level: str
    #: 标签认定的主控指标（英文键，可能不止一个）。
    true_limiting: tuple[str, ...]
    status: str  # 工具 status："ok" / "error"
    risk_level: str  # 工具判定的风险等级（失败时为空）
    risk_score: float  # 工具 risk_score（0-100）
    n_points: int
    limiting_factor: str = ""
    rule_hits: tuple[str, ...] = ()
    redline_hits: tuple[str, ...] = ()
    error: str = ""
    readings: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run_engine(
    windows: Iterable[PondWindow],
    *,
    species: str = "鲤科",
    horizon_hours: int = 72,
    limit: int | None = None,
) -> tuple[list[EvalSample], dict]:
    """逐个窗口调用规则引擎。返回 `(samples, stats)`。"""
    from ..tools.water_risk import water_risk_predict

    samples: list[EvalSample] = []
    n_failed = 0
    errors: list[str] = []
    started = time.perf_counter()

    for index, window in enumerate(windows):
        if limit is not None and index >= limit:
            break
        try:
            result = water_risk_predict(
                list(window.readings),
                pond_id=window.pond_id,
                species=species,
                horizon_hours=horizon_hours,
            )
        except Exception as exc:  # 工具约定不抛异常，但外部验证不该被单个样本打断
            # 只在这里造一个 error 结果，**不要**顺手累加 n_failed / 记 errors——
            # 下面 `status != "ok"` 那段是唯一的记账点，否则一次异常会被算成两次失败
            # （n_ok 还会算成 -1）。
            result = {"status": "error", "error_message": f"{type(exc).__name__}: {exc}"}

        status = str(result.get("status") or "error")
        message = ""
        if status != "ok":
            n_failed += 1
            message = str(result.get("error_message") or result.get("error_code") or "")
            if message:
                errors.append(f"{window.pond_id} {window.sampled_on}: {message}")

        samples.append(
            EvalSample(
                pond_id=window.pond_id,
                sampled_on=window.sampled_on,
                true_level=window.true_level,
                true_limiting=tuple(window.limiting),
                status=status,
                risk_level=str(result.get("risk_level") or ""),
                risk_score=float(result.get("risk_score") or 0.0),
                n_points=len(window.readings),
                limiting_factor=str(result.get("limiting_factor") or ""),
                rule_hits=tuple(result.get("rule_hits") or ()),
                redline_hits=tuple(result.get("redline_hits") or ()),
                error=message if status != "ok" else "",
                readings=list(window.readings),
            )
        )

    elapsed = time.perf_counter() - started
    stats = {
        "n_calls": len(samples),
        "n_failed": n_failed,
        "n_ok": len(samples) - n_failed,
        "elapsed_seconds": round(elapsed, 3),
        "ms_per_call": round(1000 * elapsed / len(samples), 2) if samples else 0.0,
        "errors": errors[:20],
    }
    return samples, stats


@dataclass
class EvalRun:
    """一次「预测 × 阈值真值」的评测结果。"""

    name: str
    mode: str  # 目前只有 "trend"：连续时序窗口
    window_hours: float
    samples: list[EvalSample]
    engine_stats: dict = field(default_factory=dict)

    @property
    def usable(self) -> list[EvalSample]:
        """工具成功返回的样本。失败样本不参与指标。"""
        return [s for s in self.samples if s.ok]

    # -- 指标 --------------------------------------------------------------

    def confusion(self) -> ConfusionMatrix:
        return ConfusionMatrix.from_pairs(
            ((s.true_level, s.risk_level) for s in self.usable),
            RISK_LEVELS,
        )

    def score_by_true_level(self) -> dict[str, dict]:
        """各真值等级下 `risk_score` 的分布——回答「分数随严重度单调吗」。"""
        return {
            level: describe([s.risk_score for s in self.usable if s.true_level == level])
            for level in RISK_LEVELS
        }

    def score_spearman(self) -> float:
        """`risk_score` 与真值严重度序号的 Spearman 相关。"""
        pairs = [
            (float(RISK_ORDINAL[s.true_level]), s.risk_score)
            for s in self.usable
            if s.true_level in RISK_ORDINAL
        ]
        if len(pairs) < 3:
            return 0.0
        return spearman([a for a, _ in pairs], [b for _, b in pairs])

    def late_prediction_rate(self) -> float:
        """门槛分析：真实高危的样本里，被判成【正常】的比例。

        这个数字单独拎出来，是因为在这个场景里**漏报比误报贵得多**：
        漏报意味着养殖户没收到该收到的提醒，可能一夜泛塘。
        """
        risky = [s for s in self.usable if s.true_level == "高危"]
        if not risky:
            return 0.0
        return sum(1 for s in risky if s.risk_level == "正常") / len(risky)

    def false_alarm_rate(self) -> float:
        """真实【正常】的样本里，被判成【高危】的比例。"""
        normal = [s for s in self.usable if s.true_level == "正常"]
        if not normal:
            return 0.0
        return sum(1 for s in normal if s.risk_level == "高危") / len(normal)

    def redline_recall(self) -> tuple[int, int]:
        """真值高危的样本里，引擎报出红线命中的条数 / 总条数。

        高危标签按定义就是「至少一个指标越过红线」。引擎自己的 `redline_hits`
        是它内部的红线规则——两个「红线」不是同一套代码，所以这个数字
        衡量的是「引擎的应急机制有没有被真正触发」。
        """
        risky = [s for s in self.usable if s.true_level == "高危"]
        return sum(1 for s in risky if s.redline_hits), len(risky)

    def limiting_agreement(self) -> tuple[int, int]:
        """引擎的主控因子与标签认定的主控指标一致的条数 / 可比条数。"""
        hit = total = 0
        for sample in self.usable:
            if not sample.true_limiting or not sample.limiting_factor:
                continue
            total += 1
            if sample.limiting_factor in sample.true_limiting:
                hit += 1
        return hit, total

    def severity_distribution(self) -> dict[str, int]:
        """预测等级的分布，用来看引擎是不是「只会说没事」。"""
        out = {level: 0 for level in RISK_LEVELS}
        for sample in self.usable:
            if sample.risk_level in out:
                out[sample.risk_level] += 1
        return out

    def true_distribution(self) -> dict[str, int]:
        out = {level: 0 for level in RISK_LEVELS}
        for sample in self.usable:
            if sample.true_level in out:
                out[sample.true_level] += 1
        return out

    def per_pond(self) -> dict[str, dict]:
        """逐塘的样本量、真值分布与准确率——总量看不出「是不是某一口塘拖的」。"""
        out: dict[str, dict] = {}
        for pond_id in sorted({s.pond_id for s in self.usable}):
            pond_samples = [s for s in self.usable if s.pond_id == pond_id]
            hit = sum(1 for s in pond_samples if s.true_level == s.risk_level)
            true_counts = {level: 0 for level in RISK_LEVELS}
            pred_counts = {level: 0 for level in RISK_LEVELS}
            for sample in pond_samples:
                if sample.true_level in true_counts:
                    true_counts[sample.true_level] += 1
                if sample.risk_level in pred_counts:
                    pred_counts[sample.risk_level] += 1
            out[pond_id] = {
                "n": len(pond_samples),
                "accuracy": round(hit / len(pond_samples), 4) if pond_samples else 0.0,
                "true": true_counts,
                "pred": pred_counts,
            }
        return out

    def misses(self, *, limit: int = 12) -> list[EvalSample]:
        """漏报：真实高危、预测不是高危。按 risk_score 从高到低（最冤的在前）。"""
        rows = [
            s for s in self.usable if s.true_level == "高危" and s.risk_level != "高危"
        ]
        return sorted(rows, key=lambda s: -s.risk_score)[:limit]

    def false_alarms(self, *, limit: int = 12) -> list[EvalSample]:
        """误报：真实正常、预测高危。按 risk_score 从低到高（最没道理的在前面）。"""
        rows = [
            s for s in self.usable if s.true_level == "正常" and s.risk_level == "高危"
        ]
        return sorted(rows, key=lambda s: s.risk_score)[:limit]

    def to_dict(self) -> dict:
        redline_hit, redline_total = self.redline_recall()
        agree_hit, agree_total = self.limiting_agreement()
        return {
            "name": self.name,
            "mode": self.mode,
            "window_hours": self.window_hours,
            "engine": self.engine_stats,
            "n_samples": len(self.samples),
            "n_usable": len(self.usable),
            "confusion": self.confusion().to_dict(),
            "spearman_score_vs_level": round(self.score_spearman(), 4),
            "late_prediction_rate": round(self.late_prediction_rate(), 4),
            "false_alarm_rate": round(self.false_alarm_rate(), 4),
            "redline_recall": {
                "hit": redline_hit,
                "total": redline_total,
                "rate": round(redline_hit / redline_total, 4) if redline_total else 0.0,
            },
            "limiting_agreement": {
                "hit": agree_hit,
                "total": agree_total,
                "rate": round(agree_hit / agree_total, 4) if agree_total else 0.0,
            },
            "severity_distribution": self.severity_distribution(),
            "true_distribution": self.true_distribution(),
            "score_by_true_level": self.score_by_true_level(),
            "per_pond": self.per_pond(),
        }


def evaluate(
    samples: Sequence[EvalSample],
    *,
    name: str = "",
    mode: str = "trend",
    window_hours: float = WINDOW_HOURS,
    engine_stats: dict | None = None,
) -> EvalRun:
    return EvalRun(
        name=name or f"{mode}/window={window_hours}h",
        mode=mode,
        window_hours=window_hours,
        samples=list(samples),
        engine_stats=dict(engine_stats or {}),
    )
