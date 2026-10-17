"""评测指标：混淆矩阵、准确率、宏平均 P/R/F1、Cohen's kappa、Spearman 等级相关。

纯标准库实现。理由：这个项目要求"零依赖离线可跑"，而这里要用到的统计量
（混淆矩阵、宏平均、kappa、秩相关）加起来不到两百行，引入 scikit-learn
只为这几个函数不划算，也会让"断网也能复现评测"这句话多一个前提。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "ClassReport",
    "ConfusionMatrix",
    "MacroReport",
    "describe",
    "pearson",
    "ranks",
    "spearman",
]


# --- 混淆矩阵 -----------------------------------------------------------------


@dataclass
class ClassReport:
    """单个类别（这里是风险等级）的指标。"""

    label: str
    support: int  # 真实属于本类的样本数
    predicted: int  # 被预测为本类的样本数
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "support": self.support,
            "predicted": self.predicted,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class MacroReport:
    """整体指标。宏平均对每个类别一视同仁，加权平均按样本数加权。"""

    n: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    kappa: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "kappa": round(self.kappa, 4),
        }


@dataclass
class ConfusionMatrix:
    """通用有序标签的混淆矩阵。

    行 = 真实标签，列 = 预测标签。标签顺序由构造时的 `labels` 决定，
    渲染与遍历都按这个顺序，保证同一份数据每次跑出来的表是一样的。
    """

    labels: Sequence[str]
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = tuple(self.labels)
        allowed = set(self.labels)
        for (true, pred) in self.counts:
            if true not in allowed or pred not in allowed:
                raise ValueError(f"标签 {true!r}/{pred!r} 不在 {self.labels} 内")

    # -- 构造 --------------------------------------------------------------

    def add(self, true: str, pred: str, n: int = 1) -> None:
        if true not in self.labels or pred not in self.labels:
            raise ValueError(f"标签 {true!r}/{pred!r} 不在 {self.labels} 内")
        self.counts[(true, pred)] = self.counts.get((true, pred), 0) + n

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> ConfusionMatrix:
        matrix = cls(labels)
        for true, pred in pairs:
            matrix.add(true, pred)
        return matrix

    # -- 基本读数 ----------------------------------------------------------

    def count(self, true: str, pred: str) -> int:
        return self.counts.get((true, pred), 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def row_total(self, label: str) -> int:
        """真实为 label 的样本数（该类的支持度）。"""
        return sum(self.count(label, pred) for pred in self.labels)

    def col_total(self, label: str) -> int:
        """被预测为 label 的样本数。"""
        return sum(self.count(true, label) for true in self.labels)

    def row(self, label: str) -> list[int]:
        return [self.count(label, pred) for pred in self.labels]

    # -- 指标 --------------------------------------------------------------

    @property
    def accuracy(self) -> float:
        if not self.total:
            return 0.0
        return sum(self.count(x, x) for x in self.labels) / self.total

    def per_class(self) -> list[ClassReport]:
        reports: list[ClassReport] = []
        for label in self.labels:
            tp = self.count(label, label)
            support = self.row_total(label)
            predicted = self.col_total(label)
            reports.append(
                ClassReport(
                    label=label,
                    support=support,
                    predicted=predicted,
                    tp=tp,
                    fp=predicted - tp,
                    fn=support - tp,
                )
            )
        return reports

    def macro(self) -> MacroReport:
        reports = self.per_class()
        k = len(reports) or 1
        macro_p = sum(r.precision for r in reports) / k
        macro_r = sum(r.recall for r in reports) / k
        macro_f1 = sum(r.f1 for r in reports) / k
        total = self.total
        weighted_f1 = (
            sum(r.f1 * r.support for r in reports) / total if total else 0.0
        )

        # Cohen's kappa：扣除"随机也能对上"的那部分一致性
        po = self.accuracy
        if total:
            pe = sum(
                (self.row_total(x) / total) * (self.col_total(x) / total)
                for x in self.labels
            )
        else:
            pe = 0.0
        kappa = (po - pe) / (1 - pe) if not math.isclose(pe, 1.0) else 0.0

        return MacroReport(
            n=total,
            accuracy=po,
            macro_precision=macro_p,
            macro_recall=macro_r,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            kappa=kappa,
        )

    # -- 输出 --------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "labels": list(self.labels),
            "matrix": [self.row(x) for x in self.labels],
            "counts": {f"{t}->{p}": n for (t, p), n in sorted(self.counts.items())},
            "per_class": [r.to_dict() for r in self.per_class()],
            "macro": self.macro().to_dict(),
        }

    def render_markdown(self, caption: str = "") -> str:
        """渲染成"行=真实、列=预测"的 Markdown 表。对角线加粗。"""
        head = "| 真实 \\ 预测 | " + " | ".join(self.labels) + " | 合计 |"
        sep = "|---|" + "---|" * (len(self.labels) + 1)
        lines = [head, sep]
        for true in self.labels:
            cells = []
            for pred in self.labels:
                n = self.count(true, pred)
                cells.append(f"**{n}**" if true == pred else str(n))
            lines.append(f"| **{true}** | " + " | ".join(cells) + f" | {self.row_total(true)} |")
        col_totals = " | ".join(str(self.col_total(x)) for x in self.labels)
        lines.append(f"| **合计** | {col_totals} | {self.total} |")
        body = "\n".join(lines)
        return f"{caption}\n\n{body}\n" if caption else body + "\n"

    def render_metrics_markdown(self) -> str:
        macro = self.macro()
        lines = [
            "| 等级 | 支持数 | 预测数 | 精确率 | 召回率 | F1 |",
            "|---|---|---|---|---|---|",
        ]
        lines.extend(
            f"| {r.label} | {r.support} | {r.predicted} | "
            f"{r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |"
            for r in self.per_class()
        )
        lines.append("")
        lines.append(
            f"- **准确率** {macro.accuracy:.3f}（{macro.n} 个样本）"
            f"　**宏平均** P {macro.macro_precision:.3f} / R {macro.macro_recall:.3f} / "
            f"F1 {macro.macro_f1:.3f}　**加权 F1** {macro.weighted_f1:.3f}"
            f"　**Cohen's κ** {macro.kappa:.3f}"
        )
        return "\n".join(lines)


# --- 等级相关 -----------------------------------------------------------------


def ranks(values: Sequence[float]) -> list[float]:
    """平均秩（并列取平均），用于 Spearman。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 秩从 1 开始
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 等级相关系数。

    用途：标签只有五档、预测只有三档，逐行"对不对"丢失了大量信息。
    等级相关回答的是另一个问题——**引擎给出的分数有没有随污染程度单调上升**。
    即使逐行准确率不高，只要单调性强，说明规则方向是对的、只是分档线没对齐。
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    # 全等值时无法定义相关，直接返回 0（避免除零后返回 nan）
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    return pearson(ranks(xs), ranks(ys))


# --- 分布描述 -----------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def describe(values: Sequence[float]) -> dict[str, float]:
    """count / min / p25 / median / p75 / max / mean。用于"分数随标签单调吗"。"""
    if not values:
        return {"count": 0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0, "mean": 0.0}
    sv = sorted(values)
    return {
        "count": len(sv),
        "min": round(float(sv[0]), 3),
        "p25": round(_percentile(sv, 0.25), 3),
        "median": round(_percentile(sv, 0.5), 3),
        "p75": round(_percentile(sv, 0.75), 3),
        "max": round(float(sv[-1]), 3),
        "mean": round(sum(sv) / len(sv), 3),
    }
