"""M8 的「人工标签」评测：用独立标注的真值集检验引擎，而不是自己跟自己对答案。

## 为什么要有这个模块

第一组（`runner`）的标签是**按同一套阈值现算出来的**，它量的是「两套口径
（最坏指标决定论 vs 加权评分）的分歧」——72% 那个数字只证明「代码没写错、
两条实现基本一致」，**不能当准确率引用**。

要回答「引擎判得对不对」，真值必须来自**人**：从同一批数据里抽样，由懂行的人
逐条看读数判级，再与引擎的判定分开算。本模块提供这条链路的三件套：

1. 抽样（`sample_windows`）——分层、去重、可复现（固定 seed）；
2. 清单读写（`write_sample` / `read_sample`）——CSV 给标注人改，JSON 给评测用；
3. 评测（`evaluate`）——只吃 `human_level` 非空的行，指标口径与第一组一致，
   另加「高危漏判率」和「相邻等级一致率」。

## 目录约定

`data/eval_groundtruth/`：

- `labels.csv`——抽样清单 + 人工标注列（`human_level` / `annotator` / `note`）；
- `windows.json`——每个样本的原始读数，评测时原样喂给引擎，保证两边看的是同一段数据；
- `manifest.json`——抽样参数（样本量、种子、抽自哪些文件、生成时间）；
- `README.md`——标注协议（见 `scripts/make_groundtruth_sample.py` 生成的模板）。

**与训练集完全隔离**：这个目录只被
`scripts/make_groundtruth_sample.py` 与 `scripts/evaluate_groundtruth.py` 读写，
任何训练脚本都不碰它；`check_isolation()` 在抽样时校验它不在 `data/derived/`、
`data/datasets/` 之下，也不与图像训练集重叠。

## 不足样本数就不出结论

`n_labeled < MIN_LABELS` 时状态记 `pending_labels`，**不计算任何准确率**——
宁可显示「待标注」，也不拿 5 条样本算一个漂亮数字。
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .labels import RISK_LEVELS
from .metrics import ConfusionMatrix
from .ponds import WINDOW_HOURS, PondSample, PondWindow, iter_windows
from .runner import run_engine

__all__ = [
    "DEFAULT_DIR",
    "DEFAULT_N",
    "DEFAULT_SEED",
    "GT_DIRNAME",
    "INDICATOR_COLUMNS",
    "LABEL_COLUMNS",
    "LEVELS",
    "MIN_LABELS",
    "LabeledWindow",
    "WindowSample",
    "check_isolation",
    "evaluate",
    "labeled_rows",
    "read_sample",
    "sample_windows",
    "write_sample",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 人工标注目录名（相对 `data/`）。
GT_DIRNAME = "eval_groundtruth"

DEFAULT_DIR = PROJECT_ROOT / "data" / GT_DIRNAME

#: 等级集合与第一组保持一致，顺序即混淆矩阵的轴。
LEVELS: tuple[str, ...] = tuple(RISK_LEVELS)

#: 少于此数不出结论。
MIN_LABELS = 20

DEFAULT_N = 100
DEFAULT_SEED = 42

#: 标注人要看的指标列（与 `PondSample.indicators()` 同名，便于对照）。
INDICATOR_COLUMNS: tuple[str, ...] = (
    "water_temp",
    "ph",
    "dissolved_oxygen",
    "ammonia_n",
    "nitrite_n",
)

#: `labels.csv` 的列。前 6 列是标识与时间，中间是指标，最后 4 列是标注位。
LABEL_COLUMNS: tuple[str, ...] = (
    "window_id",
    "pond_id",
    "started_at",
    "ended_at",
    "n_points",
    *INDICATOR_COLUMNS,
    "free_ammonia_mg_l",
    "human_level",
    "annotator",
    "note",
)

#: 需要人工填的三列。
HUMAN_COLUMNS: tuple[str, ...] = ("human_level", "annotator", "note")


# --------------------------------------------------------------------------- #
# 抽样
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowSample:
    """一个待标注样本：一段窗口的读数 + 展示用的末点指标。"""

    window_id: str
    pond_id: str
    started_at: datetime
    ended_at: datetime
    readings: tuple[dict, ...]
    values: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.readings)

    def free_ammonia(self) -> float | None:
        """末点折算出的非离子氨（mg/L）——只是给标注人看的参考值。

        注意 `unionized_ammonia()` 返回的是 `(占比, 非离子氨浓度)`，
        别把占比当浓度——第一版就写错了，0.0004 mg/L 的总氨氮被算成 0.0752。
        """
        from ..tools.water_risk import unionized_ammonia as _calc

        total = self.values.get("ammonia_n")
        ph = self.values.get("ph")
        temp = self.values.get("water_temp")
        if total is None or ph is None or temp is None:
            return None
        _fraction, free = _calc(float(total), float(ph), float(temp))
        return free

    def to_row(self) -> dict[str, str]:
        """转成 `labels.csv` 的一行；人工列留空。"""
        free = self.free_ammonia()
        row = {
            "window_id": self.window_id,
            "pond_id": self.pond_id,
            "started_at": self.started_at.isoformat(timespec="minutes"),
            "ended_at": self.ended_at.isoformat(timespec="minutes"),
            "n_points": str(self.n_points),
            "free_ammonia_mg_l": "" if free is None else f"{free:.5f}",
            "human_level": "",
            "annotator": "",
            "note": "",
        }
        for name in INDICATOR_COLUMNS:
            value = self.values.get(name)
            row[name] = "" if value is None else f"{float(value):.4f}"
        return row


def _oxygen_bucket(value: float | None) -> str:
    """抽样分层用的溶氧桶。

    为什么按溶氧分：它是这套系统里最关键的指标，而低氧窗口在数据里占比不高，
    纯随机抽样容易抽成一堆「正常」。这里只按**读数本身**分桶，
    不用引擎的评分，也不用阈值标签器的判定，避免把答案混进抽样阶段。
    """
    if value is None:
        return "缺测"
    if value < 3.0:
        return "低氧"
    if value < 5.0:
        return "偏低"
    return "充足"


def sample_windows(
    by_pond: Mapping[str, Sequence[PondSample]],
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    window_hours: float = WINDOW_HOURS,
    stride_hours: float | None = None,
    min_gap_hours: float = 24.0,
) -> list[WindowSample]:
    """分层随机抽 `n` 个窗口，同一口塘的样本至少相隔 `min_gap_hours`。

    分层键是末点溶氧桶（低氧 / 偏低 / 充足 / 缺测），层间轮转抽取，
    免得低氧样本被大量「正常」窗口淹没。抽完按 `(pond_id, started_at)` 排序
    再编号，保证同一份数据 + 同一个 seed 得到完全一样的清单。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if min_gap_hours < 0:
        raise ValueError("min_gap_hours 不能为负")

    stride = window_hours if stride_hours is None else stride_hours
    rng = random.Random(seed)

    pool: dict[str, list[PondWindow]] = {}
    for window in iter_windows(by_pond, window_hours=window_hours, stride_hours=stride):
        last = window.readings[-1] if window.readings else {}
        pool.setdefault(_oxygen_bucket(last.get("dissolved_oxygen")), []).append(window)

    buckets = [key for key in ("低氧", "偏低", "充足", "缺测") if pool.get(key)]
    picked: list[PondWindow] = []
    last_seen: dict[str, list[datetime]] = {}
    attempts = 0
    max_attempts = max(n * 500, 5000)

    while len(picked) < n and buckets and attempts < max_attempts:
        attempts += 1
        for bucket in list(buckets):
            candidates = pool[bucket]
            if not candidates:
                buckets.remove(bucket)
                continue
            window = candidates.pop(rng.randrange(len(candidates)))
            seen = last_seen.setdefault(window.pond_id, [])
            too_close = any(
                abs((window.started_at - moment).total_seconds()) < min_gap_hours * 3600.0
                for moment in seen
            )
            if too_close:
                continue
            seen.append(window.started_at)
            picked.append(window)
            if len(picked) >= n:
                break

    picked.sort(key=lambda w: (w.pond_id, w.started_at))
    return [
        WindowSample(
            window_id=f"GT{index:04d}",
            pond_id=window.pond_id,
            started_at=window.started_at,
            ended_at=window.ended_at,
            readings=tuple(window.readings),
            values=_values_of(window.readings[-1] if window.readings else {}),
        )
        for index, window in enumerate(picked, start=1)
    ]


def _values_of(point: Mapping[str, object]) -> dict[str, float | None]:
    """只取五个指标，缺的留 `None`——不拿 0 顶替（0 会让标注人看到假读数）。"""
    values: dict[str, float | None] = {}
    for name in INDICATOR_COLUMNS:
        raw = point.get(name)
        values[name] = None if raw is None else float(raw)  # type: ignore[arg-type]
    return values


# --------------------------------------------------------------------------- #
# 清单读写
# --------------------------------------------------------------------------- #


def check_isolation(directory: Path) -> None:
    """抽样目录必须与训练集/派生数据分开，否则「独立真值」不成立。

    只禁止落在 `data/derived/`、`data/datasets/` 之下；临时目录是允许的
    （评测集可以先放仓库外，标完再拷回来）。
    """
    resolved = directory.resolve()
    forbidden = (
        PROJECT_ROOT / "data" / "derived",
        PROJECT_ROOT / "data" / "datasets",
    )
    for base in forbidden:
        base_resolved = base.resolve()
        if resolved == base_resolved or base_resolved in resolved.parents:
            raise ValueError(
                f"人工标注集不能放在 {base_resolved} 下（那是训练/派生数据的目录），"
                f"当前是 {resolved}"
            )


def write_sample(
    rows: Sequence[WindowSample],
    directory: Path = DEFAULT_DIR,
    *,
    meta: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """写出 `labels.csv`、`windows.json`、`manifest.json`。"""
    check_isolation(directory)
    directory.mkdir(parents=True, exist_ok=True)

    labels_path = directory / "labels.csv"
    windows_path = directory / "windows.json"
    manifest_path = directory / "manifest.json"

    # utf-8-sig：Excel 直接双击打开时中文不乱码；读取侧同样按 utf-8-sig 处理。
    with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())

    windows_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "window_id": row.window_id,
                        "pond_id": row.pond_id,
                        "started_at": row.started_at.isoformat(),
                        "ended_at": row.ended_at.isoformat(),
                        "readings": list(row.readings),
                    }
                    for row in rows
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = {
        "n_samples": len(rows),
        "levels": list(LEVELS),
        "min_labels": MIN_LABELS,
        "columns": list(LABEL_COLUMNS),
        "human_columns": list(HUMAN_COLUMNS),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **(dict(meta) if meta else {}),
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "labels": labels_path,
        "windows": windows_path,
        "manifest": manifest_path,
    }


@dataclass(frozen=True)
class LabeledWindow:
    """清单里的一行：读数（来自 `windows.json`）+ 人工标注（来自 `labels.csv`）。"""

    window_id: str
    pond_id: str
    started_at: datetime
    ended_at: datetime
    values: Mapping[str, float | None]
    readings: tuple[dict, ...]
    human_level: str = ""
    annotator: str = ""
    note: str = ""

    @property
    def labeled(self) -> bool:
        return bool(self.human_level)


def _parse_float(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:  # 标注人填了「约 0.4」之类
        raise ValueError(f"无法解析为数字：{text!r}") from exc


def read_sample(directory: Path = DEFAULT_DIR) -> list[LabeledWindow]:
    """读回清单。`windows.json` 提供读数，`labels.csv` 提供人工标注。

    校验：`window_id` 唯一、`human_level` 只能是三个等级之一或留空。
    """
    directory = Path(directory)
    labels_path = directory / "labels.csv"
    windows_path = directory / "windows.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} 不存在：先跑 scripts/make_groundtruth_sample.py 抽样"
        )
    if not windows_path.exists():
        raise FileNotFoundError(f"{windows_path} 不存在，抽样清单不完整")

    payload = json.loads(windows_path.read_text(encoding="utf-8"))
    readings_by_id: dict[str, tuple[dict, ...]] = {}
    for example in payload.get("examples", []):
        readings_by_id[example["window_id"]] = tuple(example.get("readings") or ())

    rows: list[LabeledWindow] = []
    seen: set[str] = set()
    with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_no, raw in enumerate(csv.DictReader(handle), start=2):
            window_id = (raw.get("window_id") or "").strip()
            if not window_id:
                raise ValueError(f"{labels_path}:{line_no} 缺少 window_id")
            if window_id in seen:
                raise ValueError(f"{labels_path}:{line_no} window_id 重复：{window_id}")
            seen.add(window_id)

            level = (raw.get("human_level") or "").strip()
            if level and level not in LEVELS:
                raise ValueError(
                    f"{labels_path}:{line_no} human_level={level!r} 不在 {LEVELS} 里"
                )
            if not level and (raw.get("annotator") or "").strip():
                raise ValueError(
                    f"{labels_path}:{line_no} 填了标注人却没填 human_level"
                )

            values = {name: _parse_float(raw.get(name)) for name in INDICATOR_COLUMNS}
            rows.append(
                LabeledWindow(
                    window_id=window_id,
                    pond_id=(raw.get("pond_id") or "").strip(),
                    started_at=datetime.fromisoformat(raw["started_at"]),
                    ended_at=datetime.fromisoformat(raw["ended_at"]),
                    values=values,
                    readings=readings_by_id.get(window_id, ()),
                    human_level=level,
                    annotator=(raw.get("annotator") or "").strip(),
                    note=(raw.get("note") or "").strip(),
                )
            )
    return rows


def labeled_rows(rows: Iterable[LabeledWindow]) -> list[LabeledWindow]:
    """只留下人工填过等级的样本。"""
    return [row for row in rows if row.labeled]


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #


def _ordinal(level: str) -> int:
    return LEVELS.index(level)


def evaluate(
    rows: Sequence[LabeledWindow],
    *,
    species: str = "鲤科",
    horizon_hours: int = 72,
    min_labels: int = MIN_LABELS,
) -> dict:
    """拿人工标签当唯一真值，量引擎判级。

    `pending_labels` 时只回报「标了多少 / 还差多少」，不算任何准确率。
    """
    labelled = labeled_rows(rows)
    base = {
        "metric_kind": "human_label_accuracy",
        "metric_label": "人工标签准确率（独立真值）",
        "levels": list(LEVELS),
        "n_sampled": len(rows),
        "n_labeled": len(labelled),
        "n_missing": len(rows) - len(labelled),
        "min_labels": min_labels,
        "species": species,
        "horizon_hours": horizon_hours,
    }
    if len(labelled) < min_labels:
        return {
            **base,
            "status": "pending_labels",
            "reason": (
                f"人工标注只有 {len(labelled)} 条，少于 {min_labels} 条，"
                "按约定不出指标（宁可显示待标注，也不拿几条样本算准确率）"
            ),
            "annotators": sorted({row.annotator for row in labelled if row.annotator}),
        }

    windows = [
        PondWindow(
            pond_id=row.pond_id,
            readings=row.readings,
            started_at=row.started_at,
            ended_at=row.ended_at,
            true_level=row.human_level,
        )
        for row in labelled
    ]
    samples, stats = run_engine(windows, species=species, horizon_hours=horizon_hours)

    pairs: list[tuple[str, str]] = []
    failed = 0
    for label, sample in zip(labelled, samples):
        if sample.status != "ok" or not sample.risk_level:
            failed += 1
            continue
        pairs.append((label.human_level, sample.risk_level))

    confusion = ConfusionMatrix.from_pairs(pairs, LEVELS)
    macro = confusion.macro()

    def _rate(predicate) -> float | None:
        hits = [
            (true, pred)
            for true, pred in pairs
            if true == predicate[0]
        ]
        if not hits:
            return None
        return sum(1 for _, pred in hits if pred == predicate[1]) / len(hits)

    high_total = sum(1 for true, _ in pairs if true == "高危")
    high_hit = sum(1 for true, pred in pairs if true == "高危" and pred == "高危")
    normal_total = sum(1 for true, _ in pairs if true == "正常")
    normal_alarm = sum(1 for true, pred in pairs if true == "正常" and pred == "高危")
    adjacent = sum(1 for true, pred in pairs if abs(_ordinal(true) - _ordinal(pred)) <= 1)

    return {
        **base,
        "status": "ok" if pairs else "no_scored_samples",
        "n_scored": len(pairs),
        "n_engine_failed": failed,
        "confusion": confusion.to_dict(),
        "markdown": confusion.render_markdown("人工标签 × 引擎判定"),
        "accuracy": round(macro.accuracy, 4),
        "kappa": round(macro.kappa, 4),
        "macro_precision": round(macro.macro_precision, 4),
        "macro_recall": round(macro.macro_recall, 4),
        "macro_f1": round(macro.macro_f1, 4),
        "high_risk_recall": None if not high_total else round(high_hit / high_total, 4),
        "high_risk_missed": high_total - high_hit,
        "false_alarm_rate": None if not normal_total else round(normal_alarm / normal_total, 4),
        "adjacent_agreement": None if not pairs else round(adjacent / len(pairs), 4),
        "annotators": sorted({row.annotator for row in labelled if row.annotator}),
        "engine_stats": stats,
    }
