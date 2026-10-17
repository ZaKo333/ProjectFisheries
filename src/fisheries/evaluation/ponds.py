"""PondsData 读取、量纲归一与清洗：把鱼菜共生塘 IoT 时序变成引擎能吃的样本。

## 换成这份数据集的理由

原来两组外部验证用的都是河川/地表水数据（`china_water_pollution_data.csv`、
`Dissolved oxygen prediction in river water`），与本项目的场景（**池塘养殖**）
不是一回事，而且那份污染分级数据的标签被证明没有刻度。现在两组都改用：

`Sensor Based Aquaponics Fish Pond Datasets`（`fisheries.datasets` 里叫
`aquaponics_pond`）——11 个 CSV、1,114,970 行、约 20 秒一条，覆盖
2021-06-18 ～ 2021-10-31，每条读数都带塘号、时间戳、鱼体长体重。

## 这份数据的坑（每一条都实测过）

**1. 表头有四种写法，而且有拼写错误。** IoTPond1/2 用
`Temperature (C)`，IoTPond3/4/6/9 用 `Temperature(C)`，IoTPond10/11/12 全大写
且把 `Dissolved` 拼成 `DISOLVED`、IoTPond11 把 `Length` 拼成 `Lenght`。
所以列名一律按「去掉全部空白 + 转小写」归一后再查别名表。

**2. IoTPond7/IoTPond8 的表头里有空列名**（IoTPond8 的第一列、IoTPond7 的末三列），
`csv.DictReader` 会把它们都塞成 `''` 键、后面的覆盖前面的，
于是「时间戳在第 0 列」这件事在字典视图里就丢了。**本模块一律按列下标取值**。

**3. 时间戳有两种格式**，全部带时区名或偏移：
`2021-06-19 00:00:05 CET`（10 个文件，另有 `UTC`）与
`2021-06-18T13:06:49+01:00`（IoTPond8）。统一解析成 UTC 后再分组。
**CET 不是固定偏移**，这里按 UTC+1 处理（数据集里没有 CEST 的观测）。

**4. ⚠️ 量纲标注是错的，而且是会改变结论的那种错。**
`Dissolved Oxygen(g/ml)` / `Ammonia(g/ml)` / `Nitrate(g/ml)` 里的
`(g/ml)` 不是真实单位（g/ml 比水重 1000 倍），是复制粘贴留下的垃圾。
真实单位只能从数值分布 + **一个独立判据**反推——同一份数据里记着鱼的体长体重：

| 列 | 组内中位数 | 若按 mg/L 读的含义 | 结论 |
|---|---|---|---|
| 溶氧 | 3.28 – 18.31 | 合理（25 ℃ 饱和约 8.4） | **就是 mg/L**，不换算 |
| 氨氮 | 0.55 – 60.4 | 中位数已越过 0.2 的适宜线、常越过 1.0 致死红线 | **是 µg/L**，÷1000 |
| 硝酸盐 | 0 – 4095 | 4095 mg/L 不可能（>200 已中毒） | **是 µg/L**，÷1000 |

独立判据是鱼自己的生长：IoTPond7 的体重从 8.0 g 长到 998.5 g（**+12,360%**），
IoTPond6 从 3.3 g 到 231 g（+6,900%）。**鱼不可能在一半时间泡在氨氮致死线以上
还长成这样。** 反过来说，溶氧若按 µg/L 读就是 0.003 mg/L——鱼会当场死，
所以溶氧只能是 mg/L。同一个荒谬的标注，在两个列上指向相反的真相。

IoTPond8 是唯一标了 `(mg/L)` 的文件，它的氨氮中位数 0.00111、最大 0.9966，
恰好和其余文件 ÷1000 之后同一量级——这是上面这套推断的旁证。

**5. 哨兵值与坏读数。** 水温有 `-127`（9/11 个文件）；pH 最小值 `-3.137`
在 5 个文件里重复出现（是错误码不是读数），另有 pH 到 15.45/16.23 的；
溶氧最大 44.98 mg/L（25 ℃ 饱和约 8.4，物理上不可能）；氨氮里混着字面量
`inf` / `nan`；浊度有 -97 这样的负数。这些一律按物理量程丢弃，
**不是**当成 0 或均值填补——那会把传感器故障伪装成水质事件。

**6. 只有 44.55% 的行四个必填指标齐全且在量程内**（496,677 / 1,114,970）。
逐文件从 IoTPond8 的 0.00%、IoTPond12 的 0.72% 到 IoTPond6 的 82.93%。
丢了三分之二不丢人，丢了多少必须写清楚。

**7. 数据集里根本没有亚硝酸盐列**，所以 `nitrite_n` 恒为 `None`，
引擎的 `nitrite_opt` / `NO2_REDLINE` 在这份数据上永远不触发。报告要说明。

## 聚合与滑窗

原始 20 秒一条太密（1.1 M 行），也低于引擎 `interval_minutes` 的整数分钟粒度。
所以每个塘先按 **10 分钟**分桶取均值（`AGGREGATE_MINUTES`），
缺测的指标在该桶里为 `None`；必填项仍缺的桶整桶丢弃。

窗口取 **连续 3 小时**（18 个聚合点，`WINDOW_HOURS`），**不重叠**
（步长 = 窗长）。不重叠是刻意的：原来那份评测报告自己承认
「窗口滑动使同一条记录出现在多个样本里，样本之间不独立」，
这里直接把它消掉。

真值取**窗口末点**的阈值标签——与引擎看到的最后一条读数完全对齐
（引擎判的是「当前状态 + 趋势」），所以两边比的是同一件事。
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from .. import datasets as _datasets
from .labels import (
    PondOutcome,
    label_distribution,
    label_indicators,
    worst_indicators,
)

__all__ = [
    "AGGREGATE_MINUTES",
    "COLUMN_ALIASES",
    "DATASET_NAME",
    "DEFAULT_DIR",
    "RANGES",
    "REQUIRED_FIELDS",
    "STRIDE_HOURS",
    "UNIT_DIVISORS",
    "WINDOW_HOURS",
    "PondSample",
    "PondWindow",
    "iter_windows",
    "load_pond",
    "load_ponds",
    "parse_timestamp",
    "pond_outcomes",
    "summarize_samples",
]

#: 数据集注册表里的名字。数据在哪由 `fisheries.datasets` 统一解析，
#: 绝不写死绝对路径。用 ``default_path`` 而非 ``require``：import 期不该失败。
DATASET_NAME = "aquaponics_pond"
DEFAULT_DIR = _datasets.default_path(DATASET_NAME)

#: 规则引擎必填的四项。缺任何一项这条读数就不能进聚合。
REQUIRED_FIELDS: tuple[str, ...] = ("water_temp", "ph", "dissolved_oxygen", "ammonia_n")

#: 表头归一化后的别名 → 内部字段名。
#: 归一化 = 去掉全部空白字符 + 转小写，所以 `Temperature (C)`、`Temperature(C)`、
#: `TEMPERATURE` 都落到 `temperature(c)` / `temperature`。
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "created_at": ("created_at",),
    "water_temp": ("temperature(c)", "temperature"),
    "turbidity": ("turbidity(ntu)", "turbidity"),
    "dissolved_oxygen": (
        "dissolvedoxygen(g/ml)",
        "dissolvedoxygen(mg/l)",
        "disolvedoxygen",  # IoTPond10/11/12 把 Dissolved 拼错了
    ),
    "ph": ("ph",),
    "ammonia_n": ("ammonia(g/ml)", "ammonia(mg/l)", "ammonia"),
    "nitrate": ("nitrate(g/ml)", "nitrate(mg/l)", "nitrate"),
    "population": ("population",),
    "fish_length": (
        "fish_length(cm)",
        "total_length(cm)",
        "length",
        "lenght",  # IoTPond11 的拼写错误
    ),
    "fish_weight": ("fish_weight(g)", "weight(g)", "weight"),
}

#: 标了这些后缀的**浓度**列要按下面的除数归一。见模块 docstring 第 4 条。
_G_ML = "(g/ml)"
_MG_L = "(mg/l)"

#: 归一化除数：`(g/ml)` 标注的浓度列是 µg/L，÷1000 得到 mg/L。
#: **溶氧不在其中**——它的数值只能是 mg/L，÷1000 会变成鱼会当场死掉的量级。
UNIT_DIVISORS: dict[str, float] = {
    "ammonia_n": 1000.0,
    "nitrate": 1000.0,
}

#: 物理量程（归一化之后的单位）。超出即视为坏读数／哨兵值，置 ``None``。
RANGES: dict[str, tuple[float, float]] = {
    "water_temp": (0.0, 45.0),  # 常见哨兵 -127 由此丢弃
    "ph": (0.0, 14.0),  # 常见哨兵 -3.137 / 15.45 由此丢弃
    "dissolved_oxygen": (0.0, 20.0),  # 25 ℃ 饱和约 8.4，20 已极宽
    "ammonia_n": (0.0, 50.0),  # mg/L；50 已是极端值
    "nitrite_n": (0.0, 50.0),
    "turbidity": (0.0, 4000.0),  # NTU；实测有 -97 这样的负数
    "nitrate": (0.0, 500.0),  # mg/L
    "fish_length": (0.0, 200.0),  # cm
    "fish_weight": (0.0, 20000.0),  # g
    "population": (0.0, 1e9),
}

#: 聚合桶宽（分钟）。见模块 docstring「聚合与滑窗」。
AGGREGATE_MINUTES = 10
#: 一个评测窗口覆盖的时长（小时）= 18 个聚合点。
WINDOW_HOURS = 3.0
#: 窗口步长（小时）。等于窗长 = 不重叠，样本之间独立。
STRIDE_HOURS = 3.0

#: 时区名 → 相对 UTC 的小时偏移。数据集里实际只出现 CET / UTC / +01:00。
_TZ_OFFSETS: dict[str, int] = {
    "UTC": 0,
    "GMT": 0,
    "WET": 0,
    "WEST": 1,
    "CET": 1,
    "CEST": 2,
    "EST": -5,
    "EDT": -4,
}
_TZ_SUFFIX = re.compile(r"\s+([A-Za-z]{2,5})$")
_TS_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
)


# ---------------------------------------------------------------------------
# 解析原语
# ---------------------------------------------------------------------------


def parse_timestamp(raw: str | None) -> datetime | None:
    """把数据集里的两种时间戳格式解析成带 UTC 时区的 ``datetime``。

    >>> parse_timestamp("2021-06-19 00:00:05 CET").isoformat()
    '2021-06-18T23:00:05+00:00'
    >>> parse_timestamp("2021-06-18T13:06:49+01:00").isoformat()
    '2021-06-18T12:06:49+00:00'
    """
    text = (raw or "").strip()
    if not text:
        return None

    # ISO 8601（含偏移或 Z）——IoTPond8 走这条。
    try:
        iso = datetime.fromisoformat(text.replace("Z", "+00:00") if text.endswith("Z") else text)
    except ValueError:
        iso = None
    if iso is not None:
        if iso.tzinfo is None:
            iso = iso.replace(tzinfo=timezone.utc)
        return iso.astimezone(timezone.utc)

    # 「本地时间 + 时区名」——其余 10 个文件走这条。
    offset_hours = 0
    match = _TZ_SUFFIX.search(text)
    if match and match.group(1).upper() in _TZ_OFFSETS:
        offset_hours = _TZ_OFFSETS[match.group(1).upper()]
        text = text[: match.start()].strip()

    for fmt in _TS_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return (naive - timedelta(hours=offset_hours)).replace(tzinfo=timezone.utc)
    return None


def _to_float(raw: str | None) -> float | None:
    """转 float；空串、``inf``、``nan``、非数字一律返回 ``None``。

    ``inf`` / ``nan`` 在这份数据里真实存在（氨氮列），必须当缺测而不是当数值——
    否则一个 ``inf`` 会把整桶的均值变成 ``inf``，再被判成高危。
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _normalized(raw: str) -> str:
    return "".join(raw.split()).lower()


def resolve_columns(header: Sequence[str]) -> dict[str, tuple[int, str]]:
    """表头 → `{内部字段: (列下标, 原始表头)}`。

    按列下标定位，避免 ``DictReader`` 遇到多个空列名时互相覆盖
    （IoTPond7/IoTPond8 的表头里就有空列名）。时间戳列找不到名字时退回第 0 列。
    """
    resolved: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(header):
        key = _normalized(raw)
        if not key:
            continue
        for field_name, aliases in COLUMN_ALIASES.items():
            if key in aliases and field_name not in resolved:
                resolved[field_name] = (index, raw)
                break
    if "created_at" not in resolved and header:
        # IoTPond8 的时间戳列没有列名。
        resolved["created_at"] = (0, header[0])
    return resolved


def _ceil_to_bucket(moment: datetime, minutes: int) -> datetime:
    """把时刻对齐到 ``minutes`` 分钟的整点桶（向下取整）。"""
    epoch = int(moment.timestamp())
    width = minutes * 60
    return datetime.fromtimestamp(epoch - epoch % width, tz=timezone.utc)


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PondSample:
    """一个聚合点（默认 10 分钟均值）——引擎真正吃到的东西。"""

    pond_id: str
    sampled_at: datetime
    water_temp: float
    ph: float
    dissolved_oxygen: float
    ammonia_n: float
    nitrite_n: float | None = None
    turbidity: float | None = None
    #: 这个桶里参与聚合的原始读数条数。
    n_raw: int = 0

    def indicators(self) -> dict[str, float | None]:
        """分级表要的五个指标。"""
        return {
            "water_temp": self.water_temp,
            "ph": self.ph,
            "dissolved_oxygen": self.dissolved_oxygen,
            "ammonia_n": self.ammonia_n,
            "nitrite_n": self.nitrite_n,
        }

    def to_point(self) -> dict:
        """转成 `WaterQualityPoint` 的字典形态。

        只带引擎认得的键——``WaterQualityPoint`` 是 ``extra="forbid"``，
        多塞一个 ``nitrate`` 会让整个请求被 pydantic 拒绝。
        """
        point: dict = {"timestamp": self.sampled_at.isoformat()}
        point.update({k: v for k, v in self.indicators().items() if v is not None})
        return point


@dataclass(frozen=True)
class PondWindow:
    """一个待评测样本：一段 3 小时时序 + 它的真值。"""

    pond_id: str
    readings: tuple[dict, ...]
    started_at: datetime
    ended_at: datetime
    true_level: str
    #: 导致该等级的指标（来自窗口末点）。
    limiting: tuple[str, ...] = ()

    @property
    def sampled_on(self) -> date:
        return self.ended_at.date()

    @property
    def hours(self) -> float:
        return (self.ended_at - self.started_at).total_seconds() / 3600.0

    @property
    def n_points(self) -> int:
        return len(self.readings)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    """聚合中的桶。"""

    pond_id: str
    sampled_at: datetime
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, values: Mapping[str, float | None]) -> None:
        for name, value in values.items():
            if value is None:
                continue
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + 1


def _growth_pct(weights: Sequence[float]) -> float | None:
    """首段与末段各取 10% 的中位数，算体重增长率（%）。

    用中位数而不是首末单点：原始读数里的体重列有跳变，
    拿单点算增长率会被一个坏点带偏。
    """
    clean = [w for w in weights if w is not None]
    if len(clean) < 20:
        return None
    head = median(clean[: len(clean) // 10])
    tail = median(clean[-len(clean) // 10 :])
    if head <= 0:
        return None
    return (tail - head) / head * 100.0


def load_pond(
    path: str | Path,
    *,
    aggregate_minutes: int = AGGREGATE_MINUTES,
) -> tuple[list[PondSample], dict]:
    """读一个池塘 CSV，做量纲归一 + 量程清洗 + 时间聚合。返回 `(samples, stats)`。

    ``stats`` 记录读了多少行、每个字段丢了多少、聚合出多少点、体重增长多少——
    评测报告必须能说清样本是怎么从 1,114,970 行里筛出来的，否则数字不可信。
    """
    path = Path(path)
    if not path.is_file():
        # 交给注册表抛错：它的消息里带着「数据可以放哪、怎么下载」。
        _datasets.require(DATASET_NAME)
        raise FileNotFoundError(f"找不到数据集：{path}")

    pond_id = path.stem
    stats: dict = {
        "pond_id": pond_id,
        "path": str(path),
        "n_lines": 0,
        "n_bad_timestamp": 0,
        "n_dropped_missing_required": 0,
        "n_out_of_range": 0,
        "out_of_range_by_field": Counter(),
        "n_samples": 0,
        "growth_pct": None,
        "columns": {},
    }
    buckets: dict[datetime, _Bucket] = {}
    weights: list[float] = []

    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [], stats
        columns = resolve_columns(header)
        stats["columns"] = {name: raw for name, (_, raw) in columns.items()}
        if "created_at" not in columns:
            raise ValueError(f"{path.name} 里找不到时间戳列")

        def cell(row: Sequence[str], field_name: str) -> float | None:
            """读一列 → 量纲归一 → 量程清洗。越界/哨兵返回 ``None``。

            **顺序不能反**：``RANGES`` 写的是归一化之后的量的量程，
            先查量程再换算会把「合法但看着很大」的 µg/L 读数当成坏值丢掉
            （氨氮的 µg/L 原始值上限是 mg/L 的 1000 倍，先查量程等于只保留
            0.05 mg/L 以下的读数，而且是静默地丢）。
            """
            position = columns.get(field_name)
            if position is None or position[0] >= len(row):
                return None
            value = _to_float(row[position[0]])
            if value is None:
                return None
            if field_name in UNIT_DIVISORS:
                value = value / UNIT_DIVISORS[field_name]
            low, high = RANGES[field_name]
            if field_name == "fish_weight":
                # 体重既不进引擎入参也不参与标签，只用来算生长率。
                # 它不该影响「清洗掉多少读数」的统计，所以只记数、不计数。
                if low <= value <= high:
                    weights.append(value)
                return value
            if value < low or value > high:
                stats["n_out_of_range"] += 1
                stats["out_of_range_by_field"][field_name] += 1
                return None
            return value

        for row in reader:
            stats["n_lines"] += 1
            raw_ts = row[columns["created_at"][0]] if columns["created_at"][0] < len(row) else ""
            moment = parse_timestamp(raw_ts)
            if moment is None:
                stats["n_bad_timestamp"] += 1
                continue

            # 体重只用来算生长率，供 `labels.growth_coherence` 做那条
            # 「不依赖自建阈值」的合理性检验；返回值本身丢掉。
            # 它放在必填门之前：鱼的生长是塘口的事实，不该因为某个传感器
            # 当掉了一会儿就被从统计里剔掉。
            cell(row, "fish_weight")

            # 只取引擎与标签真正会用到的列。`turbidity` / `nitrate` / `population`
            # / `fish_length` 读了也没人用——把它们算进「清洗掉多少读数」只会
            # 让第 1 节的数字虚高，看不出真正的样本损失在哪。
            values: dict[str, float | None] = {}
            for field_name in (*REQUIRED_FIELDS, "nitrite_n"):
                values[field_name] = cell(row, field_name)

            missing = [n for n in REQUIRED_FIELDS if values.get(n) is None]
            if missing:
                stats["n_dropped_missing_required"] += 1
                continue

            bucket_at = _ceil_to_bucket(moment, aggregate_minutes)
            bucket = buckets.get(bucket_at)
            if bucket is None:
                bucket = _Bucket(pond_id=pond_id, sampled_at=bucket_at)
                buckets[bucket_at] = bucket
            bucket.add(values)

    samples: list[PondSample] = []
    for bucket_at in sorted(buckets):
        bucket = buckets[bucket_at]
        means = {
            name: bucket.sums[name] / bucket.counts[name] for name in bucket.sums
        }
        if any(means.get(name) is None for name in REQUIRED_FIELDS):
            continue
        samples.append(
            PondSample(
                pond_id=pond_id,
                sampled_at=bucket_at,
                water_temp=float(means["water_temp"]),
                ph=float(means["ph"]),
                dissolved_oxygen=float(means["dissolved_oxygen"]),
                ammonia_n=float(means["ammonia_n"]),
                nitrite_n=means.get("nitrite_n"),
                turbidity=means.get("turbidity"),
                n_raw=max(bucket.counts.values()) if bucket.counts else 0,
            )
        )

    stats["n_samples"] = len(samples)
    stats["growth_pct"] = _growth_pct(weights)
    stats["out_of_range_by_field"] = dict(stats["out_of_range_by_field"])
    return samples, stats


def load_ponds(
    paths: Iterable[str | Path] | None = None,
    *,
    aggregate_minutes: int = AGGREGATE_MINUTES,
    limit_files: int | None = None,
) -> tuple[dict[str, list[PondSample]], dict]:
    """读全部池塘，返回 `({pond_id: [samples]}, stats)`。

    文件顺序按「名字长度 + 字母」排——这样 IoTPond2 排在 IoTPond10 前面，
    IoTPond1 排在 IoTPond2 前面，读起来跟人预期一致。
    """
    if paths is None:
        if not DEFAULT_DIR.is_dir():
            _datasets.require(DATASET_NAME)
            raise FileNotFoundError(f"找不到数据集目录：{DEFAULT_DIR}")
        selected = sorted(
            (p for p in DEFAULT_DIR.glob("*.csv") if p.is_file()),
            key=lambda p: (len(p.name), p.name.lower()),
        )
    else:
        selected = [Path(p) for p in paths]
    if limit_files is not None:
        selected = selected[:limit_files]

    by_pond: dict[str, list[PondSample]] = {}
    per_file: list[dict] = []
    for path in selected:
        samples, stats = load_pond(path, aggregate_minutes=aggregate_minutes)
        by_pond[stats["pond_id"]] = samples
        per_file.append(stats)

    totals = {
        "n_files": len(selected),
        "n_lines": sum(int(s["n_lines"]) for s in per_file),
        "n_samples": sum(int(s["n_samples"]) for s in per_file),
        "n_bad_timestamp": sum(int(s["n_bad_timestamp"]) for s in per_file),
        "n_dropped_missing_required": sum(
            int(s["n_dropped_missing_required"]) for s in per_file
        ),
        "n_out_of_range": sum(int(s["n_out_of_range"]) for s in per_file),
        "per_file": per_file,
    }
    return by_pond, totals


# ---------------------------------------------------------------------------
# 滑窗与概况
# ---------------------------------------------------------------------------


def iter_windows(
    by_pond: Mapping[str, Sequence[PondSample]],
    *,
    window_hours: float = WINDOW_HOURS,
    stride_hours: float = STRIDE_HOURS,
) -> Iterator[PondWindow]:
    """按「连续 ``window_hours``、步长 ``stride_hours``」切窗口。

    真值取窗口末点的阈值标签——引擎看到的就是这些点，两边比的是同一件事。
    默认步长等于窗长（不重叠），样本之间因此互相独立。
    """
    if window_hours <= 0 or stride_hours <= 0:
        raise ValueError("window_hours / stride_hours 必须为正")
    span = timedelta(hours=window_hours)
    step = timedelta(hours=stride_hours)

    for pond_id in sorted(by_pond):
        samples = list(by_pond[pond_id])
        if len(samples) < 3:
            continue
        cursor = samples[0].sampled_at
        last = samples[-1].sampled_at
        while cursor + span <= last + timedelta(seconds=1):
            chunk = [s for s in samples if cursor <= s.sampled_at < cursor + span]
            if len(chunk) >= 3:
                tail = chunk[-1]
                yield PondWindow(
                    pond_id=pond_id,
                    readings=tuple(s.to_point() for s in chunk),
                    started_at=chunk[0].sampled_at,
                    ended_at=tail.sampled_at,
                    true_level=label_indicators(tail.indicators()),
                    limiting=worst_indicators(tail.indicators()),
                )
            cursor += step


def summarize_samples(by_pond: Mapping[str, Sequence[PondSample]]) -> dict:
    """数据概况：聚合点数、塘数、时间范围、标签分布、逐塘样本量。

    ``per_pond`` 是 `{塘号: 聚合点数}`（只要一个数），
    ``per_pond_detail`` 是 `{塘号: {标签计数 + 高危占比}}`（给报告画表用）。
    两个都返回，是因为前者够用、后者够看，硬合并成一个会让两边都不好用。
    """
    all_samples = [s for samples in by_pond.values() for s in samples]
    labels = [label_indicators(s.indicators()) for s in all_samples]
    moments = [s.sampled_at for s in all_samples]
    per_pond = {pid: len(samples) for pid, samples in by_pond.items()}
    per_pond_detail: dict[str, dict] = {}
    for pid, samples in by_pond.items():
        counts = label_distribution([label_indicators(s.indicators()) for s in samples])
        total = sum(counts.values()) or 1
        per_pond_detail[pid] = {
            "n_samples": len(samples),
            **counts,
            "high_risk_rate": counts["高危"] / total,
        }
    return {
        "n_samples": len(all_samples),
        "n_ponds": len(by_pond),
        "label_counts": label_distribution(labels),
        "time_from": min(moments).isoformat() if moments else "",
        "time_to": max(moments).isoformat() if moments else "",
        "samples_per_pond_min": min(per_pond.values(), default=0),
        "samples_per_pond_max": max(per_pond.values(), default=0),
        "per_pond": per_pond,
        "per_pond_detail": per_pond_detail,
    }


def pond_outcomes(
    by_pond: Mapping[str, Sequence[PondSample]],
    growth_by_pond: Mapping[str, float | None],
) -> list[PondOutcome]:
    """逐塘汇总标签分布与体重增长，供 `labels.growth_coherence` 使用。"""
    out: list[PondOutcome] = []
    for pond_id in sorted(by_pond):
        samples = list(by_pond[pond_id])
        if not samples:
            continue
        labels = [label_indicators(s.indicators()) for s in samples]
        counts = label_distribution(labels)
        total = sum(counts.values()) or 1
        dissolved = [s.dissolved_oxygen for s in samples]
        ammonia = [s.ammonia_n for s in samples]
        out.append(
            PondOutcome(
                pond_id=pond_id,
                n_labeled=len(samples),
                high_risk_rate=counts["高危"] / total,
                warning_rate=counts["预警"] / total,
                do_median=median(dissolved) if dissolved else None,
                ammonia_median=median(ammonia) if ammonia else None,
                growth_pct=growth_by_pond.get(pond_id),
            )
        )
    return out
