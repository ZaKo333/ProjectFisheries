"""`fisheries.evaluation` 子包的单元测试：钉住**评测代码自身**，而不是评测数字。

这里**不读任何真实数据集**（`aquaponics_pond` 有 11 个 CSV、114 万行，
数字随它变），全部用小 CSV（`tmp_path`）或手工构造的对象。
报告里的那些准确率/ρ 是 `scripts/evaluate_water_rules.py` 的产物，
而本文件要保证的是：**算它们的这段代码没算错**。

覆盖：`ponds`（四种表头方言、量纲归一与量程清洗的先后、聚合、滑窗）、
`labels`（最差指标决定论、生长自洽性）、`metrics`（混淆矩阵/宏平均/κ/秩相关）、
`runner`（黑盒调用引擎的记账）、`do_trend`（外推 vs 持久性基线）、`report`（渲染不炸）。
"""

from __future__ import annotations

import csv
import dataclasses
import io
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fisheries import evaluation as ev
from fisheries.evaluation import do_trend, labels, metrics, ponds, report, runner
from fisheries.tools.water_risk import WaterQualityPoint

UTC = timezone.utc

# ---------------------------------------------------------------------------
# 夹具：数据集里的四种表头方言 + 行构造器
# ---------------------------------------------------------------------------

#: 方言 A（IoTPond1/2）：列名里有空格。
HEADER_A = (
    "created_at,entry_id,Temperature (C),Turbidity (NTU),"
    "Dissolved Oxygen(g/ml),PH,Ammonia(g/ml),Nitrate(g/ml),"
    "Population,Fish_Length (cm),Fish_Weight (g)"
)
#: 方言 B（IoTPond3/4/6/9）：同上，但括号前没有空格。
HEADER_B = (
    "created_at,entry_id,Temperature(C),Turbidity(NTU),"
    "Dissolved Oxygen(g/ml),PH,Ammonia(g/ml),Nitrate(g/ml),"
    "Population,Fish_Length(cm),Fish_Weight(g)"
)
#: 方言 C（IoTPond10/11/12）：全大写 + 源数据自己的拼写错误 `DISOLVED`。
#: 注意这份表头**没有时间戳列**——`resolve_columns` 的兜底规则会碰它。
HEADER_C = "TEMPERATURE,TURBIDITY,DISOLVED OXYGEN,pH,AMMONIA,NITRATE,Population,Length,Weight"
#: 真实文件里的全大写表头是带时间戳列的（另见 IoTPond11 把 Length 拼成 Lenght）。
HEADER_C_TS = "created_at,entry_id," + HEADER_C
HEADER_C_TS_TYPO = HEADER_C_TS.replace(",Length,", ",Lenght,")
#: 方言 D（IoTPond8）：时间戳列的表头是**空的**。
HEADER_D = (
    ",entry_id,Temperature (C),Turbidity (NTU),"
    "Dissolved Oxygen(g/ml),PH,Ammonia(g/ml),Nitrate(g/ml),"
    "Population,Fish_Length (cm),Fish_Weight (g)"
)

#: A / B / D 三种方言的列顺序（跟 HEADER_A 一致）。
_ROW_ORDER = (
    "created_at", "entry_id", "water_temp", "turbidity", "dissolved_oxygen",
    "ph", "ammonia_n", "nitrate", "population", "fish_length", "fish_weight",
)
_ROW_DEFAULTS = {
    "created_at": "2021-06-19 00:00:05 UTC",
    "entry_id": "1",
    "water_temp": "25.0",
    "turbidity": "10.0",
    "dissolved_oxygen": "6.0",
    "ph": "7.5",
    "ammonia_n": "100.0",  # µg/L → 0.1 mg/L
    "nitrate": "100.0",
    "population": "500",
    "fish_length": "20.0",
    "fish_weight": "300.0",
}


def row(**overrides: object) -> str:
    """按 `_ROW_ORDER` 拼一行 CSV，未指定的列走默认值（都是合法读数）。"""
    values = dict(_ROW_DEFAULTS)
    values.update({key: str(value) for key, value in overrides.items()})
    return ",".join(values[name] for name in _ROW_ORDER)


def write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def header_cells(header: str) -> list[str]:
    return header.split(",")


# ---------------------------------------------------------------------------
# 夹具：对象构造器
# ---------------------------------------------------------------------------


def make_sample(
    pond_id: str = "P1",
    *,
    minute: int = 0,
    step_minutes: int = 10,
    day: int = 19,
    water_temp: float = 25.0,
    ph: float = 7.5,
    dissolved_oxygen: float = 6.0,
    ammonia_n: float = 0.1,
    nitrite_n: float | None = None,
    n_raw: int = 1,
) -> ponds.PondSample:
    start = datetime(2021, 6, 19, tzinfo=UTC) + timedelta(days=day - 19)
    return ponds.PondSample(
        pond_id=pond_id,
        sampled_at=start + timedelta(minutes=step_minutes * minute),
        water_temp=water_temp,
        ph=ph,
        dissolved_oxygen=dissolved_oxygen,
        ammonia_n=ammonia_n,
        nitrite_n=nitrite_n,
        n_raw=n_raw,
    )


def make_series(
    pond_id: str = "P1",
    count: int = 19,
    *,
    step_minutes: int = 10,
    dissolved_oxygen: float = 6.0,
    ammonia_n: float = 0.1,
) -> list[ponds.PondSample]:
    return [
        make_sample(
            pond_id,
            minute=i,
            step_minutes=step_minutes,
            dissolved_oxygen=dissolved_oxygen,
            ammonia_n=ammonia_n,
        )
        for i in range(count)
    ]


def make_do_series(
    pond_id: str = "P1",
    count: int = 40,
    *,
    do0: float = 1.0,
    per_point: float = 0.1,
    step_minutes: int = 10,
) -> list[ponds.PondSample]:
    """线性溶氧序列——用来让外推的斜率与真值完全对上。"""
    return [
        make_sample(
            pond_id,
            minute=i,
            step_minutes=step_minutes,
            dissolved_oxygen=do0 + per_point * i,
        )
        for i in range(count)
    ]


def make_window(
    *,
    pond_id: str = "P1",
    true_level: str = "正常",
    limiting: tuple[str, ...] = (),
    n_points: int = 3,
    day: int = 19,
    dissolved_oxygen: float = 6.0,
) -> ponds.PondWindow:
    start = datetime(2021, 6, 19, tzinfo=UTC) + timedelta(days=day - 19)
    readings = tuple(
        {
            "timestamp": (start + timedelta(minutes=10 * i)).isoformat(),
            "water_temp": 25.0,
            "ph": 7.5,
            "dissolved_oxygen": dissolved_oxygen,
            "ammonia_n": 0.1,
        }
        for i in range(n_points)
    )
    return ponds.PondWindow(
        pond_id=pond_id,
        readings=readings,
        started_at=start,
        ended_at=start + timedelta(minutes=10 * (n_points - 1)),
        true_level=true_level,
        limiting=limiting,
    )


def make_eval_sample(
    *,
    true_level: str = "正常",
    risk_level: str = "正常",
    risk_score: float = 10.0,
    pond_id: str = "P1",
    day: int = 19,
    status: str = "ok",
    true_limiting: tuple[str, ...] = (),
    limiting_factor: str = "",
    redline_hits: tuple[str, ...] = (),
) -> runner.EvalSample:
    return runner.EvalSample(
        pond_id=pond_id,
        sampled_on=date(2021, 6, day),
        true_level=true_level,
        true_limiting=true_limiting,
        status=status,
        risk_level=risk_level,
        risk_score=risk_score,
        n_points=3,
        limiting_factor=limiting_factor,
        redline_hits=redline_hits,
    )


def make_run(samples: list[runner.EvalSample] | None = None) -> runner.EvalRun:
    return runner.evaluate(
        samples
        if samples is not None
        else [
            make_eval_sample(true_level="正常", risk_level="正常", risk_score=5.0),
            make_eval_sample(true_level="预警", risk_level="预警", risk_score=45.0, day=20),
            make_eval_sample(true_level="高危", risk_level="预警", risk_score=70.0, day=21),
        ]
    )


# ===========================================================================
# 1. 模块常量与「旧 API 已删干净」
# ===========================================================================


class TestModuleSurface:
    def test_unit_divisors_only_touch_concentrations(self) -> None:
        assert ponds.UNIT_DIVISORS == {"ammonia_n": 1000.0, "nitrate": 1000.0}
        # 溶氧是 mg/L，÷1000 会变成鱼当场死掉的量级。
        assert "dissolved_oxygen" not in ponds.UNIT_DIVISORS
        assert set(ponds.UNIT_DIVISORS) <= set(ponds.RANGES)

    def test_key_constants(self) -> None:
        assert ponds.DATASET_NAME == "aquaponics_pond"
        assert ponds.AGGREGATE_MINUTES == 10
        assert ponds.WINDOW_HOURS == 3.0
        assert ponds.STRIDE_HOURS == ponds.WINDOW_HOURS  # 不重叠
        assert ponds.REQUIRED_FIELDS == ("water_temp", "ph", "dissolved_oxygen", "ammonia_n")
        assert ponds.RANGES["water_temp"] == (0.0, 45.0)
        assert ponds.RANGES["ammonia_n"] == (0.0, 50.0)
        assert labels.RISK_LEVELS == ("正常", "预警", "高危")
        assert labels.RISK_ORDINAL["高危"] == 2

    def test_every_band_field_is_a_known_alias(self) -> None:
        known = {field for aliases in ponds.COLUMN_ALIASES.values() for field in aliases}
        assert known  # 别名表非空
        assert {band.field for band in labels.AQUA_BANDS} == set(labels.BANDS_BY_FIELD)
        assert set(labels.BANDS_BY_FIELD) == {
            "dissolved_oxygen", "ammonia_n", "nitrite_n", "ph", "water_temp",
        }

    def test_package_reexports_all_resolve(self) -> None:
        missing = [name for name in ev.__all__ if not hasattr(ev, name)]
        assert missing == []

    def test_deleted_api_is_really_gone(self) -> None:
        for name in (
            "POLLUTION_LEVELS", "MAPPINGS", "DEFAULT_MAPPING", "map_pollution_level",
            "POLLUTION_ORDINAL", "load_rows", "group_by_station", "iter_trend_windows",
            "iter_instant_windows", "summarize_rows", "RiverSeries", "load_river_series",
            "DEFAULT_TRAIN_CSV", "DEFAULT_INTERVAL_HOURS",
        ):
            assert not hasattr(ev, name), name
        package_dir = Path(ev.__file__).parent
        assert not (package_dir / "dataset.py").exists()
        assert not (package_dir / "label_signal.py").exists()

    def test_renamed_attributes_are_gone(self) -> None:
        # `window.station` 改成了 `window.pond_id`；dataclass 的实例字段不出现在类上，
        # 所以只能查 fields() 而不是 hasattr()。
        window_fields = {field.name for field in dataclasses.fields(ponds.PondWindow)}
        assert "pond_id" in window_fields
        assert "station" not in window_fields
        assert "true_risk" not in {f.name for f in dataclasses.fields(runner.EvalSample)}
        assert "mapping" not in {f.name for f in dataclasses.fields(runner.EvalRun)}


# ===========================================================================
# 2. parse_timestamp / resolve_columns
# ===========================================================================


class TestParseTimestamp:
    def test_tz_name_utc(self) -> None:
        assert ponds.parse_timestamp("2021-06-19 00:00:05 UTC") == datetime(
            2021, 6, 19, 0, 0, 5, tzinfo=UTC
        )

    def test_tz_name_cet(self) -> None:
        # CET 不是固定偏移，代码按 UTC+1 处理（数据集里没有 CEST 观测）。
        assert ponds.parse_timestamp("2021-06-19 00:00:05 CET") == datetime(
            2021, 6, 18, 23, 0, 5, tzinfo=UTC
        )

    def test_iso_offset(self) -> None:
        assert ponds.parse_timestamp("2021-06-18T13:06:49+01:00") == datetime(
            2021, 6, 18, 12, 6, 49, tzinfo=UTC
        )

    def test_zulu_suffix(self) -> None:
        assert ponds.parse_timestamp("2021-06-19T00:00:05Z") == datetime(
            2021, 6, 19, 0, 0, 5, tzinfo=UTC
        )

    def test_naive_iso_is_read_as_utc(self) -> None:
        parsed = ponds.parse_timestamp("2021-06-19 00:00:05")
        assert parsed == datetime(2021, 6, 19, 0, 0, 5, tzinfo=UTC)
        assert parsed is not None and parsed.tzinfo is UTC

    def test_microseconds_and_day_first_formats(self) -> None:
        assert ponds.parse_timestamp("2021-06-19 00:00:05.250000") == datetime(
            2021, 6, 19, 0, 0, 5, 250000, tzinfo=UTC
        )
        assert ponds.parse_timestamp("13-06-2021 00:00:05") == datetime(
            2021, 6, 13, 0, 0, 5, tzinfo=UTC
        )

    def test_lowercase_and_padded_tz_name(self) -> None:
        assert ponds.parse_timestamp("  2021-06-19 00:00:05 cest ") == datetime(
            2021, 6, 18, 22, 0, 5, tzinfo=UTC
        )

    def test_unparseable_returns_none(self) -> None:
        for raw in ("", "   ", "not-a-timestamp", "2021/06/19 00:00:05",
                    "2021-06-19 00:00:05 XYZ", "32-06-2021 00:00:05"):
            assert ponds.parse_timestamp(raw) is None, raw

    def test_none_returns_none(self) -> None:
        assert ponds.parse_timestamp(None) is None


class TestResolveColumns:
    def test_dialect_a_with_spaces(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_A))
        assert cols["created_at"] == (0, "created_at")
        assert cols["water_temp"] == (2, "Temperature (C)")
        assert cols["turbidity"] == (3, "Turbidity (NTU)")
        assert cols["dissolved_oxygen"] == (4, "Dissolved Oxygen(g/ml)")
        assert cols["ph"] == (5, "PH")
        assert cols["ammonia_n"] == (6, "Ammonia(g/ml)")
        assert cols["nitrate"] == (7, "Nitrate(g/ml)")
        assert cols["population"] == (8, "Population")
        assert cols["fish_length"] == (9, "Fish_Length (cm)")
        assert cols["fish_weight"] == (10, "Fish_Weight (g)")
        assert "nitrite_n" not in cols  # 这份数据根本没有亚硝酸盐列

    def test_dialect_b_without_spaces(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_B))
        assert cols["water_temp"] == (2, "Temperature(C)")
        assert cols["fish_length"] == (9, "Fish_Length(cm)")
        assert cols["fish_weight"] == (10, "Fish_Weight(g)")
        assert cols["dissolved_oxygen"][0] == 4

    def test_dialect_c_all_caps_and_typos(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_C))
        assert cols["water_temp"] == (0, "TEMPERATURE")
        assert cols["turbidity"] == (1, "TURBIDITY")
        assert cols["dissolved_oxygen"] == (2, "DISOLVED OXYGEN")  # 源数据拼错 Dissolved
        assert cols["ph"] == (3, "pH")
        assert cols["ammonia_n"] == (4, "AMMONIA")
        assert cols["nitrate"] == (5, "NITRATE")
        assert cols["population"] == (6, "Population")
        assert cols["fish_length"] == (7, "Length")
        assert cols["fish_weight"] == (8, "Weight")
        # 这份表头没有时间戳列名 → 按文档退回第 0 列（这里恰好是水温列，
        # 所以真实文件一定带 created_at，见 HEADER_C_TS）。
        assert cols["created_at"][0] == 0

    def test_length_typo_alias(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_C_TS_TYPO))
        assert cols["fish_length"] == (9, "Lenght")

    def test_dialect_c_with_timestamp(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_C_TS))
        assert cols["created_at"] == (0, "created_at")
        assert cols["water_temp"] == (2, "TEMPERATURE")

    def test_dialect_d_blank_timestamp_header(self) -> None:
        cols = ponds.resolve_columns(header_cells(HEADER_D))
        assert cols["created_at"] == (0, "")
        assert "entry_id" not in cols  # 别名表里没有 entry_id
        assert cols["water_temp"] == (2, "Temperature (C)")

    def test_dialect_d_blank_header_cannot_be_read_by_dictreader(self) -> None:
        # 这是「一律按列下标取值」的理由：DictReader 遇到多个空列名会互相覆盖。
        cells = header_cells(HEADER_D) + ["", ""]
        assert cells.count("") >= 3
        cols = ponds.resolve_columns(cells)
        assert cols["created_at"][0] == 0
        reader = csv.DictReader(io.StringIO(",".join(cells)))
        # 空列名在字典视图里只剩一个 '' 键，位置信息全丢了。
        assert list(reader.fieldnames or []).count("") == 3

    def test_empty_header_yields_nothing(self) -> None:
        assert ponds.resolve_columns([]) == {}

    def test_first_alias_match_wins(self) -> None:
        cols = ponds.resolve_columns(["created_at", "Temperature", "Temperature(C)"])
        assert cols["water_temp"] == (1, "Temperature")  # 先出现的先占


# ===========================================================================
# 3. load_pond：清洗、量纲归一、聚合
# ===========================================================================


class TestLoadPond:
    def test_clean_rows_survive_with_label_buckets_and_counters(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "IoTPondX.csv",
            HEADER_A,
            [
                # 同一桶（00:00–00:09）两行 → 1 个聚合点。
                # 氨氮原始值 1200 µg/L ÷1000 = 1.2 mg/L：越过 1.0 红线 → 高危。
                row(created_at="2021-06-19 00:00:05 UTC", entry_id="1", ammonia_n="1200.0"),
                row(created_at="2021-06-19 00:01:05 UTC", entry_id="2", ammonia_n="1200.0",
                    fish_weight="310.0"),
                # 水温哨兵 -127：越界丢弃（行也因必填缺失被丢）。
                row(created_at="2021-06-19 00:02:05 UTC", entry_id="3", water_temp="-127"),
                # 必填项为空：丢。
                row(created_at="2021-06-19 00:03:05 UTC", entry_id="4", ph=""),
                # 01:00 桶：溶氧 4.0（适宜线 5.0 以下）→ 预警。
                row(created_at="2021-06-19 01:00:05 UTC", entry_id="5", water_temp="26.0",
                    dissolved_oxygen="4.0", ph="7.6"),
                # 02:00 桶：全部适宜 → 正常。
                row(created_at="2021-06-19 02:00:05 UTC", entry_id="6", water_temp="24.0",
                    dissolved_oxygen="7.0", ph="7.4", ammonia_n="50.0"),
                row(created_at="not-a-timestamp", entry_id="7"),
                row(created_at="2021-06-19 02:01:05 UTC", entry_id="8", ph="-3.137"),
                row(created_at="2021-06-19 02:02:05 UTC", entry_id="9", dissolved_oxygen="44.98"),
                # 60000 µg/L ÷1000 = 60 mg/L > 50 → 越界。
                row(created_at="2021-06-19 02:03:05 UTC", entry_id="10", ammonia_n="60000.0"),
            ],
        )

        samples, stats = ponds.load_pond(path)

        assert stats["pond_id"] == "IoTPondX"
        assert stats["n_lines"] == 10
        assert stats["n_bad_timestamp"] == 1
        assert stats["n_dropped_missing_required"] == 5
        assert stats["n_out_of_range"] == 4
        assert stats["out_of_range_by_field"] == {
            "water_temp": 1, "ph": 1, "dissolved_oxygen": 1, "ammonia_n": 1,
        }
        assert stats["n_samples"] == 3
        assert stats["growth_pct"] is None  # 只有 9 个有效体重，不足 20 个

        levels = [labels.label_indicators(s.indicators()) for s in samples]
        assert levels == ["高危", "预警", "正常"]
        # 三个标签桶都存在。
        assert labels.label_distribution(levels) == {"正常": 1, "预警": 1, "高危": 1}

        first = samples[0]
        assert first.sampled_at == datetime(2021, 6, 19, 0, 0, tzinfo=UTC)
        assert first.n_raw == 2
        assert first.ammonia_n == pytest.approx(1.2)
        assert first.nitrite_n is None
        assert first.turbidity is None  # 浊度列根本没人读
        assert samples[1].dissolved_oxygen == pytest.approx(4.0)
        assert samples[2].water_temp == pytest.approx(24.0)

    def test_range_check_runs_after_unit_division(self, tmp_path: Path) -> None:
        """量程表写的是归一化之后的量，先查量程会静默丢掉合法读数。"""
        path = write_csv(
            tmp_path / "order.csv",
            HEADER_A,
            [row(created_at="2021-06-19 00:00:05 UTC", ammonia_n="1200.0")],
        )
        samples, stats = ponds.load_pond(path)

        assert stats["n_out_of_range"] == 0  # 原始值 1200 不该触发量程
        assert stats["n_dropped_missing_required"] == 0
        assert stats["n_samples"] == 1
        # 1.2 mg/L：超出适宜带（0.2）但远在量程内（50），必须是「预警/高危」而非「丢弃」。
        assert samples[0].ammonia_n == pytest.approx(1.2)
        assert labels.label_indicators(samples[0].indicators()) == "高危"
        assert labels.worst_indicators(samples[0].indicators()) == ("ammonia_n",)

    def test_ammonia_above_50_after_division_is_dropped(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "toobig.csv",
            HEADER_A,
            [row(created_at="2021-06-19 00:00:05 UTC", ammonia_n="50001.0")],
        )
        samples, stats = ponds.load_pond(path)
        assert samples == []
        assert stats["out_of_range_by_field"] == {"ammonia_n": 1}

    def test_sentinel_temperature_is_dropped(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "sentinel.csv",
            HEADER_A,
            [row(created_at="2021-06-19 00:00:05 UTC", water_temp="-127")],
        )
        samples, stats = ponds.load_pond(path)
        assert samples == []
        assert stats["n_out_of_range"] == 1
        assert stats["out_of_range_by_field"] == {"water_temp": 1}
        assert stats["n_dropped_missing_required"] == 1

    def test_inf_and_nan_are_missing_not_numbers(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "nonfinite.csv",
            HEADER_A,
            [
                row(created_at="2021-06-19 00:00:05 UTC", entry_id="1", ammonia_n="inf"),
                row(created_at="2021-06-19 00:01:05 UTC", entry_id="2", ammonia_n="nan"),
                row(created_at="2021-06-19 00:02:05 UTC", entry_id="3", ph="abc"),
            ],
        )
        samples, stats = ponds.load_pond(path)
        assert samples == []
        # inf/nan/非数字都算缺测，**不算**越界——否则会伪装成水质事件。
        assert stats["n_out_of_range"] == 0
        assert stats["n_dropped_missing_required"] == 3

    def test_four_label_buckets_and_n_raw(self, tmp_path: Path) -> None:
        # A 桶聚合 2 行，B 桶 1 行。
        path = write_csv(
            tmp_path / "buckets.csv",
            HEADER_A,
            [
                row(created_at="2021-06-19 00:00:05 UTC", entry_id="1"),
                row(created_at="2021-06-19 00:09:59 UTC", entry_id="2"),
                row(created_at="2021-06-19 00:10:00 UTC", entry_id="3"),
            ],
        )
        samples, stats = ponds.load_pond(path)
        assert stats["n_samples"] == 2
        assert [s.n_raw for s in samples] == [2, 1]
        assert samples[0].sampled_at == datetime(2021, 6, 19, 0, 0, tzinfo=UTC)
        assert samples[1].sampled_at == datetime(2021, 6, 19, 0, 10, tzinfo=UTC)

    def test_aggregate_minutes_is_configurable(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "hourly.csv",
            HEADER_A,
            [row(created_at=f"2021-06-19 00:{minute:02d}:05 UTC", entry_id=str(i))
             for i, minute in enumerate((0, 5, 30, 59))],
        )
        samples, _ = ponds.load_pond(path, aggregate_minutes=60)
        assert len(samples) == 1
        assert samples[0].n_raw == 4

    def test_growth_pct_from_median_of_first_and_last_tenth(self, tmp_path: Path) -> None:
        # 30 行同一桶：体重 100,110,…,390 → 首 10% 中位数 110，末 10% 中位数 380。
        rows = [
            row(created_at=f"2021-06-19 00:00:{i:02d} UTC", entry_id=str(i),
                fish_weight=str(100 + 10 * i))
            for i in range(30)
        ]
        path = write_csv(tmp_path / "growth.csv", HEADER_A, rows)
        samples, stats = ponds.load_pond(path)
        assert len(samples) == 1
        assert samples[0].n_raw == 30
        assert stats["growth_pct"] == pytest.approx((380 - 110) / 110 * 100, rel=1e-9)

    def test_dialect_b_loads(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "B.csv",
            HEADER_B,
            [row(created_at="2021-06-19 00:00:05 UTC", ammonia_n="1500.0")],
        )
        samples, stats = ponds.load_pond(path)
        assert stats["columns"]["water_temp"] == "Temperature(C)"
        assert samples[0].ammonia_n == pytest.approx(1.5)

    def test_dialect_c_loads(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "C.csv",
            HEADER_C_TS,
            [row(created_at="2021-06-19 00:00:05 UTC", ammonia_n="1500.0")],
        )
        samples, stats = ponds.load_pond(path)
        assert stats["columns"]["dissolved_oxygen"] == "DISOLVED OXYGEN"
        assert stats["columns"]["fish_length"] == "Length"
        assert samples[0].ammonia_n == pytest.approx(1.5)

    def test_dialect_d_blank_timestamp_header_loads(self, tmp_path: Path) -> None:
        path = write_csv(
            tmp_path / "IoTPond8.csv",
            HEADER_D,
            [row(created_at="2021-06-18T13:06:49+01:00", ammonia_n="1200.0")],
        )
        samples, stats = ponds.load_pond(path)
        assert stats["columns"]["created_at"] == ""
        assert stats["n_bad_timestamp"] == 0
        # 12:06:49 UTC → 10 分钟桶 → 12:00。
        assert samples[0].sampled_at == datetime(2021, 6, 18, 12, 0, tzinfo=UTC)

    def test_short_row_is_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "short.csv"
        path.write_text(HEADER_A + "\n" + "2021-06-19 00:00:05 UTC,1,25.0\n", encoding="utf-8")
        samples, stats = ponds.load_pond(path)
        assert samples == []
        assert stats["n_lines"] == 1
        assert stats["n_bad_timestamp"] == 0
        assert stats["n_dropped_missing_required"] == 1

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        samples, stats = ponds.load_pond(path)
        assert samples == []
        assert stats["n_lines"] == 0
        assert stats["n_samples"] == 0

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ponds.load_pond(tmp_path / "nope.csv")

    def test_load_ponds_totals_and_ordering(self, tmp_path: Path) -> None:
        paths = [
            write_csv(tmp_path / "IoTPond10.csv", HEADER_A,
                      [row(created_at="2021-06-19 00:00:05 UTC")]),
            write_csv(tmp_path / "IoTPond2.csv", HEADER_A,
                      [row(created_at="2021-06-19 00:00:05 UTC", water_temp="-127")]),
        ]
        by_pond, totals = ponds.load_ponds(paths)
        assert set(by_pond) == {"IoTPond10", "IoTPond2"}
        assert len(by_pond["IoTPond10"]) == 1
        assert by_pond["IoTPond2"] == []
        assert totals["n_files"] == 2
        assert totals["n_lines"] == 2
        assert totals["n_samples"] == 1
        assert totals["n_out_of_range"] == 1
        assert len(totals["per_file"]) == 2
        assert [s["pond_id"] for s in totals["per_file"]] == ["IoTPond10", "IoTPond2"]

    def test_load_ponds_limit_files(self, tmp_path: Path) -> None:
        paths = [
            write_csv(tmp_path / f"P{i}.csv", HEADER_A,
                      [row(created_at="2021-06-19 00:00:05 UTC")])
            for i in range(3)
        ]
        by_pond, totals = ponds.load_ponds(paths, limit_files=2)
        assert len(by_pond) == 2
        assert totals["n_files"] == 2


# ===========================================================================
# 4. iter_windows / summarize_samples / pond_outcomes
# ===========================================================================


class TestIterWindows:
    def test_window_length_stride_and_non_overlap(self) -> None:
        by_pond = {"P1": make_series("P1", 19, step_minutes=10)}  # 00:00 … 03:00
        windows = list(ponds.iter_windows(by_pond, window_hours=1.0, stride_hours=1.0))

        assert len(windows) == 3
        assert [w.started_at for w in windows] == [
            datetime(2021, 6, 19, 0, 0, tzinfo=UTC),
            datetime(2021, 6, 19, 1, 0, tzinfo=UTC),
            datetime(2021, 6, 19, 2, 0, tzinfo=UTC),
        ]
        assert [w.ended_at for w in windows] == [
            datetime(2021, 6, 19, 0, 50, tzinfo=UTC),
            datetime(2021, 6, 19, 1, 50, tzinfo=UTC),
            datetime(2021, 6, 19, 2, 50, tzinfo=UTC),
        ]
        assert [w.n_points for w in windows] == [6, 6, 6]
        assert all(w.hours == pytest.approx(50 / 60) for w in windows)

        # 不重叠：后一窗的起点严格晚于前一窗的终点，且时间戳无交集。
        for first, second in zip(windows, windows[1:]):
            assert second.started_at > first.ended_at
            first_stamps = {r["timestamp"] for r in first.readings}
            second_stamps = {r["timestamp"] for r in second.readings}
            assert not (first_stamps & second_stamps)

    def test_window_carries_expected_fields(self) -> None:
        samples = make_series("Pond-7", 19, dissolved_oxygen=7.0)
        windows = list(ponds.iter_windows({"Pond-7": samples}, window_hours=1.0,
                                          stride_hours=1.0))
        window = windows[0]
        assert window.pond_id == "Pond-7"
        assert window.true_level == "正常"
        assert window.limiting == ()
        assert window.started_at == samples[0].sampled_at
        assert window.ended_at == samples[5].sampled_at
        assert window.n_points == 6
        assert window.sampled_on == window.ended_at.date()
        assert window.hours == pytest.approx(50 / 60)  # 6 点 × 10 分钟，右端点不在窗内

    def test_true_level_comes_from_the_last_point_in_the_window(self) -> None:
        samples = make_series("P1", 19, dissolved_oxygen=7.0)
        # 1 小时窗 + 1 小时步长 → 三个窗覆盖下标 0-5 / 6-11 / 12-17；
        # 下标 17（02:50）是最后一窗的末点，18（03:00）落在所有窗之外。
        samples[17] = replace(samples[17], ammonia_n=2.0)  # 越过 1.0 红线
        samples[18] = replace(samples[18], ammonia_n=3.0)  # 窗外，不该有任何影响
        windows = list(ponds.iter_windows({"P1": samples}, window_hours=1.0, stride_hours=1.0))
        assert len(windows) == 3
        assert windows[-1].true_level == "高危"
        assert windows[-1].limiting == ("ammonia_n",)
        assert windows[0].true_level == "正常"  # 前面的点只承担趋势证据

    def test_default_window_is_three_hours(self) -> None:
        by_pond = {"P1": make_series("P1", 13, step_minutes=30)}  # 00:00 … 06:00
        windows = list(ponds.iter_windows(by_pond))
        assert len(windows) == 2
        assert [w.started_at.minute for w in windows] == [0, 0]
        assert [w.started_at.hour for w in windows] == [0, 3]
        assert [w.ended_at for w in windows] == [
            datetime(2021, 6, 19, 2, 30, tzinfo=UTC),
            datetime(2021, 6, 19, 5, 30, tzinfo=UTC),
        ]
        # 窗口时长总是**小于** window_hours：贴着右边界那个点被排除在外。
        assert all(w.hours == pytest.approx(2.5) for w in windows)
        assert all(w.hours < ponds.WINDOW_HOURS for w in windows)

    def test_chunk_needs_three_points(self) -> None:
        samples = [
            make_sample("P1", minute=0),
            make_sample("P1", minute=30),
            make_sample("P1", minute=180),
        ]
        assert list(ponds.iter_windows({"P1": samples}, window_hours=1.0,
                                       stride_hours=1.0)) == []

    def test_ponds_with_too_few_samples_are_skipped(self) -> None:
        by_pond = {"P1": make_series("P1", 2), "P2": make_series("P2", 19)}
        windows = list(ponds.iter_windows(by_pond, window_hours=1.0, stride_hours=1.0))
        assert {w.pond_id for w in windows} == {"P2"}

    def test_ponds_are_iterated_in_sorted_order(self) -> None:
        by_pond = {
            "P10": make_series("P10", 19),
            "P2": make_series("P2", 19),
        }
        ponds_seen = [w.pond_id for w in ponds.iter_windows(by_pond, window_hours=1.0,
                                                            stride_hours=1.0)]
        assert ponds_seen == ["P10"] * 3 + ["P2"] * 3

    def test_empty_input_yields_nothing(self) -> None:
        assert list(ponds.iter_windows({})) == []

    def test_non_positive_durations_raise(self) -> None:
        by_pond = {"P1": make_series("P1", 19)}
        with pytest.raises(ValueError):
            list(ponds.iter_windows(by_pond, window_hours=0.0))
        with pytest.raises(ValueError):
            list(ponds.iter_windows(by_pond, stride_hours=-1.0))


class TestEngineReadings:
    def test_to_point_has_only_engine_keys(self) -> None:
        point = make_sample("P1", nitrite_n=0.05).to_point()
        assert set(point) == {
            "timestamp", "water_temp", "ph", "dissolved_oxygen", "ammonia_n", "nitrite_n",
        }
        # WaterQualityPoint 是 extra="forbid"：多一个 nitrate 整个请求都会被拒。
        assert "nitrate" not in point
        parsed = WaterQualityPoint(**point)
        assert parsed.nitrite_n == pytest.approx(0.05)

    def test_to_point_drops_missing_nitrite(self) -> None:
        point = make_sample("P1", nitrite_n=None).to_point()
        assert set(point) == {"timestamp", "water_temp", "ph", "dissolved_oxygen", "ammonia_n"}
        WaterQualityPoint(**point)  # 不抛异常

    def test_window_readings_are_accepted_by_the_engine_model(self) -> None:
        windows = list(ponds.iter_windows({"P1": make_series("P1", 19)},
                                          window_hours=1.0, stride_hours=1.0))
        for reading in windows[0].readings:
            assert "nitrate" not in reading
            point = WaterQualityPoint(**reading)
            assert point.dissolved_oxygen == pytest.approx(6.0)

    def test_extra_key_is_rejected_by_the_engine_model(self) -> None:
        point = make_sample("P1").to_point()
        with pytest.raises(ValidationError):
            WaterQualityPoint(**point, nitrate=1.0)
        # 顺带确认溶氧的量级没被 ÷1000（÷1000 后是 0.006，鱼会当场死）。
        assert point["dissolved_oxygen"] > 1.0


class TestSummarizeSamples:
    def test_summary_counts_and_ranges(self) -> None:
        by_pond = {
            "P1": [
                make_sample("P1", minute=0, dissolved_oxygen=7.0),
                make_sample("P1", minute=1, dissolved_oxygen=4.0),
                make_sample("P1", minute=2, ammonia_n=2.0),
            ],
            "P2": [make_sample("P2", minute=0, dissolved_oxygen=7.0)],
        }
        summary = ponds.summarize_samples(by_pond)

        assert summary["n_samples"] == 4
        assert summary["n_ponds"] == 2
        assert summary["label_counts"] == {"正常": 2, "预警": 1, "高危": 1}
        assert summary["per_pond"] == {"P1": 3, "P2": 1}
        assert summary["samples_per_pond_min"] == 1
        assert summary["samples_per_pond_max"] == 3
        assert summary["time_from"].startswith("2021-06-19T00:00:00")
        assert summary["time_to"].startswith("2021-06-19T00:20:00")
        detail = summary["per_pond_detail"]["P1"]
        assert detail["n_samples"] == 3
        assert detail["高危"] == 1
        assert detail["high_risk_rate"] == pytest.approx(1 / 3)

    def test_empty_summary(self) -> None:
        summary = ponds.summarize_samples({})
        assert summary["n_samples"] == 0
        assert summary["n_ponds"] == 0
        assert summary["label_counts"] == {"正常": 0, "预警": 0, "高危": 0}
        assert summary["time_from"] == ""
        assert summary["samples_per_pond_min"] == 0
        assert summary["per_pond_detail"] == {}


class TestPondOutcomes:
    def test_outcome_medians_and_rates(self) -> None:
        by_pond = {
            "P1": [
                make_sample("P1", minute=0, dissolved_oxygen=6.0, ammonia_n=0.1),
                make_sample("P1", minute=1, dissolved_oxygen=4.0, ammonia_n=0.3),
                make_sample("P1", minute=2, dissolved_oxygen=2.0, ammonia_n=0.5),
                make_sample("P1", minute=3, dissolved_oxygen=2.0, ammonia_n=2.0),
            ],
        }
        outcomes = ponds.pond_outcomes(by_pond, {"P1": 123.5})
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.pond_id == "P1"
        assert outcome.n_labeled == 4
        assert outcome.high_risk_rate == pytest.approx(0.25)
        assert outcome.warning_rate == pytest.approx(0.5)
        assert outcome.do_median == pytest.approx(3.0)  # 6 / 4 / 2 / 2 → (2 + 4) / 2
        assert outcome.ammonia_median == pytest.approx(0.4)
        assert outcome.growth_pct == pytest.approx(123.5)

    def test_empty_pond_is_skipped_and_missing_growth_is_none(self) -> None:
        by_pond = {"P1": [], "P2": make_series("P2", 1, dissolved_oxygen=7.0)}
        outcomes = ponds.pond_outcomes(by_pond, {})
        assert [o.pond_id for o in outcomes] == ["P2"]
        assert outcomes[0].growth_pct is None


class TestGrowthCoherence:
    def test_needs_at_least_two_usable_ponds(self) -> None:
        outcomes = [
            labels.PondOutcome("P1", 10, 0.1, 0.1, growth_pct=None),
            labels.PondOutcome("P2", 10, 0.2, 0.1, growth_pct=50.0),
        ]
        coherence = labels.growth_coherence(outcomes)
        assert coherence.n_pairs == 1
        assert coherence.n == 1
        assert coherence.rho == 0.0
        assert "样本太少" in coherence.verdict()

    def test_negative_rho_is_consistent_with_the_labels(self) -> None:
        outcomes = [
            labels.PondOutcome(f"P{i}", 10, rate, 0.0, growth_pct=growth)
            for i, (rate, growth) in enumerate(
                [(0.0, 300.0), (0.1, 200.0), (0.2, 100.0), (0.3, 0.0)]
            )
        ]
        coherence = labels.growth_coherence(outcomes)
        assert coherence.n_pairs == 4
        assert coherence.rho == pytest.approx(-1.0)
        assert "方向一致" in coherence.verdict()
        assert "方向相反" not in coherence.verdict()

    def test_weak_negative_rho_is_flagged_as_weak(self) -> None:
        outcomes = [
            labels.PondOutcome(f"P{i}", 10, rate, 0.0, growth_pct=growth)
            for i, (rate, growth) in enumerate(
                [(0.0, 20.0), (0.1, 30.0), (0.2, 40.0), (0.3, 10.0)]
            )
        ]
        coherence = labels.growth_coherence(outcomes)
        assert coherence.rho == pytest.approx(-0.2)
        assert "方向一致但很弱" in coherence.verdict()

    def test_positive_rho_is_reported_as_a_contradiction(self) -> None:
        outcomes = [
            labels.PondOutcome(f"P{i}", 10, rate, 0.0, growth_pct=growth)
            for i, (rate, growth) in enumerate(
                [(0.0, 0.0), (0.1, 100.0), (0.2, 200.0), (0.3, 300.0)]
            )
        ]
        coherence = labels.growth_coherence(outcomes)
        assert coherence.rho == pytest.approx(1.0)
        assert "方向相反" in coherence.verdict()
        assert "长" in coherence.verdict()

    def test_ponds_without_growth_are_excluded(self) -> None:
        outcomes = [
            labels.PondOutcome("P1", 10, 0.0, 0.1, growth_pct=None),
            labels.PondOutcome("P2", 10, 0.1, 0.1, growth_pct=None),
            labels.PondOutcome("P3", 0, 0.2, 0.1, growth_pct=10.0),
        ]
        coherence = labels.growth_coherence(outcomes)
        assert coherence.n_pairs == 0
        assert coherence.ponds == ()


# ===========================================================================
# 5. labels：最差指标决定论
# ===========================================================================


class TestIndicatorBands:
    @pytest.mark.parametrize(
        ("field", "value", "severity"),
        [
            ("dissolved_oxygen", 6.0, 0),
            ("dissolved_oxygen", 5.0, 0),   # 适宜带含端点
            ("dissolved_oxygen", 4.9, 1),
            ("dissolved_oxygen", 3.0, 1),
            ("dissolved_oxygen", 1.5, 1),   # 红线是「< 1.5」
            ("dissolved_oxygen", 1.49, 2),
            ("ammonia_n", 0.2, 0),
            ("ammonia_n", 0.21, 1),
            ("ammonia_n", 0.5, 1),
            ("ammonia_n", 1.0, 1),          # 红线是「> 1.0」
            ("ammonia_n", 1.01, 2),
            ("nitrite_n", 0.1, 0),
            ("nitrite_n", 0.31, 1),
            ("nitrite_n", 0.51, 2),
            ("ph", 6.5, 0),
            ("ph", 8.5, 0),
            ("ph", 6.4, 1),
            ("ph", 5.5, 1),
            ("ph", 5.49, 2),
            ("ph", 9.5, 1),
            ("ph", 9.51, 2),
            ("water_temp", 20.0, 0),
            ("water_temp", 30.0, 0),
            ("water_temp", 19.9, 1),
            ("water_temp", 33.0, 1),
            ("water_temp", 33.1, 1),
        ],
    )
    def test_severity_boundaries(self, field: str, value: float, severity: int) -> None:
        assert labels.BANDS_BY_FIELD[field].severity(value) == severity

    def test_water_temp_has_no_redline(self) -> None:
        band = labels.BANDS_BY_FIELD["water_temp"]
        assert band.breaches_redline(1e6) is False
        assert band.severity(1e6) == 1
        assert band.breaches_optimum(1e6) is True

    def test_band_metadata(self) -> None:
        band = labels.BANDS_BY_FIELD["dissolved_oxygen"]
        assert (band.label, band.unit) == ("溶氧", "mg/L")
        assert band.optimum == (5.0, None)
        assert band.watch == (3.0, None)
        assert band.redline == (1.5, None)


class TestIndicatorSeverities:
    def test_missing_values_are_not_severities(self) -> None:
        severities = labels.indicator_severities(
            {"dissolved_oxygen": 6.0, "ph": None, "ammonia_n": 2.0}
        )
        assert severities == {"dissolved_oxygen": 0, "ammonia_n": 2}
        assert labels.indicator_severities({}) == {}

    def test_worst_indicators_picks_the_top_severity(self) -> None:
        assert labels.worst_indicators({"dissolved_oxygen": 6.0, "ph": 7.5}) == ()
        assert labels.worst_indicators({}) == ()
        assert labels.worst_indicators({"dissolved_oxygen": None, "ph": None}) == ()
        assert labels.worst_indicators({"dissolved_oxygen": 4.0, "ph": 7.5}) == (
            "dissolved_oxygen",
        )

    def test_worst_indicators_can_return_several_fields(self) -> None:
        values = {"dissolved_oxygen": 1.0, "ammonia_n": 2.0, "ph": 7.5}
        # 顺序跟 AQUA_BANDS 一致（溶氧在氨氮前面）。
        assert labels.worst_indicators(values) == ("dissolved_oxygen", "ammonia_n")


class TestLabelIndicators:
    def test_all_good_is_normal(self) -> None:
        values = {"water_temp": 25.0, "ph": 7.5, "dissolved_oxygen": 6.0, "ammonia_n": 0.1}
        assert labels.label_indicators(values) == "正常"
        assert labels.worst_indicators(values) == ()

    def test_warning_bucket(self) -> None:
        values = {"water_temp": 25.0, "ph": 7.5, "dissolved_oxygen": 4.0, "ammonia_n": 0.1}
        assert labels.label_indicators(values) == "预警"

    def test_redline_forces_the_top_bucket(self) -> None:
        values = {"water_temp": 25.0, "ph": 7.5, "dissolved_oxygen": 6.0, "ammonia_n": 1.2}
        assert labels.label_indicators(values) == "高危"
        assert labels.worst_indicators(values) == ("ammonia_n",)

    def test_missing_value_yields_warning_bucket(self) -> None:
        # 一个可测指标都没有 → 无法判定 → 预警（而不是当成「正常」蒙混过去）。
        assert labels.label_indicators({}) == "预警"
        assert labels.label_indicators({"dissolved_oxygen": None}) == "预警"
        assert labels.label_indicators({"dissolved_oxygen": None, "ph": None}) == "预警"

    def test_one_measured_optimum_field_is_enough_for_normal(self) -> None:
        # 语义提醒：只要有**一个**指标测出来且落在适宜带内，就判「正常」；
        # 缺测本身不降级（缺测的惩罚只体现在「全都测不出来」时的兜底预警）。
        assert labels.label_indicators({"dissolved_oxygen": None, "ph": 7.5}) == "正常"

    def test_missing_indicator_does_not_block_the_others(self) -> None:
        values = {"dissolved_oxygen": 6.0, "ph": 7.5, "ammonia_n": 0.1}
        assert labels.label_indicators(values) == "正常"

    def test_label_distribution(self) -> None:
        assert labels.label_distribution(["正常", "正常", "高危", "未知"]) == {
            "正常": 2, "预警": 0, "高危": 1,
        }
        assert labels.label_distribution([]) == {"正常": 0, "预警": 0, "高危": 0}


# ===========================================================================
# 6. metrics
# ===========================================================================


class TestConfusionMatrix:
    def test_from_pairs_and_basic_readings(self) -> None:
        matrix = metrics.ConfusionMatrix.from_pairs(
            [("正常", "正常"), ("预警", "正常"), ("高危", "高危")],
            labels.RISK_LEVELS,
        )
        assert matrix.total == 3
        assert matrix.count("高危", "高危") == 1
        assert matrix.count("高危", "预警") == 0
        assert matrix.row_total("正常") == 1
        assert matrix.col_total("高危") == 1
        assert matrix.row("预警") == [1, 0, 0]  # 预警真值被预测成了 正常
        assert matrix.accuracy == pytest.approx(2 / 3)

    def test_add_and_validation(self) -> None:
        matrix = metrics.ConfusionMatrix(labels.RISK_LEVELS)
        matrix.add("正常", "高危", 2)
        assert matrix.total == 2
        with pytest.raises(ValueError):
            matrix.add("正常", "灭世")
        with pytest.raises(ValueError):
            metrics.ConfusionMatrix(labels.RISK_LEVELS, {("正常", "灭世"): 1})
        with pytest.raises(ValueError):
            metrics.ConfusionMatrix.from_pairs([("灭世", "正常")], labels.RISK_LEVELS)

    def test_per_class_precision_recall_f1(self) -> None:
        matrix = metrics.ConfusionMatrix.from_pairs(
            [("正常", "正常"), ("正常", "正常"), ("正常", "高危")],
            labels.RISK_LEVELS,
        )
        reports = {r.label: r for r in matrix.per_class()}
        normal = reports["正常"]
        assert (normal.support, normal.predicted, normal.tp, normal.fp, normal.fn) == (3, 2, 2, 0, 1)
        assert normal.precision == pytest.approx(1.0)
        assert normal.recall == pytest.approx(2 / 3)
        assert normal.f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
        red = reports["高危"]
        assert (red.support, red.predicted, red.tp, red.fp, red.fn) == (0, 1, 0, 1, 0)
        assert red.precision == 0.0 and red.recall == 0.0 and red.f1 == 0.0

    def test_macro_metrics_for_a_perfect_matrix(self) -> None:
        matrix = metrics.ConfusionMatrix.from_pairs(
            [(level, level) for level in labels.RISK_LEVELS], labels.RISK_LEVELS
        )
        macro = matrix.macro()
        assert macro.n == 3
        assert macro.accuracy == pytest.approx(1.0)
        assert macro.macro_precision == pytest.approx(1.0)
        assert macro.macro_recall == pytest.approx(1.0)
        assert macro.macro_f1 == pytest.approx(1.0)
        assert macro.weighted_f1 == pytest.approx(1.0)
        assert macro.kappa == pytest.approx(1.0)

    def test_kappa_is_zero_when_every_prediction_is_the_same_class(self) -> None:
        # 真实标签与预测标签都挤在一个类里：准确率 1.0，但随机也能对上 → κ = 0。
        matrix = metrics.ConfusionMatrix.from_pairs(
            [("正常", "正常")] * 4, labels.RISK_LEVELS
        )
        macro = matrix.macro()
        assert macro.accuracy == pytest.approx(1.0)
        assert macro.kappa == 0.0
        assert macro.weighted_f1 == pytest.approx(1.0)
        assert macro.macro_f1 == pytest.approx(1 / 3)  # 两个空类各记 0

    def test_empty_matrix(self) -> None:
        matrix = metrics.ConfusionMatrix(labels.RISK_LEVELS)
        macro = matrix.macro()
        assert matrix.total == 0
        assert matrix.accuracy == 0.0
        assert macro.n == 0
        assert macro.accuracy == 0.0
        assert macro.macro_f1 == 0.0
        assert macro.weighted_f1 == 0.0
        assert macro.kappa == 0.0

    def test_to_dict_and_markdown(self) -> None:
        matrix = metrics.ConfusionMatrix.from_pairs(
            [("正常", "正常"), ("高危", "预警")], labels.RISK_LEVELS
        )
        payload = matrix.to_dict()
        assert payload["labels"] == list(labels.RISK_LEVELS)
        assert payload["matrix"][0] == [1, 0, 0]
        assert payload["counts"] == {"正常->正常": 1, "高危->预警": 1}
        assert payload["per_class"][0]["label"] == "正常"
        assert payload["macro"]["n"] == 2

        table = matrix.render_markdown(caption="### 表")
        assert table.startswith("### 表")
        assert "| 真实 \\ 预测 | 正常 | 预警 | 高危 | 合计 |" in table
        assert "**1**" in table  # 对角线加粗
        assert table.count("|") > 10

        summary = matrix.render_metrics_markdown()
        assert "Cohen's κ" in summary
        assert "准确率" in summary


class TestRankCorrelation:
    def test_ranks_with_ties(self) -> None:
        assert metrics.ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
        assert metrics.ranks([]) == []

    def test_spearman_monotone(self) -> None:
        assert metrics.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
        assert metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        # 单调但非线性 → 秩相关仍是 1。
        assert metrics.spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.0)

    def test_spearman_degenerate_inputs(self) -> None:
        assert metrics.spearman([1, 1, 1], [1, 2, 3]) == 0.0  # 常量无秩
        assert metrics.spearman([1, 2], [1, 2, 3]) == 0.0     # 长度不一致
        assert metrics.spearman([1], [1]) == 0.0              # 样本太少
        assert metrics.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1.0)

    def test_pearson(self) -> None:
        assert metrics.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert metrics.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
        assert metrics.pearson([1, 1, 1], [1, 2, 3]) == 0.0  # 零方差
        assert metrics.pearson([1, 2], [1, 2, 3]) == 0.0     # 长度不一致


class TestDescribe:
    def test_distribution(self) -> None:
        stats = metrics.describe([1.0, 2.0, 3.0, 4.0])
        assert stats == {
            "count": 4, "min": 1.0, "p25": 1.75, "median": 2.5,
            "p75": 3.25, "max": 4.0, "mean": 2.5,
        }

    def test_single_value(self) -> None:
        assert metrics.describe([7.5])["median"] == 7.5
        assert metrics.describe([7.5])["count"] == 1

    def test_empty(self) -> None:
        assert metrics.describe([]) == {
            "count": 0, "min": 0.0, "p25": 0.0, "median": 0.0,
            "p75": 0.0, "max": 0.0, "mean": 0.0,
        }


# ===========================================================================
# 7. runner：黑盒调用引擎的记账
# ===========================================================================


class TestRunEngine:
    def test_collects_predictions_and_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple] = []

        def fake(readings, pond_id="unknown", species="鲤科", horizon_hours=72):
            calls.append((readings, pond_id, species, horizon_hours))
            return {
                "status": "ok",
                "risk_level": "高危",
                "risk_score": 88.5,
                "limiting_factor": "ammonia_n",
                "rule_hits": ["ammonia_redline"],
                "redline_hits": ["ammonia_n"],
            }

        monkeypatch.setattr("fisheries.tools.water_risk.water_risk_predict", fake)
        window = make_window(pond_id="Pond-3", true_level="高危", limiting=("ammonia_n",))

        samples, stats = runner.run_engine([window], species="罗非鱼", horizon_hours=48)

        assert stats["n_calls"] == 1
        assert stats["n_ok"] == 1
        assert stats["n_failed"] == 0
        assert stats["ms_per_call"] >= 0.0
        assert stats["errors"] == []
        assert calls[0][1] == "Pond-3"
        assert calls[0][2] == "罗非鱼"
        assert calls[0][3] == 48
        assert calls[0][0] == list(window.readings)

        sample = samples[0]
        assert sample.ok is True
        assert sample.risk_level == "高危"
        assert sample.risk_score == pytest.approx(88.5)
        assert sample.limiting_factor == "ammonia_n"
        assert sample.rule_hits == ("ammonia_redline",)
        assert sample.redline_hits == ("ammonia_n",)
        assert sample.true_level == "高危"
        assert sample.true_limiting == ("ammonia_n",)
        assert sample.n_points == 3
        assert sample.sampled_on == date(2021, 6, 19)
        assert sample.error == ""

    def test_limit_stops_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fisheries.tools.water_risk.water_risk_predict",
            lambda readings, **kwargs: {"status": "ok", "risk_level": "正常", "risk_score": 1.0},
        )
        windows = [make_window(day=19 + i) for i in range(4)]
        samples, stats = runner.run_engine(windows, limit=2)
        assert len(samples) == 2
        assert stats["n_calls"] == 2

    def test_exception_is_accounted_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(readings, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("fisheries.tools.water_risk.water_risk_predict", boom)
        samples, stats = runner.run_engine([make_window(pond_id="P9")])

        sample = samples[0]
        assert sample.ok is False
        assert sample.status == "error"
        assert sample.risk_level == ""
        assert sample.risk_score == 0.0
        assert "RuntimeError: boom" in sample.error
        assert any("P9" in entry for entry in stats["errors"])
        # 一次异常只算一次失败、只留一行 error（源码已修：except 分支不再重复记账）。
        assert stats["n_failed"] == 1
        assert stats["n_ok"] == 0
        assert len([entry for entry in stats["errors"] if "RuntimeError: boom" in entry]) == 1

    def test_error_status_is_counted_as_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fisheries.tools.water_risk.water_risk_predict",
            lambda readings, **kwargs: {
                "status": "error",
                "error_code": "INVALID_INPUT",
                "error_message": "入参校验失败 → readings: 太短",
                "risk_level": None,
                "risk_score": None,
            },
        )
        samples, stats = runner.run_engine([make_window()])
        assert stats["n_ok"] == 0
        assert stats["n_failed"] == 1
        assert samples[0].status == "error"
        assert samples[0].risk_score == 0.0
        assert "入参校验失败" in samples[0].error
        assert stats["errors"] == ["P1 2021-06-19: 入参校验失败 → readings: 太短"]

    def test_error_without_message_keeps_errors_list_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "fisheries.tools.water_risk.water_risk_predict",
            lambda readings, **kwargs: {"status": "error", "error_code": "COMPUTE_FAILED"},
        )
        samples, stats = runner.run_engine([make_window()])
        assert stats["n_failed"] == 1
        # 没有 error_message 时退化成 error_code，而不是留空。
        assert stats["errors"] == ["P1 2021-06-19: COMPUTE_FAILED"]
        assert samples[0].error == "COMPUTE_FAILED"

    def test_missing_status_counts_as_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fisheries.tools.water_risk.water_risk_predict",
            lambda readings, **kwargs: {"risk_level": "正常", "risk_score": 3.0},
        )
        samples, stats = runner.run_engine([make_window()])
        assert stats["n_failed"] == 1
        assert samples[0].status == "error"
        assert samples[0].risk_level == "正常"

    def test_mixed_ok_and_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        results = iter(
            [
                {"status": "ok", "risk_level": "正常", "risk_score": 5.0},
                {"status": "ok", "risk_level": "高危", "risk_score": 90.0},
            ]
        )

        def fake(readings, **kwargs):
            return next(results)

        monkeypatch.setattr("fisheries.tools.water_risk.water_risk_predict", fake)
        windows = [make_window(day=19), make_window(day=20)]
        samples, stats = runner.run_engine(windows, limit=2)
        assert (stats["n_calls"], stats["n_ok"], stats["n_failed"]) == (2, 2, 0)
        assert [s.risk_score for s in samples] == [5.0, 90.0]

    def test_empty_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "fisheries.tools.water_risk.water_risk_predict",
            lambda readings, **kwargs: {"status": "ok"},
        )
        samples, stats = runner.run_engine([])
        assert samples == []
        assert stats["n_calls"] == 0
        assert stats["ms_per_call"] == 0.0
        assert stats["n_ok"] == 0


class TestEvalRun:
    def test_usable_excludes_failures(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(),
                make_eval_sample(status="error", risk_level="", risk_score=0.0),
            ]
        )
        assert len(run.samples) == 2
        assert len(run.usable) == 1
        assert run.confusion().total == 1  # 失败样本不进混淆矩阵

    def test_default_name_and_engine_stats_copy(self) -> None:
        run = runner.evaluate([make_eval_sample()])
        assert run.name == "trend/window=3.0h"
        assert run.mode == "trend"
        assert run.window_hours == pytest.approx(3.0)
        stats = {"n_calls": 1}
        run2 = runner.evaluate([], name="自定义", mode="instant", window_hours=6.0,
                               engine_stats=stats)
        assert run2.name == "自定义"
        assert run2.mode == "instant"
        assert run2.window_hours == pytest.approx(6.0)
        assert run2.engine_stats == stats
        assert run2.engine_stats is not stats

    def test_confusion_matrix_and_accuracy(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=90.0),
                make_eval_sample(true_level="正常", risk_level="正常", risk_score=5.0),
                make_eval_sample(true_level="正常", risk_level="高危", risk_score=60.0, day=20),
                make_eval_sample(true_level="预警", risk_level="预警", risk_score=45.0, day=21),
            ]
        )
        matrix = run.confusion()
        assert matrix.total == 4
        assert matrix.accuracy == pytest.approx(0.75)
        assert matrix.count("正常", "高危") == 1

    def test_late_prediction_rate(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="高危", risk_level="正常", risk_score=30.0),
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=90.0, day=20),
            ]
        )
        assert run.late_prediction_rate() == pytest.approx(0.5)
        assert runner.evaluate([make_eval_sample()]).late_prediction_rate() == 0.0

    def test_false_alarm_rate(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="正常", risk_level="高危", risk_score=80.0),
                make_eval_sample(true_level="正常", risk_level="正常", risk_score=5.0, day=20),
            ]
        )
        assert run.false_alarm_rate() == pytest.approx(0.5)
        assert runner.evaluate([make_eval_sample(true_level="高危")]).false_alarm_rate() == 0.0

    def test_redline_recall(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="高危", risk_level="高危", redline_hits=("ammonia_n",)),
                make_eval_sample(true_level="高危", risk_level="预警", day=20),
                make_eval_sample(true_level="正常", risk_level="正常", day=21),
            ]
        )
        assert run.redline_recall() == (1, 2)
        assert runner.evaluate([make_eval_sample()]).redline_recall() == (0, 0)

    def test_limiting_agreement(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_limiting=("ammonia_n",), limiting_factor="ammonia_n"),
                make_eval_sample(true_limiting=("dissolved_oxygen",), limiting_factor="ph",
                                 day=20),
                make_eval_sample(true_limiting=(), limiting_factor="ph", day=21),
                make_eval_sample(true_limiting=("ph",), limiting_factor="", day=22),
            ]
        )
        assert run.limiting_agreement() == (1, 2)

    def test_score_spearman(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="正常", risk_level="正常", risk_score=10.0),
                make_eval_sample(true_level="预警", risk_level="预警", risk_score=50.0, day=20),
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=90.0, day=21),
            ]
        )
        assert run.score_spearman() == pytest.approx(1.0)
        reversed_run = runner.evaluate(
            [
                make_eval_sample(true_level="正常", risk_level="正常", risk_score=90.0),
                make_eval_sample(true_level="预警", risk_level="预警", risk_score=50.0, day=20),
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=10.0, day=21),
            ]
        )
        assert reversed_run.score_spearman() == pytest.approx(-1.0)
        # 少于 3 个可用样本 → 直接 0.0（相关无意义）。
        assert runner.evaluate([make_eval_sample()]).score_spearman() == 0.0

    def test_distributions_and_score_by_level(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="正常", risk_level="预警", risk_score=20.0),
                make_eval_sample(true_level="正常", risk_level="预警", risk_score=40.0, day=20),
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=95.0, day=21),
            ]
        )
        assert run.true_distribution() == {"正常": 2, "预警": 0, "高危": 1}
        assert run.severity_distribution() == {"正常": 0, "预警": 2, "高危": 1}
        by_level = run.score_by_true_level()
        assert by_level["正常"]["count"] == 2
        assert by_level["正常"]["median"] == pytest.approx(30.0)
        assert by_level["预警"]["count"] == 0

    def test_per_pond(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(pond_id="P1", true_level="正常", risk_level="正常"),
                make_eval_sample(pond_id="P1", true_level="高危", risk_level="预警", day=20),
                make_eval_sample(pond_id="P2", true_level="高危", risk_level="高危",
                                 risk_score=90.0, day=21),
            ]
        )
        per_pond = run.per_pond()
        assert sorted(per_pond) == ["P1", "P2"]
        assert per_pond["P1"]["n"] == 2
        assert per_pond["P1"]["accuracy"] == pytest.approx(0.5)
        assert per_pond["P1"]["true"] == {"正常": 1, "预警": 0, "高危": 1}
        assert per_pond["P1"]["pred"] == {"正常": 1, "预警": 1, "高危": 0}
        assert per_pond["P2"]["accuracy"] == pytest.approx(1.0)

    def test_misses_and_false_alarms(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="高危", risk_level="预警", risk_score=60.0),
                make_eval_sample(true_level="高危", risk_level="正常", risk_score=80.0, day=20),
                make_eval_sample(true_level="正常", risk_level="高危", risk_score=70.0, day=21),
                make_eval_sample(true_level="正常", risk_level="高危", risk_score=50.0, day=22),
            ]
        )
        assert [s.risk_score for s in run.misses()] == [80.0, 60.0]      # 最冤的在前
        assert [s.risk_score for s in run.false_alarms()] == [50.0, 70.0]  # 最没道理的在前
        assert len(run.misses(limit=1)) == 1

    def test_to_dict(self) -> None:
        run = runner.evaluate(
            [
                make_eval_sample(true_level="高危", risk_level="高危", risk_score=90.0,
                                 redline_hits=("ammonia_n",), true_limiting=("ammonia_n",),
                                 limiting_factor="ammonia_n"),
                make_eval_sample(true_level="正常", risk_level="正常", risk_score=5.0, day=20),
                make_eval_sample(true_level="预警", risk_level="正常", risk_score=30.0, day=21),
                make_eval_sample(status="error", risk_level="", risk_score=0.0, day=22),
            ],
            engine_stats={"n_calls": 4, "ms_per_call": 1.5},
        )
        payload = run.to_dict()
        assert payload["name"] == "trend/window=3.0h"
        assert payload["mode"] == "trend"
        assert payload["n_samples"] == 4
        assert payload["n_usable"] == 3
        assert payload["engine"] == {"n_calls": 4, "ms_per_call": 1.5}
        assert payload["confusion"]["macro"]["n"] == 3
        assert payload["spearman_score_vs_level"] == pytest.approx(run.score_spearman(), abs=1e-4)
        assert payload["redline_recall"] == {"hit": 1, "total": 1, "rate": 1.0}
        assert payload["limiting_agreement"]["total"] == 1
        assert payload["limiting_agreement"]["hit"] == 1
        assert payload["late_prediction_rate"] == 0.0
        assert payload["false_alarm_rate"] == 0.0
        assert set(payload["severity_distribution"]) == set(labels.RISK_LEVELS)
        assert set(payload["true_distribution"]) == set(labels.RISK_LEVELS)
        assert set(payload["score_by_true_level"]) == set(labels.RISK_LEVELS)
        assert set(payload["per_pond"]) == {"P1"}


# ===========================================================================
# 8. do_trend：引擎外推 vs 持久性基线
# ===========================================================================


class TestIterDoSeries:
    def test_series_shape_and_ids(self) -> None:
        by_pond = {"P1": make_do_series("P1", 30)}
        series = list(do_trend.iter_do_series(by_pond))
        # range(18-1, 30-6, 6) → end = 17, 23
        assert [s.series_id for s in series] == ["P1#17", "P1#23"]
        assert all(s.n_lags == 18 for s in series)
        assert all(s.pond_id == "P1" for s in series)
        assert all(s.usable for s in series)
        assert series[0].last == pytest.approx(1.0 + 0.1 * 17)
        assert series[0].target == pytest.approx(1.0 + 0.1 * 23)
        assert series[0].origin == "2021-06-19T02:50:00+00:00"

    def test_target_windows_are_not_reused_when_stride_equals_history(self) -> None:
        # 步长 = lead_points（默认）；当 history == stride 时历史窗口与目标都不重叠。
        by_pond = {"P1": make_do_series("P1", 30)}
        series = list(do_trend.iter_do_series(by_pond, history_points=6, lead_points=6))
        assert [s.series_id for s in series] == ["P1#5", "P1#11", "P1#17", "P1#23"]
        used: set[int] = set()
        for item in series:
            indices = {
                round((value - 1.0) / 0.1) for value in item.lags
            }  # 溶氧值唯一确定下标
            assert not (indices & used)
            used |= indices
            target_index = round((float(item.target) - 1.0) / 0.1)
            assert target_index == round((item.last - 1.0) / 0.1) + 6

    def test_limit_per_pond(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40)}
        series = list(do_trend.iter_do_series(by_pond, limit_per_pond=1))
        assert [s.series_id for s in series] == ["P1#17"]

    def test_too_few_samples(self) -> None:
        by_pond = {"P1": make_do_series("P1", 23)}
        assert list(do_trend.iter_do_series(by_pond)) == []

    def test_ponds_are_iterated_in_sorted_order(self) -> None:
        by_pond = {"P2": make_do_series("P2", 30), "P1": make_do_series("P1", 30)}
        series = list(do_trend.iter_do_series(by_pond))
        assert [s.pond_id for s in series] == ["P1", "P1", "P2", "P2"]

    def test_invalid_arguments(self) -> None:
        by_pond = {"P1": make_do_series("P1", 30)}
        with pytest.raises(ValueError):
            list(do_trend.iter_do_series(by_pond, history_points=1))
        with pytest.raises(ValueError):
            list(do_trend.iter_do_series(by_pond, lead_points=0))


class TestPondDoSeries:
    def test_properties(self) -> None:
        item = do_trend.PondDoSeries("P1#5", "P1", (1.0, 2.0, 3.0), target=4.0)
        assert item.n_lags == 3
        assert item.last == 3.0
        assert item.usable is True

    def test_no_target_is_not_usable(self) -> None:
        item = do_trend.PondDoSeries("P1#5", "P1", (1.0, 2.0), target=None)
        assert item.usable is False

    def test_single_lag_is_not_usable(self) -> None:
        item = do_trend.PondDoSeries("P1#5", "P1", (1.0,), target=2.0)
        assert item.usable is False
        with pytest.raises(ValueError):
            do_trend.engine_next(item, lead_hours=1.0)


class TestProjections:
    def test_persistence_returns_the_last_lag(self) -> None:
        item = do_trend.PondDoSeries("P1#5", "P1", (1.0, 2.0, 2.5), target=3.0)
        assert do_trend.persistence_next(item) == 2.5

    def test_engine_next_is_last_plus_slope_times_lead(self) -> None:
        lags = tuple(1.0 + 0.1 * i for i in range(18))
        item = do_trend.PondDoSeries("P1#17", "P1", lags, target=3.3)
        # 每点 0.1、步长 10 分钟 → 斜率 0.6 / 小时。
        assert do_trend.engine_next(item, step_minutes=10, lead_hours=1.0) == pytest.approx(3.3)
        assert do_trend.engine_next(item, step_minutes=10, lead_hours=2.0) == pytest.approx(3.9)
        assert do_trend.engine_next(item, step_minutes=60, lead_hours=1.0) == pytest.approx(2.8)

    def test_engine_next_needs_two_lags(self) -> None:
        item = do_trend.PondDoSeries("P1#0", "P1", (1.0,), target=2.0)
        with pytest.raises(ValueError):
            do_trend.engine_next(item, lead_hours=1.0)


class TestScoreSeries:
    def test_linear_ramp_favours_the_engine(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40, per_point=0.1)}
        series = list(do_trend.iter_do_series(by_pond, history_points=6, lead_points=6))
        score = do_trend.score_series(series, step_minutes=10, lead_hours=1.0)

        assert score.n == 5
        assert score.engine_mae == pytest.approx(0.0)   # 完全落在同一条直线上
        assert score.engine_rmse == pytest.approx(0.0)
        assert score.persistence_mae == pytest.approx(0.6)
        assert score.improvement_pct == pytest.approx(100.0)
        assert score.engine_better == 5
        assert score.persistence_better == 0
        assert score.ties == 0
        assert score.win_rate == pytest.approx(1.0)
        assert score.direction_hits == 5
        assert score.direction_accuracy == pytest.approx(1.0)
        assert score.decline_n == 0
        assert score.decline_recall == 0.0

    def test_declining_series_recall(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40, do0=10.0, per_point=-0.1)}
        series = list(do_trend.iter_do_series(by_pond, history_points=6, lead_points=6))
        score = do_trend.score_series(series, step_minutes=10, lead_hours=1.0)
        assert score.decline_n == 5
        assert score.decline_hits == 5
        assert score.decline_recall == pytest.approx(1.0)
        assert score.direction_accuracy == pytest.approx(1.0)

    def test_flat_series_is_a_tie(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40, per_point=0.0)}
        series = list(do_trend.iter_do_series(by_pond, history_points=6, lead_points=6))
        score = do_trend.score_series(series, step_minutes=10, lead_hours=1.0)
        assert score.ties == score.n == 5
        assert score.engine_better == 0
        assert score.engine_mae == pytest.approx(0.0)
        assert score.persistence_mae == pytest.approx(0.0)
        assert score.improvement_pct == 0.0   # 基线 MAE 为 0，改善率无定义 → 0
        assert score.win_rate == 0.0          # 没有分出胜负

    def test_unusable_series_are_skipped(self) -> None:
        usable = do_trend.PondDoSeries("P1#17", "P1", tuple(1.0 + 0.1 * i for i in range(6)),
                                       target=1.6)
        no_target = do_trend.PondDoSeries("P1#23", "P1", (1.0, 2.0, 3.0), target=None)
        too_short = do_trend.PondDoSeries("P1#29", "P1", (1.0,), target=2.0)
        score = do_trend.score_series([usable, no_target, too_short], step_minutes=10,
                                      lead_hours=1.0)
        assert score.n == 1

    def test_empty_input(self) -> None:
        score = do_trend.score_series([], step_minutes=10, lead_hours=1.0)
        assert score == do_trend.ProjectionScore()
        assert score.n == 0
        assert score.improvement_pct == 0.0
        assert score.direction_accuracy == 0.0
        assert score.decline_recall == 0.0
        assert score.win_rate == 0.0


class TestProjectionScoreProperties:
    def test_improvement_pct(self) -> None:
        assert do_trend.ProjectionScore(n=1, engine_mae=0.5,
                                        persistence_mae=1.0).improvement_pct == pytest.approx(50.0)
        assert do_trend.ProjectionScore(n=1, engine_mae=1.5,
                                        persistence_mae=1.0).improvement_pct == pytest.approx(-50.0)
        assert do_trend.ProjectionScore(n=1, engine_mae=0.5,
                                        persistence_mae=0.0).improvement_pct == 0.0

    def test_rates(self) -> None:
        score = do_trend.ProjectionScore(
            n=4, engine_better=3, persistence_better=1, ties=0,
            direction_hits=3, direction_n=4, decline_n=2, decline_hits=1,
        )
        assert score.win_rate == pytest.approx(0.75)
        assert score.direction_accuracy == pytest.approx(0.75)
        assert score.decline_recall == pytest.approx(0.5)


class TestDoTrendReport:
    def test_verdict_without_series(self) -> None:
        assert do_trend.DoTrendReport().verdict() == "没有可用的溶氧序列，无法比较。"

    def test_verdict_better_than_baseline(self) -> None:
        report_obj = do_trend.DoTrendReport(
            overall=do_trend.ProjectionScore(n=10, engine_mae=0.5, persistence_mae=1.0),
            by_history={6: do_trend.ProjectionScore(n=5)},
            n_ponds=2,
        )
        text = report_obj.verdict()
        assert "外推优于基线 50.0%" in text
        assert "1 小时后" in text

    def test_verdict_worse_than_baseline(self) -> None:
        report_obj = do_trend.DoTrendReport(
            overall=do_trend.ProjectionScore(n=10, engine_mae=1.5, persistence_mae=1.0),
            by_history={6: do_trend.ProjectionScore(n=5)},
            n_ponds=1,
        )
        text = report_obj.verdict()
        assert "外推**差于**基线 50.0%" in text
        assert "点预测不该采信" in text

    def test_verdict_tie(self) -> None:
        report_obj = do_trend.DoTrendReport(
            overall=do_trend.ProjectionScore(n=10, engine_mae=0.5, persistence_mae=0.5),
            by_history={6: do_trend.ProjectionScore(n=5)},
        )
        assert "两者基本打平" in report_obj.verdict()

    def test_verdict_threshold_is_a_strict_one_percent(self) -> None:
        under = do_trend.DoTrendReport(
            overall=do_trend.ProjectionScore(n=10, engine_mae=0.995, persistence_mae=1.0),
        )
        assert "两者基本打平" in under.verdict()  # 0.5% 的改善不算「优于」
        # 判据是 `improvement > 1.0`：浮点让「正好 1%」算成 1.0000000000000002，
        # 于是落进「优于」分支。钉住它，免得以后把阈值悄悄改成 >=。
        edge = do_trend.DoTrendReport(
            overall=do_trend.ProjectionScore(n=10, engine_mae=0.99, persistence_mae=1.0),
        )
        assert "优于基线 1.0%" in edge.verdict()

    def test_lead_hours_and_sources_defaults(self) -> None:
        report_obj = do_trend.DoTrendReport()
        assert report_obj.lead_hours == pytest.approx(1.0)
        assert report_obj.step_minutes == 10
        assert report_obj.sources == ()


class TestEvaluateDoTrend:
    def test_end_to_end_on_synthetic_pond(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40)}
        trend = do_trend.evaluate_do_trend(by_pond, history_points=18, lead_points=6)

        assert trend.n_ponds == 1
        assert trend.sources == ("P1",)
        assert trend.lead_hours == pytest.approx(1.0)
        assert trend.step_minutes == 10
        assert trend.n_series == 3
        assert trend.overall.n == 3
        # 6 / 12 / 18 点历史各跑一遍（都 ≤ history_points）。
        assert sorted(trend.by_history) == [6, 12, 18]
        assert trend.by_history[6].n == 5
        assert trend.by_history[12].n == 4
        assert trend.by_history[18].n == 3
        assert "外推优于基线" in trend.verdict()

    def test_history_choices_are_capped_by_history_points(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40)}
        trend = do_trend.evaluate_do_trend(by_pond, history_points=6, lead_points=6)
        assert sorted(trend.by_history) == [6]
        assert trend.n_series == 5

    def test_limit_per_pond(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40)}
        trend = do_trend.evaluate_do_trend(by_pond, limit_per_pond=1)
        assert trend.n_series == 1
        assert trend.overall.n == 1
        assert all(score.n == 1 for score in trend.by_history.values())

    def test_multiple_ponds(self) -> None:
        by_pond = {"P1": make_do_series("P1", 40), "P2": make_do_series("P2", 40, do0=6.0)}
        trend = do_trend.evaluate_do_trend(by_pond)
        assert trend.n_ponds == 2
        assert trend.sources == ("P1", "P2")
        assert trend.overall.n == 6


# ===========================================================================
# 9. report：渲染不炸 + 那条警告
# ===========================================================================


def make_data_stats() -> dict:
    return {
        "n_files": 11,
        "n_lines": 1_114_970,
        "n_samples": 496_677,
        "n_bad_timestamp": 3,
        "n_dropped_missing_required": 600_000,
        "n_out_of_range": 1234,
        "per_file": [
            {"pond_id": "IoTPond7", "growth_pct": 12360.0},
            {"pond_id": "IoTPond6", "growth_pct": 6900.0},
            {"pond_id": "IoTPond8", "growth_pct": None},
        ],
    }


def make_label_distribution() -> dict:
    return {
        "n_samples": 4,
        "n_ponds": 2,
        "label_counts": {"正常": 2, "预警": 1, "高危": 1},
        "time_from": "2021-06-19T00:00:00+00:00",
        "time_to": "2021-06-19T02:00:00+00:00",
        "samples_per_pond_min": 1,
        "samples_per_pond_max": 3,
        "per_pond": {"P1": 3, "P2": 1},
        "per_pond_detail": {
            "P1": {"n_samples": 3, "正常": 1, "预警": 1, "高危": 1,
                   "high_risk_rate": 1 / 3},
            "P2": {"n_samples": 1, "正常": 1, "预警": 0, "高危": 0,
                   "high_risk_rate": 0.0},
        },
    }


def make_coherence() -> labels.CoherenceReport:
    return labels.growth_coherence(
        [
            labels.PondOutcome(f"P{i}", 10, rate, 0.0, growth_pct=growth)
            for i, (rate, growth) in enumerate(
                [(0.0, 300.0), (0.1, 200.0), (0.2, 100.0), (0.3, 0.0)]
            )
        ]
    )


class TestRenderRunSection:
    def test_renders_matrix_rates_and_level_table(self) -> None:
        run = make_run()
        text = report.render_run_section(run, caption="### 4.1 主口径", notes=["注意 A"])

        assert "### 4.1 主口径" in text
        assert "**混淆矩阵**" in text
        assert "| 真实 \\ 预测 | 正常 | 预警 | 高危 | 合计 |" in text
        assert "**漏报率**" in text
        assert "**误报率**" in text
        assert "**红线召回**" in text
        assert "**主控因子一致率**" in text
        assert "`risk_score` 随真值等级的变化" in text
        assert "- 注意 A" in text
        assert "κ =" in text and "宏平均 F1" in text

    def test_level_table_can_be_skipped(self) -> None:
        run = make_run()
        text = report.render_run_section(run, include_level_table=False)
        assert "`risk_score` 随真值等级的变化" not in text
        assert "**混淆矩阵**" in text

    def test_default_includes_the_level_table(self) -> None:
        assert "`risk_score` 随真值等级的变化" in report.render_run_section(make_run())

    def test_old_keyword_is_gone(self) -> None:
        with pytest.raises(TypeError):
            report.render_run_section(make_run(), include_pollution_table=True)

    def test_failed_samples_are_reported_as_excluded(self) -> None:
        run = runner.evaluate(
            [make_eval_sample(), make_eval_sample(status="error", risk_level="", day=20)]
        )
        text = report.render_run_section(run)
        assert "样本 1 个（工具失败 1 个，不计入指标）" in text

    def test_no_caption_means_no_header(self) -> None:
        text = report.render_run_section(make_run())
        assert "### 4.1" not in text


class TestRenderReport:
    def render(self, **overrides: object) -> str:
        kwargs = {
            "data_stats": make_data_stats(),
            "label_distribution": make_label_distribution(),
            "primary": make_run(),
            "sensitivity": [make_run()],
            "skipped": 7,
            "generated_at": "2026-01-01 00:00",
            "species": "鲤科",
            "horizon_hours": 72,
            "coherence": make_coherence(),
            "do_trend": do_trend.evaluate_do_trend({"P1": make_do_series("P1", 40)}),
            "conclusion": "结论一句话",
            "limitations": ["局限一"],
        }
        kwargs.update(overrides)
        return report.render_report(**kwargs)

    def test_independence_warning_is_rendered_verbatim(self) -> None:
        text = self.render()
        # report.py 里的原话：「⚠ **正因为标签由同一套阈值算出，第一组验证不是独立的。**」
        assert "正因为标签由同一套阈值算出，第一组验证不是独立的" in text
        assert "第一组验证不是独立的" in text

    def test_main_sections_are_present(self) -> None:
        text = self.render()
        assert text.startswith("# M8 外部验证报告")
        assert "## 2. 数据与标签" in text
        assert "### 2.3 标签口径：最差指标决定论" in text
        assert "## 3. 评测协议" in text
        assert "## 4. 外部验证一" in text
        assert "### 4.1 主口径" in text
        assert "### 4.2 敏感性分析" in text
        assert "### 4.3 逐塘分解" in text
        assert "### 4.4 误报与漏报清单" in text
        assert "## 5. 外部验证二" in text
        assert "## 6. 结论与局限" in text
        assert "结论一句话" in text
        assert "- 局限一" in text

    def test_dataset_and_growth_examples(self) -> None:
        text = self.render()
        assert "aquaponics_pond" in text
        assert "IoTPond7" in text
        assert "12,360%" in text          # 生长最显著的塘被举为例
        assert "1,114,970" in text

    def test_growth_section_omitted_without_coherence(self) -> None:
        text = self.render(coherence=None)
        assert "### 2.6 生长自洽性检验" not in text
        assert "## 2. 数据与标签" in text  # 其余章节照旧

    def test_do_trend_section_fallback(self) -> None:
        text = self.render(do_trend=None)
        assert "## 5. 外部验证二：溶氧趋势外推\n\n未运行。" in text

    def test_do_trend_section_renders_numbers(self) -> None:
        text = self.render()
        assert "**历史窗口 6 点**" not in text  # 表头是「历史窗口 6 点」不带 **
        assert "| 历史窗口 6 点 | 5 |" in text
        assert "**结论**：" in text

    def test_vision_section_fallback(self) -> None:
        text = self.render()
        assert "未运行——本机没有 `data/derived/metrics.json`" in text

    def test_vision_section_when_metrics_exist(self) -> None:
        metrics_payload = {
            "model": "yolov8-fish.pt",
            "test_size": 120,
            "top1_accuracy": 0.875,
            "macro_f1": 0.81,
            "mAP50_weak_fullframe": 0.72,
            "hyperparams": {"epochs": 50, "imgsz": 224, "batch": 16, "seed": 42},
            "dataset": "fish_disease",
            "identity_rule": "按图片指纹切分",
            "mAP50_caveat": "弱口径",
            "classes": [{"key": "healthy", "zh": "健康鱼"}],
            "per_class": {"healthy": {"support": 30, "precision": 0.9, "recall": 0.9,
                                      "f1": 0.9}},
        }
        text = self.render(vision_metrics=metrics_payload)
        assert "## 8. 附：图像模态" in text
        assert "### 8.1 测试集指标（120 张" in text
        assert "87.5%" in text
        assert "未运行——本机没有" not in text

    def test_minimal_inputs_do_not_crash(self) -> None:
        text = report.render_report(
            data_stats={},
            label_distribution={},
            primary=make_run(),
            sensitivity=[],
            skipped=0,
            generated_at="—",
            species="鲤科",
            horizon_hours=72,
        )
        assert isinstance(text, str)
        assert "## 2. 数据与标签" in text
        assert "未运行——本机没有" in text
        assert "第一组验证不是独立的" in text

    def test_render_report_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            report.render_report({})  # type: ignore[misc]


class TestReportHelpers:
    def test_fmt_rate_handles_zero_total(self) -> None:
        assert report._fmt_rate(0, 0) == "—"
        assert report._fmt_rate(1, 4) == "1/4（25.0%）"

    def test_sample_line_uses_placeholders_for_missing_values(self) -> None:
        sample = make_eval_sample(true_level="高危", risk_level="", true_limiting=("ph",))
        line = report._sample_line(sample)
        assert "| P1 | 2021-06-19 | 高危（ph） | **—** | 10.0 | — |" in line
