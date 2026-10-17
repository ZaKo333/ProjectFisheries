"""服务端水质表格摄入层（``fisheries.tables``）的单元测试。

浏览器端另有一份职责相同的解析器（``web/js/parse.js``），它不在这套测试的覆盖
范围内 —— 本文件只钉服务端这一侧。三层，跟模块划分一一对应：

* ``TestHeaderMatching`` —— 真实数据集里的表头五花八门，列名识别错了整个报告
  就是错的，所以这一层测得最细；
* ``TestValueParsing`` —— 单元格取值（全角、千分位、哨兵值、时间戳时区）；
* ``TestTableIngest`` —— 摄入质量（缺失列、负值、时间轴、间隔、编码）。

全部离线，不碰网络也不碰知识库。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from fisheries.tables import (
    EXAMPLE_TABLE,
    TableError,
    ingest_rows,
    ingest_table,
    map_columns,
    normalize_header,
    parse_timestamp,
    to_float,
)

_HEADERS = [
    "timestamp",
    "water_temp",
    "ph",
    "dissolved_oxygen",
    "ammonia_n",
    "nitrite_n",
]


def _rows(n: int = 6, *, do: float = 6.0, ammonia: float = 0.1) -> list[list[str]]:
    """造 n 行小时级、基本平稳的读数。"""
    return [
        [
            f"2026-10-13 {8 + i:02d}:00:00",
            "27.0",
            "8.10",
            f"{do + 0.01 * i:.2f}",
            f"{ammonia + 0.001 * i:.3f}",
            "0.05",
        ]
        for i in range(n)
    ]


def _readings(n: int = 6, *, do: float = 6.0, ammonia: float = 0.1):
    return ingest_rows(_HEADERS, _rows(n, do=do, ammonia=ammonia)).readings


# --------------------------------------------------------------------------- #
# 表头识别
# --------------------------------------------------------------------------- #

class TestHeaderMatching:
    def test_normalize_strips_units_and_case(self) -> None:
        assert normalize_header("Dissolved Oxygen(g/ml)") == "dissolvedoxygengml"
        assert normalize_header("Water_Temperature_C") == "watertemperaturec"
        assert normalize_header("  PH  ") == "ph"
        assert normalize_header("氨氮") == "氨氮"

    def test_real_dataset_pond1_headers(self) -> None:
        """IoTPond1.csv 的真实表头。"""
        headers = [
            "created_at", "entry_id", "Temperature (C)", "Turbidity(NTU)",
            "Dissolved Oxygen(g/ml)", "PH", "Ammonia(g/ml)", "Nitrate(g/ml)",
            "Population", "Fish_Length(cm)", "Fish_Weight(g)",
        ]
        mapping = map_columns(headers)
        assert mapping.columns["timestamp"] == "created_at"
        assert mapping.columns["water_temp"] == "Temperature (C)"
        assert mapping.columns["dissolved_oxygen"] == "Dissolved Oxygen(g/ml)"
        assert mapping.columns["ph"] == "PH"
        assert mapping.columns["ammonia_n"] == "Ammonia(g/ml)"
        # 只有硝酸盐、没有亚硝酸盐：不能顶替，必须报缺失
        assert "nitrite_n" not in mapping.columns
        assert mapping.columns["nitrate_n"] == "Nitrate(g/ml)"

    def test_pond10_typo_header_is_matched(self) -> None:
        """IoTPond10/11.csv 把 DISSOLVED 拼成了 DISOLVED。"""
        mapping = map_columns([
            "created_at", "TEMPERATURE", "DISOLVED OXYGEN", "pH", "AMMONIA",
        ])
        assert mapping.columns["dissolved_oxygen"] == "DISOLVED OXYGEN"

    def test_phosphorus_column_does_not_hijack_ph(self) -> None:
        """`Total_Phosphorus_mg_L` 里含 'ph'，绝不能拿来当 pH。

        这是当初放弃模糊匹配的直接原因：按子串匹配的话，
        ``china_water_pollution_data.csv`` 会把总磷当成 pH。
        """
        mapping = map_columns([
            "Date", "pH", "Total_Phosphorus_mg_L", "Ammonia_N_mg_L",
        ])
        assert mapping.columns["ph"] == "pH"

    def test_nitrate_and_nitrite_are_not_confused(self) -> None:
        mapping = map_columns([
            "Date", "Nitrate_mg_L", "Nitrite_mg_L", "Ammonia_N_mg_L",
        ])
        assert mapping.columns["nitrate_n"] == "Nitrate_mg_L"
        assert mapping.columns["nitrite_n"] == "Nitrite_mg_L"

    def test_separate_date_and_time_columns(self) -> None:
        mapping = map_columns(["date", "time", "temperature", "ph", "do"])
        assert mapping.columns["_date"] == "date"
        assert mapping.columns["_time"] == "time"
        assert "timestamp" not in mapping.columns

    def test_duplicate_alias_is_reported_not_silently_used(self) -> None:
        """两列都像氨氮时取第一个，但要把这件事说出来。"""
        mapping = map_columns(["timestamp", "ammonia", "Ammonia_N"])
        assert mapping.columns["ammonia_n"] == "ammonia"
        assert any("ammonia" in d for d in mapping.duplicates)

    def test_unmatched_columns_are_listed(self) -> None:
        mapping = map_columns(["timestamp", "Temperature", "Turbidity", "Population"])
        assert "Turbidity" in mapping.ignored
        assert "Population" in mapping.ignored


class TestValueParsing:
    @pytest.mark.parametrize("raw,expected", [
        ("4.505", 4.505),
        (" 8.1 ", 8.1),
        ("0", 0.0),
    ])
    def test_plain_numbers(self, raw: str, expected: float) -> None:
        assert to_float(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "N/A", "nan", "null", "-", "abc"])
    def test_non_numbers_become_none(self, raw: str) -> None:
        assert to_float(raw) is None

    def test_sensor_sentinels_are_treated_as_missing(self) -> None:
        """IoTPond10 用 -127 表示「没读到」，它不能当成真值。"""
        assert to_float("-127") is None
        assert to_float("-999") is None

    @pytest.mark.parametrize("raw", ["inf", "-inf", "Infinity", "-Infinity", "INF"])
    def test_infinite_text_is_treated_as_missing(self, raw: str) -> None:
        """``inf`` 能过 ``float()``，但它不是读数。

        放进去不会让这一格报错，而是让整份数据在工具层被整体拒收
        （``INVALID_INPUT``）——用户看到的是「上传成功、分析失败」，
        一个单元格毁掉整张表。导出工具算不出数时确实会写 ``inf``，
        所以这里必须当缺测，而不是当数值。
        """
        assert to_float(raw) is None

    @pytest.mark.parametrize("raw", [float("inf"), float("-inf")])
    def test_infinite_float_objects_are_treated_as_missing(self, raw: float) -> None:
        """表格层和 ``evaluation.ponds`` 走的是不同分支，两边都得挡住。"""
        assert to_float(raw) is None

    @pytest.mark.parametrize("raw,expected", [
        ("2021-06-19 00:00:05 CET", "2021-06-19 00:00:05"),
        ("2023-06-01", "2023-06-01 00:00:00"),
        ("2021/12/05 13:45", "2021-12-05 13:45:00"),
        ("2026-10-13T08:00:00", "2026-10-13 08:00:00"),
    ])
    def test_timestamp_formats(self, raw: str, expected: str) -> None:
        parsed = parse_timestamp(raw)
        assert parsed is not None
        assert parsed.strftime("%Y-%m-%d %H:%M:%S") == expected

    def test_timestamp_is_timezone_naive(self) -> None:
        """带 CET 的时间戳要丢掉时区，否则和没有时区的行混排会炸。"""
        parsed = parse_timestamp("2021-06-19 00:00:05 CET")
        assert parsed is not None
        assert parsed.tzinfo is None

    def test_utc_offset_is_converted_not_discarded(self) -> None:
        """``Z`` 与 ``+08:00`` 写的是同一个瞬间，解析结果必须相同。

        曾经这里是 ``dt.replace(tzinfo=None)``——直接抹掉偏移而不换算，
        于是同一时刻被读成差 8 小时的两个值，而且不报任何错：
        混合时区的表排序错乱、UTC 导出的表显示成错的墙上时间。
        """
        utc_marked = parse_timestamp("2026-07-20T00:00:00Z")
        separated = parse_timestamp("2026-07-20T08:00:00+08:00")
        assert utc_marked is not None
        assert separated is not None
        assert utc_marked == separated

    def test_naive_and_aware_land_on_the_same_instant(self) -> None:
        """带偏移的输入要换算到与纯数字分支同一口径。

        纯数字走 ``datetime.fromtimestamp``（本地时间），带偏移的走
        ``astimezone()``（同样本地时间）——两条分支不能各说各话。
        """
        from datetime import datetime

        epoch = parse_timestamp("1789000000")
        assert epoch is not None
        assert epoch == datetime.fromtimestamp(1789000000)

    def test_unparseable_timestamp_is_none(self) -> None:
        assert parse_timestamp("不是时间") is None


# --------------------------------------------------------------------------- #
# 摄入
# --------------------------------------------------------------------------- #

class TestTableIngest:
    def test_happy_path(self) -> None:
        result = ingest_rows(_HEADERS, _rows(6))
        assert result.ok
        assert result.n_points == 6
        assert result.interval_minutes == 60
        assert result.interval_source == "timestamp"
        assert result.errors == []
        assert result.readings[0]["water_temp"] == pytest.approx(27.0)

    def test_missing_required_column_is_an_error(self) -> None:
        result = ingest_rows(
            ["timestamp", "water_temp", "ph"], _rows(6),
        )
        assert not result.ok
        assert any("dissolved_oxygen" in e or "溶氧" in e for e in result.errors)

    def test_too_few_rows_is_not_ok(self) -> None:
        result = ingest_rows(_HEADERS, _rows(2))
        assert not result.ok
        assert result.n_points == 2

    def test_negative_values_become_missing_with_a_warning(self) -> None:
        rows = _rows(6)
        rows[3][3] = "-127"          # 溶氧
        result = ingest_rows(_HEADERS, rows)
        assert result.n_cells_missing.get("dissolved_oxygen") == 1
        # 必填项缺失 → 整行被丢弃（而不是留个 None 键，那会让工具报字段缺失）
        assert result.n_rows_incomplete == 1
        assert len(result.readings) == 5
        assert all("dissolved_oxygen" in r for r in result.readings)
        assert any("负值" in w for w in result.warnings)
        assert any("丢弃" in w for w in result.warnings)

    def test_dropping_rows_can_make_the_table_unusable(self) -> None:
        """丢到不足 3 个点时必须明确报错，而不是让工具层抛校验异常。"""
        rows = _rows(6)
        for i in range(4):
            rows[i][3] = ""          # 前四行没有溶氧
        result = ingest_rows(_HEADERS, rows)
        assert not result.ok
        assert result.n_rows_incomplete == 4
        assert any("只剩 2 个可用点" in e for e in result.errors)

    def test_infinite_values_are_counted_and_explained(self) -> None:
        """``inf`` 要按缺测处理，并且**说清楚它原本不是空的**。

        导出工具算不出数时会写 ``inf``，这一格跟空着的成因不一样：
        空着是没采集，``inf`` 是算溢出了。只报「缺失」会让人以为是设备没读数。
        """
        rows = _rows(6)
        for i in range(3):
            rows[i][4] = "inf"       # 氨氮
        result = ingest_rows(_HEADERS, rows)
        assert result.n_cells_missing.get("ammonia_n") == 3
        assert any("非有限值" in w for w in result.warnings)
        assert all(math.isfinite(r["ammonia_n"]) for r in result.readings)

    def test_truncation_keeps_the_latest_rows(self) -> None:
        result = ingest_rows(_HEADERS, _rows(48), max_rows=24)
        assert result.n_rows_used == 24
        assert result.n_rows_truncated == 24
        # 保留尾部：最早的点应该是第 24 行（08+24=次日 08 点）
        assert result.readings[0]["timestamp"].startswith("2026-10-14")

    def test_unordered_rows_are_sorted_and_reported(self) -> None:
        rows = _rows(6)
        rows.reverse()
        result = ingest_rows(_HEADERS, rows)
        assert result.interval_source == "timestamp"
        assert any("顺序" in w for w in result.warnings)

    def test_missing_timestamps_are_interpolated_without_shifting_others(
        self,
    ) -> None:
        """``water_risk._resolve_interval`` 要求**所有**点都有时间戳。

        只要有几行时间戳解析失败，工具就会整体退回 ``interval_minutes`` 参数，
        趋势外推的时长含义就错了。所以这里补一条单调时间轴——但补插必须
        **只动缺失行**：如果按「首点往后每步加一个间隔」去铺，补出来的点会
        抢走真实时间戳的位置，把后面所有实测点的时刻整体推后一格。
        """
        rows = _rows(6)
        rows[2][0] = "坏时间"          # 第 3 行（10:00）时间戳坏掉
        result = ingest_rows(_HEADERS, rows, interval_minutes=30)
        assert result.n_timestamps_failed == 1
        assert result.n_timestamp_filled == 1
        assert result.n_timestamps_parsed == 5
        assert any("时间戳" in w for w in result.warnings)

        stamps = [r["timestamp"] for r in result.readings]
        assert all(stamps)                                  # 一个不缺
        assert stamps == sorted(stamps)                     # 单调不降
        # 真实点一个都没被挪动
        assert stamps[0].endswith("08:00:00")
        assert stamps[1].endswith("09:00:00")
        assert stamps[3].endswith("11:00:00")
        assert stamps[5].endswith("13:00:00")
        # 坏掉的那一行落在两个已知邻居之间（09:00 与 11:00 的中间行序位置）
        assert stamps[2].startswith("2026-10-13T10:00:00")

    def test_no_timestamp_column_falls_back_to_parameter(self) -> None:
        result = ingest_rows(
            ["water_temp", "ph", "dissolved_oxygen", "ammonia_n"],
            [["27", "8.1", "6.0", "0.1"]] * 6,
            interval_minutes=30,
        )
        assert result.interval_source == "parameter"
        assert result.interval_minutes == 30
        assert all("timestamp" not in r for r in result.readings)

    def test_nitrate_only_warns_that_nitrite_is_missing(self) -> None:
        """数据集里全是硝酸盐，没有亚硝酸盐——这必须显式说出来。"""
        result = ingest_rows(
            ["timestamp", "Temperature", "pH", "Dissolved Oxygen", "Ammonia", "Nitrate"],
            [["2026-10-13 08:00:00", "27", "8.1", "6.0", "0.1", "2.0"]] * 6,
        )
        assert any("硝酸盐" in w and "亚硝酸盐" in w for w in result.warnings)

    def test_scale_mismatch_is_flagged(self) -> None:
        """IoTPond11 的氨氮量级是 12151，明显不是 mg/L。"""
        result = ingest_rows(
            _HEADERS, _rows(6, ammonia=12151.86),
        )
        assert any("氨氮" in w and "超出物理量程" in w for w in result.warnings)

    def test_summary_markdown_mentions_the_basics(self) -> None:
        text = ingest_rows(_HEADERS, _rows(6)).summary_markdown()
        assert "读入 6 行" in text
        assert "60 分钟" in text
        assert "列映射" in text

    def test_preview_has_header_and_rows(self) -> None:
        result = ingest_rows(_HEADERS, _rows(6))
        # 预览表头用中文标签，而不是规范字段名——这一栏是给养殖户看的
        assert result.preview[0][0] == "时间"
        assert result.preview[0][1] == "水温"
        assert len(result.preview) >= 2

    def test_example_table_is_valid(self) -> None:
        """仓库自带的示例表必须一直是可用的——演示第一步就靠它。"""
        result = ingest_table(EXAMPLE_TABLE, max_rows=48)
        assert result.ok, result.errors
        assert result.n_points == 24
        assert result.interval_minutes == 60
        assert result.interval_source == "timestamp"
        assert result.n_cells_missing == {}

    def test_unknown_suffix_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "data.json"
        bad.write_text("{}", encoding="utf-8")
        with pytest.raises(TableError, match="不认识的表格格式"):
            ingest_table(bad)

    def test_legacy_xls_gets_an_actionable_message(self, tmp_path: Path) -> None:
        bad = tmp_path / "old.xls"
        bad.write_bytes(b"\xd0\xcf\x11\xe0")
        with pytest.raises(TableError, match="另存为"):
            ingest_table(bad)

    def test_missing_file_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(TableError, match="找不到"):
            ingest_table(tmp_path / "nope.csv")

