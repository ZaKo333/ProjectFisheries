"""现场演示页（``fisheries.webui``）的单元测试。

分四层测，跟模块的划分一一对应：

* ``TestHeaderMatching`` —— 真实数据集里的表头五花八门，列名识别错了
  整个报告就是错的，所以这一层测得最细；
* ``TestTableIngest`` —— 摄入质量（缺失、负值、时间轴、间隔）；
* ``TestWebSession`` + ``TestRender`` —— 端到端跑一次决策并排版；
* ``TestApp`` —— Gradio 页面能搭起来，且回调返回的输出个数与 ``outputs`` 一致。

全部离线：知识库关闭、LLM 用 mock、图片用临时生成的小 PNG。
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from fisheries.webui import render
from fisheries.webui.runner import WebRequest, WebSession
from fisheries.webui.table import (
    EXAMPLE_TABLE,
    TableError,
    ingest_rows,
    ingest_table,
    map_columns,
    normalize_header,
    parse_timestamp,
    to_float,
)

# 1x1 PNG。病害识别在模拟模式下只做后缀检查和内容哈希，
# 但给一个真能解码的图更稳——将来接上真实权重也不会因为图坏了而失败。
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB"
    "AAH/p2uEAAAAAElFTkSuQmCC"
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
    rows: list[list[str]] = []
    for i in range(n):
        rows.append([
            f"2026-10-13 {8 + i:02d}:00:00",
            "27.0",
            "8.10",
            f"{do + 0.01 * i:.2f}",
            f"{ammonia + 0.001 * i:.3f}",
            "0.05",
        ])
    return rows


def _readings(n: int = 6, *, do: float = 6.0, ammonia: float = 0.1):
    return ingest_rows(_HEADERS, _rows(n, do=do, ammonia=ammonia)).readings


@pytest.fixture(scope="module")
def png_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("img") / "fish.png"
    path.write_bytes(_PNG_1PX)
    return str(path)


@pytest.fixture
def session() -> WebSession:
    """不加载知识库的会话——测试不该依赖本机有没有建过向量库。"""
    return WebSession(use_knowledge=False)


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


# --------------------------------------------------------------------------- #
# 跑一次决策
# --------------------------------------------------------------------------- #

class TestWebSession:
    def test_no_input_is_refused_with_a_readable_message(
        self, session: WebSession
    ) -> None:
        result = session.run(WebRequest())
        assert not result.ok
        assert "水质表格" in result.error

    def test_fewer_than_three_points_is_not_water_input(
        self, session: WebSession
    ) -> None:
        """两点构不成趋势，等同于没给水质——不能拿它去调工具。"""
        req = WebRequest(readings=_readings(2))
        assert not req.has_water
        assert not session.run(req).ok

    def test_water_only_run(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings()))
        assert result.ok
        assert result.risk_level in {"正常", "预警", "高危"}
        assert result.water is not None
        assert result.water["status"] == "ok"
        assert result.vision is None
        assert result.n_readings == 6
        assert result.llm_backends

    def test_image_only_run(self, session: WebSession, png_file: str) -> None:
        result = session.run(WebRequest(image_path=png_file))
        assert result.ok
        assert result.vision is not None
        assert result.water is None
        assert result.n_readings == 0

    def test_both_modalities_run(self, session: WebSession, png_file: str) -> None:
        result = session.run(WebRequest(readings=_readings(), image_path=png_file))
        assert result.ok
        assert result.water is not None and result.vision is not None
        assert result.fusion.get("modalities") == ["water", "vision"]

    def test_high_risk_water_is_not_diluted_by_a_healthy_photo(
        self, session: WebSession
    ) -> None:
        """缺陷 A 的回归：水质高危 + 照片「健康」不能变成预警。

        这条走的是完整 Agent 链路，不是直接调 ``fuse_assessment``，
        所以它同时守着「网页拿到的报告等级」这一层。
        """
        readings = _readings(6, do=1.9, ammonia=0.72)
        ingest = ingest_rows(_HEADERS, [
            [f"2026-10-13 0{2 + i}:00:00", "27.0", "8.10",
             f"{1.9 + 0.001 * i:.3f}", f"{0.72 + 0.001 * i:.3f}", "0.35"]
            for i in range(6)
        ])
        assert not ingest.errors
        result = session.run(WebRequest(readings=ingest.readings))
        assert result.water is not None
        assert result.water["risk_level"] == "高危"
        assert result.risk_level == "高危"

    def test_knowledge_is_off_when_session_has_none(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings(), use_knowledge=True))
        assert result.ok
        assert not result.knowledge_block
        assert result.knowledge_summary == {}

    def test_request_fields_are_carried_through(self, session: WebSession) -> None:
        result = session.run(
            WebRequest(readings=_readings(), pond_id="塘口-7", species="虾蟹")
        )
        assert result.pond_id == "塘口-7"
        assert result.species == "虾蟹"
        assert result.water is not None
        assert result.water["species"] == "虾蟹"

    def test_actions_are_plain_dicts_not_models(self, session: WebSession) -> None:
        """``report`` 里已经是 dump 过的 dict；再调 ``model_dump()`` 会 AttributeError。"""
        result = session.run(WebRequest(readings=_readings(do=4.2)))
        assert result.ok
        for item in result.water_actions + result.disease_actions + result.advisories:
            assert isinstance(item, dict)
            assert set(item) >= {"priority", "target", "action"}

    def test_alerts_are_plain_dicts(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings(do=4.2, ammonia=0.4)))
        assert result.ok
        for alert in result.alerts:
            assert isinstance(alert, dict)
            assert set(alert) >= {"code", "level", "title", "detail"}


# --------------------------------------------------------------------------- #
# 排版
# --------------------------------------------------------------------------- #

class TestRender:
    @pytest.fixture(scope="class")
    def warned(self) -> object:
        session = WebSession(use_knowledge=False)
        ingest = ingest_rows(_HEADERS, _rows(6, do=4.2, ammonia=0.4))
        result = session.run(WebRequest(readings=ingest.readings, pond_id="塘口-1"))
        result.ingest = ingest
        return result

    def test_summary_shows_level_and_score(self, warned: object) -> None:
        text = render.render_summary_markdown(warned)
        assert warned.risk_level in text
        assert "融合分" in text
        assert "塘口-1" in text

    def test_alerts_render_as_html_cards(self, warned: object) -> None:
        html = render.render_alerts(warned)
        assert "<div" in html
        if warned.alerts:
            assert warned.alerts[0]["title"] in html or "告警" in html

    def test_alerts_without_any_alert_says_so(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings()))
        html = render.render_alerts(result)
        assert "没有产生告警" in html or "告警" in html

    def test_report_separates_water_and_disease_actions(self, warned: object) -> None:
        text = render.render_report(warned)
        assert "水质调控" in text
        assert "病害处置" in text

    def test_data_quality_without_ingest(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings()))
        assert "没有上传水质表格" in render.render_data_quality(result)

    def test_data_quality_with_ingest(self, warned: object) -> None:
        text = render.render_data_quality(warned)
        assert "读入" in text
        assert "水质指标现状" in text

    def test_vision_without_image(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings()))
        assert "没有上传鱼体图片" in render.render_vision(result)

    def test_vision_with_image(self, session: WebSession, png_file: str) -> None:
        result = session.run(WebRequest(image_path=png_file))
        text = render.render_vision(result)
        assert "鱼体图片识别" in text
        assert "降级模式" in text  # 没装真实权重时必须自曝

    def test_fusion_shows_floor_explanation(self, warned: object) -> None:
        text = render.render_fusion(warned)
        assert "融合依据" in text
        assert "等级下限" in text

    def test_fusion_lists_hit_floors(self, session: WebSession) -> None:
        result = session.run(WebRequest(readings=_readings(do=1.9, ammonia=0.72)))
        result.fusion["level_floors"] = ["WATER_LEVEL:高危"]
        text = render.render_fusion(result)
        assert "WATER_LEVEL:高危" in text

    def test_trace_lists_steps(self, warned: object) -> None:
        text = render.render_trace(warned)
        assert "启动 Agent 调度" in text

    def test_status_mentions_elapsed_and_backends(self, warned: object) -> None:
        text = render.render_status(warned)
        assert "耗时" in text
        assert "调度模型" in text

    def test_knowledge_section_without_knowledge(self, warned: object) -> None:
        assert "知识库引用" in render.render_knowledge(warned)

    def test_error_block_escapes_html(self) -> None:
        html = render.render_error("<script>alert(1)</script>")
        assert "&lt;script&gt;" in html
        assert "<script>" not in html

    def test_all_renderers_survive_a_failed_result(self, session: WebSession) -> None:
        """``ok=False`` 时每个渲染器都必须安静地返回空串，不能抛异常。

        页面在同一个回调里会依次调它们，任何一个抛异常都会让整次点击失败。
        """
        failed = session.run(WebRequest())
        for fn in (
            render.render_summary_markdown,
            render.render_alerts,
            render.render_report,
            render.render_data_quality,
            render.render_vision,
            render.render_fusion,
            render.render_knowledge,
            render.render_trace,
            render.render_status,
        ):
            assert fn(failed) == ""


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #

class TestApp:
    def test_build_app_and_callback_output_count(
        self, session: WebSession, png_file: str
    ) -> None:
        """回调返回的元组长度必须等于 ``outputs`` 的长度。

        差一个 Gradio 会在点击时报 "The number of outputs does not match"，
        而这种错正好只在现场演示时暴露。
        """
        pytest.importorskip("gradio")
        from fisheries.webui.app import _run, build_app

        app = build_app(session=session)
        assert app is not None

        good = _run(
            str(EXAMPLE_TABLE), "", png_file, "塘口-1", "鲤科", "hash", True, 48, session
        )
        assert len(good) == 8
        assert "风险等级" in good[0]

        bad = _run(None, "", None, "p", "鲤科", "hash", True, 48, session)
        assert len(bad) == 8
        assert "运行失败" in bad[0]

    def test_pasted_table_path(
        self, session: WebSession, png_file: str
    ) -> None:
        """粘贴通道：现场文件选择器不好用时的备用入口。"""
        pytest.importorskip("gradio")
        from fisheries.webui.app import _run

        pasted = "timestamp,water_temp,ph,dissolved_oxygen,ammonia_n\n" + "\n".join(
            f"2026-10-13 {8 + i:02d}:00:00,27.0,8.10,{4.2 + 0.01 * i:.2f},0.40"
            for i in range(6)
        )
        out = _run(None, pasted, None, "p", "鲤科", "hash", False, 48, session)
        assert len(out) == 8
        assert "风险等级" in out[0]

    def test_bad_suffix_in_file_slot_gets_a_hint(
        self, session: WebSession, png_file: str
    ) -> None:
        """把图片误传进表格槽时，要明确提示换到图片槽。"""
        pytest.importorskip("gradio")
        from fisheries.webui.app import _run

        out = _run(png_file, "", None, "p", "鲤科", "hash", False, 48, session)
        assert len(out) == 8
        assert "鱼体图片" in out[0]


# --------------------------------------------------------------------------- #
# 内置水质模板
# --------------------------------------------------------------------------- #

class TestTemplates:
    def test_choices_cover_every_template(self) -> None:
        pytest.importorskip("gradio")
        from fisheries.water_templates import all_templates
        from fisheries.webui.app import _template_choices

        choices = _template_choices()
        assert [value for _label, value in choices] == [t.key for t in all_templates()]
        # 下拉框显示名字而不是 key：现场没人想读 normal_summer。
        assert all(label and value for label, value in choices)

    def test_fill_returns_the_file_the_species_and_a_note(self) -> None:
        pytest.importorskip("gradio")
        from fisheries.water_templates import get_template, template_path
        from fisheries.webui.app import _template_fill

        template = get_template("dawn_hypoxia")
        path, _pasted, species, note = _template_fill("dawn_hypoxia")
        assert Path(path) == template_path("dawn_hypoxia")
        assert Path(path).exists(), "「用该模板填充」不能给出一条不存在的路径"
        assert species == template.species
        assert template.name in note
        assert template.expected_risk in note

    def test_fill_clears_the_paste_box(self) -> None:
        """粘贴框不清空，用户先粘过一段文本再点模板，跑的还是那段旧文本。

        ``_load_table`` 里粘贴优先于文件，所以这不是"顺手清一下"，
        而是模板能不能生效的前提。
        """
        pytest.importorskip("gradio")
        from fisheries.webui.app import _template_fill

        _path, pasted, _species, _note = _template_fill("normal_summer")
        assert pasted == ""

    def test_paste_still_wins_over_the_file(self, session: WebSession) -> None:
        """反过来钉住上面那条规则本身：粘贴优先是既有行为，不能被模板改掉。"""
        pytest.importorskip("gradio")
        from fisheries.water_templates import template_path
        from fisheries.webui.app import _run

        pasted = "timestamp,water_temp,ph,dissolved_oxygen,ammonia_n\n" + "\n".join(
            f"2026-10-13 {8 + i:02d}:00:00,27.0,8.10,{4.2 + 0.01 * i:.2f},0.40"
            for i in range(6)
        )
        out = _run(str(template_path("dawn_hypoxia")), pasted, None, "p", "鲤科", "hash", False, 48, session)
        assert len(out) == 8
        # 用粘贴的 6 个好点跑，就不该出现模板那份数据才有的高危结论。
        assert "高危" not in out[0]

    def test_selected_template_reaches_the_report(self, session: WebSession) -> None:
        """端到端：模板文件 → 规则引擎 → 报告里出现该模板的预期等级。"""
        pytest.importorskip("gradio")
        from fisheries.water_templates import get_template, template_path
        from fisheries.webui.app import _run

        template = get_template("dawn_hypoxia")
        out = _run(str(template_path("dawn_hypoxia")), "", None, "p", "鲤科", "hash", False, 48, session)
        assert len(out) == 8
        assert "风险等级" in out[0]
        assert template.expected_risk in out[0]

    def test_fill_without_a_selection_asks_for_one(self) -> None:
        pytest.importorskip("gradio")
        from fisheries.webui.app import _template_fill

        note = _template_fill(None)[-1]
        assert "选一个模板" in note

    def test_fill_with_an_unknown_key_explains_why(self) -> None:
        pytest.importorskip("gradio")
        from fisheries.webui.app import _template_fill

        note = _template_fill("no_such_template")[-1]
        assert note.startswith("❌")
        assert "no_such_template" in note
        # 错误信息里要列出可用模板，用户才知道下一步点哪个。
        assert "dawn_hypoxia" in note
