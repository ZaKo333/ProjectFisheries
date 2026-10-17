"""内置水质模板（``fisheries.water_templates``）的单元测试。

这里最关键的一组断言是
:meth:`TestExpectedRisk.test_every_template_hits_its_expected_level` ——
每个模板的 ``expected_risk`` 不是文案描述，而是**真的把数据交给规则引擎跑一遍**
得到的结果。它同时钉住两件事：

* 模板数据本身没有因为改曲线而"降级"（本该高危的场景悄悄变成正常）；
* 规则引擎的阈值没有被无意改动（改了阈值，这里会先炸）。

``templates/`` 下的 CSV 与 README 都是**生成物**，所以另有一组断言专门检查
"磁盘上的字节 == 代码渲染出来的字节"，也就是
``scripts/build_water_templates.py --check`` 在测的那件事。这样即使没人记得跑
生成脚本，测试也会拦住"改了代码但没重跑生成"的提交。

还有一个容易被忽略的点：模板表头**故意不统一**（五种风格），
所以每个模板都要真的过一遍 ``ingest_rows`` 别名层 —— 表头风格写错在这里立刻暴露，
而不是等到现场演示时才发现整列被当成"无法识别的列"丢掉。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from fisheries.tables import MIN_POINTS, ingest_rows
from fisheries.tools.water_risk import MIN_RELIABLE_POINTS, SPECIES_CHOICES, water_risk_predict
from fisheries.water_templates import (
    HEADER_STYLES,
    TEMPLATE_INTERVAL_MINUTES,
    TEMPLATES_DIR,
    all_templates,
    catalog_markdown,
    catalog_rows,
    get_template,
    materialize,
    render_csv,
    template_keys,
    template_path,
    write_all,
)

#: 刻意写死：增删模板必须是一次有意识的改动，顺手改坏这里会立刻红。
EXPECTED_KEYS: tuple[str, ...] = (
    "normal_summer",
    "normal_shrimp",
    "dawn_hypoxia",
    "do_redline_crash",
    "ammonia_accumulation",
    "nitrite_spike",
    "ph_high_alkaline",
    "winter_ice",
    "low_do_warning",
    "temp_drop_after_rain",
    "ammonia_climbing",
    "spring_low_temp",
)

_KNOWN_LEVELS = ("正常", "预警", "高危")


def _predict(key: str) -> dict:
    template = get_template(key)
    return water_risk_predict(
        template.readings,
        pond_id=template.key,
        interval_minutes=template.interval_minutes,
        species=template.species,
    )


# --------------------------------------------------------------------------- #
# 目录
# --------------------------------------------------------------------------- #

class TestCatalog:
    def test_keys_are_exactly_the_documented_set(self) -> None:
        assert template_keys() == EXPECTED_KEYS

    def test_expected_risk_is_a_known_level(self) -> None:
        for template in all_templates():
            assert template.expected_risk in _KNOWN_LEVELS

    def test_every_species_has_a_threshold_profile(self) -> None:
        """模板里的品种必须是有阈值档案的，否则前端「自动切品种」会切到不存在的值。

        ``SPECIES_CHOICES`` 从 ``SPECIES_PROFILES`` 派生，所以这条同时钉住了
        「模板品种 ⊆ 档案品种」和「界面可选项 = 档案品种」两件事。
        """
        for template in all_templates():
            assert template.species in SPECIES_CHOICES

    def test_names_scenarios_and_focus_are_filled(self) -> None:
        """空标题会让下拉框出现一行空白，空说明会让用户不知道为什么用这个模板。"""
        for template in all_templates():
            assert template.name.strip()
            assert template.scenario.strip()
            assert template.focus.strip()

    def test_interval_matches_the_module_constant(self) -> None:
        for template in all_templates():
            assert template.interval_minutes == TEMPLATE_INTERVAL_MINUTES

    def test_every_template_clears_the_reliable_point_floor(self) -> None:
        """低于最低可靠点数会被判低置信度，演示时反而要额外解释，不如直接给够。"""
        for template in all_templates():
            assert template.n_points >= MIN_RELIABLE_POINTS
            assert template.n_points >= MIN_POINTS

    def test_header_styles_are_all_exercised(self) -> None:
        """五种表头风格各有模板覆盖，别名层的分支才不会没人走。"""
        used = {t.header_style for t in all_templates()}
        assert used == set(HEADER_STYLES)

    def test_timestamps_are_ascending(self) -> None:
        """倒序的时间戳会被趋势外推当成"在往回走"，结果毫无意义。"""
        for template in all_templates():
            stamps = [p.timestamp for p in template.readings]
            assert all(s is not None for s in stamps)
            assert stamps == sorted(stamps)

    def test_csv_filename_matches_the_key(self) -> None:
        for template in all_templates():
            assert template.csv_filename == f"{template.key}.csv"


# --------------------------------------------------------------------------- #
# 预期等级（本文件的核心）
# --------------------------------------------------------------------------- #

class TestExpectedRisk:
    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_every_template_hits_its_expected_level(self, key: str) -> None:
        out = _predict(key)
        assert out["status"] == "ok", out.get("error_message")
        template = get_template(key)
        assert out["risk_level"] == template.expected_risk, (
            f"{key} 预期 {template.expected_risk}，规则引擎实际判为 {out['risk_level']}"
            f"（得分 {out['risk_score']}，红线 {out['redline_hits']}，规则 {out['rule_hits']}）"
        )

    def test_covers_all_three_levels(self) -> None:
        """只测高危、或只测正常，都盖不住"等级边界"这件事。"""
        levels = {t.expected_risk for t in all_templates()}
        assert levels == set(_KNOWN_LEVELS)

    def test_normal_templates_score_zero_and_hit_nothing(self) -> None:
        """正常模板必须完全干净——有一点越界就说明曲线调到警戒带里去了。"""
        for template in all_templates():
            if template.expected_risk != "正常":
                continue
            out = _predict(template.key)
            assert out["risk_score"] == pytest.approx(0.0)
            assert out["redline_hits"] == []
            assert out["rule_hits"] == []
            assert out["exceeded_indicators"] == []
            # severity_map: 指标 -> 0.0 正常 / 0.5 警戒 / 1.0 危险
            assert set(out["severity_map"].values()) == {0.0}

    def test_danger_templates_always_say_why(self) -> None:
        """定了高危却给不出任何红线/规则，用户看到的就是一个没有理由的结论。"""
        for template in all_templates():
            if template.expected_risk != "高危":
                continue
            out = _predict(template.key)
            assert out["redline_hits"] or out["rule_hits"], template.key

    def test_danger_templates_cover_distinct_failure_modes(self) -> None:
        """八个高危模板若都栽在同一处，等于只测了一条规则。"""
        reasons: set[str] = set()
        for template in all_templates():
            if template.expected_risk != "高危":
                continue
            out = _predict(template.key)
            reasons.update(out["redline_hits"])
            reasons.update(out["rule_hits"])
        assert {
            "DO_BELOW_OPTIMAL",
            "NH3N_ABOVE_WARNING",
            "NO2N_ABOVE_WARNING",
            "PH_OUT_OF_OPTIMAL",
            "TEMP_OUT_OF_OPTIMAL",
        } <= reasons

    def test_shrimp_template_uses_the_shrimp_profile(self) -> None:
        """虾蟹塘的溶氧档位比鲤科低（4.0/2.5 vs 5.0/3.0），
        同一份数据按两个品种跑必须给出不同结论，否则"自动切品种"是白做的。"""
        template = get_template("normal_shrimp")
        as_shrimp = water_risk_predict(template.readings, species="虾蟹")
        as_carp = water_risk_predict(template.readings, species="鲤科")
        assert as_shrimp["risk_level"] == template.expected_risk
        lows = [p.dissolved_oxygen for p in template.readings if p.dissolved_oxygen is not None]
        assert min(lows) < 5.0, "虾蟹模板的溶氧应当低于鲤科的适宜下限，否则这条断言没有意义"
        assert as_carp["risk_score"] > as_shrimp["risk_score"]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #

class TestCsv:
    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_roundtrip_through_the_real_ingest_layer(self, key: str) -> None:
        """走一遍真实的别名层：零警告、零忽略列才算"这个表头真的认得出"。"""
        template = get_template(key)
        rows = list(csv.reader(io.StringIO(render_csv(template))))
        ingest = ingest_rows(rows[0], rows[1:], source=template.csv_filename)
        assert ingest.errors == []
        assert ingest.ok
        assert ingest.n_points == template.n_points
        assert ingest.warnings == []
        assert ingest.ignored_columns == []

    @pytest.mark.parametrize("key", EXPECTED_KEYS)
    def test_ingested_values_survive_the_roundtrip(self, key: str) -> None:
        """解析回来的数值要与模板定义一致，别让列错位悄悄过关。"""
        template = get_template(key)
        rows = list(csv.reader(io.StringIO(render_csv(template))))
        ingest = ingest_rows(rows[0], rows[1:], source=template.csv_filename)
        first = ingest.readings[0]
        origin = template.readings[0]
        for indicator in ("water_temp", "ph", "dissolved_oxygen", "ammonia_n", "nitrite_n"):
            assert first[indicator] == pytest.approx(getattr(origin, indicator), abs=1e-3)

    def test_uses_lf_and_no_bom_and_ends_with_a_newline(self) -> None:
        """``\\r\\n`` 会让同一份模板在不同系统上产出不同字节，``--check`` 就废了。"""
        for template in all_templates():
            raw = render_csv(template).encode("utf-8")
            assert b"\r\n" not in raw
            assert not raw.startswith(b"\xef\xbb\xbf")
            assert raw.endswith(b"\n")

    def test_render_is_deterministic(self) -> None:
        for template in all_templates():
            assert render_csv(template) == render_csv(get_template(template.key))

    def test_csv_is_sorted_by_time_so_tools_can_extrapolate(self) -> None:
        text = render_csv(get_template("dawn_hypoxia"))
        stamps = [line.split(",")[0] for line in text.splitlines()[1:]]
        assert stamps == sorted(stamps)


# --------------------------------------------------------------------------- #
# 落盘产物
# --------------------------------------------------------------------------- #

class TestFiles:
    def test_committed_csvs_match_the_code(self) -> None:
        """``scripts/build_water_templates.py --check`` 在测的同一件事。"""
        for template in all_templates():
            path = TEMPLATES_DIR / template.csv_filename
            assert path.exists(), f"{path} 不存在，请跑 scripts/build_water_templates.py"
            assert path.read_bytes() == render_csv(template).encode("utf-8"), (
                f"{path.name} 与 src/fisheries/water_templates.py 不一致，"
                f"请跑 scripts/build_water_templates.py 重新生成"
            )

    def test_committed_readme_matches_the_code(self) -> None:
        path = TEMPLATES_DIR / "README.md"
        assert path.exists()
        assert path.read_bytes() == catalog_markdown().encode("utf-8")

    def test_temp_example_csv_uses_the_same_newline_policy(self) -> None:
        """templates/ 与 examples/ 是同一类输入文件，换行符政策不能两套。"""
        example = TEMPLATES_DIR.parent / "examples" / "pond_demo_water.csv"
        raw = example.read_bytes()
        assert b"\r\n" not in raw
        assert not raw.startswith(b"\xef\xbb\xbf")

    def test_write_all_writes_every_template(self, tmp_path: Path) -> None:
        written = write_all(tmp_path)
        assert len(written) == len(EXPECTED_KEYS)
        for template in all_templates():
            assert (tmp_path / template.csv_filename).read_text(encoding="utf-8") == render_csv(
                template
            )

    def test_write_all_is_idempotent(self, tmp_path: Path) -> None:
        write_all(tmp_path)
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        write_all(tmp_path)
        after = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        assert before == after

    def test_materialize_fills_in_a_missing_file(self, tmp_path: Path) -> None:
        """只拷了 src/ 没带 templates/ 时，「用该模板填充」不该失败。"""
        path = materialize("dawn_hypoxia", tmp_path)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == render_csv(get_template("dawn_hypoxia"))

    def test_materialize_does_not_overwrite_an_existing_file(self, tmp_path: Path) -> None:
        """磁盘上内容过期该由 --check 报出来，静默覆盖只会掩盖"忘了重跑生成脚本"。"""
        path = template_path("dawn_hypoxia", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("故意留下的旧内容", encoding="utf-8")
        assert materialize("dawn_hypoxia", tmp_path) == path
        assert path.read_text(encoding="utf-8") == "故意留下的旧内容"


# --------------------------------------------------------------------------- #
# 取模板的失败路径
# --------------------------------------------------------------------------- #

class TestErrors:
    def test_unknown_key_lists_the_available_ones(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_template("no_such_template")
        message = str(excinfo.value.args[0])
        assert "no_such_template" in message
        for key in EXPECTED_KEYS:
            assert key in message

    def test_blank_key_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            get_template("")

    def test_key_is_stripped(self) -> None:
        assert get_template("  dawn_hypoxia  ").key == "dawn_hypoxia"

    def test_template_path_rejects_unknown_key(self) -> None:
        with pytest.raises(KeyError):
            template_path("no_such_template")


# --------------------------------------------------------------------------- #
# 目录输出
# --------------------------------------------------------------------------- #

class TestCatalogOutput:
    def test_rows_agree_with_templates(self) -> None:
        rows = catalog_rows()
        assert [r["key"] for r in rows] == list(EXPECTED_KEYS)
        for row, template in zip(rows, all_templates(), strict=True):
            assert row["name"] == template.name
            assert row["species"] == template.species
            assert row["expected_risk"] == template.expected_risk
            assert row["n_points"] == template.n_points
            assert row["csv_filename"] == template.csv_filename

    def test_rows_expose_a_start_and_end_stamp(self) -> None:
        for row in catalog_rows():
            assert row["start"] and row["end"]
            assert row["start"] < row["end"]

    def test_markdown_lists_every_template_and_every_style(self) -> None:
        text = catalog_markdown()
        for key in EXPECTED_KEYS:
            assert f"`{key}`" in text
        for style in HEADER_STYLES:
            assert f"`{style}`" in text

    def test_markdown_says_it_is_generated(self) -> None:
        """README 若被手工编辑，下次生成就悄悄覆盖掉——先说清楚。"""
        assert "不要手工编辑" in catalog_markdown()
