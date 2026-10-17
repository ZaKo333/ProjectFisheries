"""P0 主链路②：上传水质时序数据 → 风险告警 + 预测结果。

为什么这条必须端到端跑：模板 CSV 有三种表头（``units_en`` / ``zh`` / 简写），
前端解析用的列别名表和后端 ``fisheries.tables`` 是两份实现，规则引擎的阈值又是
第三份；再把读数喂给 ``/assess`` 才看得出「这份数据被理解成什么样」。
模板自带 ``expected_risk``——它是**模板作者写下的验收值**，不是我们照着接口输出改的。
"""

from __future__ import annotations

import pytest

from conftest import assert_level, level_index, post_assess, readings_from_template

pytestmark = pytest.mark.integration


def test_normal_summer_template_stays_normal(client, templates):
    """正常塘模板：解析 → 评估，等级必须是模板自称的「正常」。"""
    tpl, readings = readings_from_template(client, templates, "normal_summer")
    assert tpl["expected_risk"] == "正常"
    assert len(readings) == 24

    payload = post_assess(
        client,
        pond_id="IT-WATER-1",
        species=tpl["species"],
        readings=readings,
        mock_source="hash",
    )

    assert payload["risk_level"] == tpl["expected_risk"], payload["reason"]
    assert_level(payload["risk_level"])

    water = payload["water"]
    assert water["status"] == "ok", water
    assert water["risk_level"] == payload["risk_level"]  # 单模态：融合等级 = 水质等级
    assert water["risk_score"] <= 20.0, "正常塘的规则分不该高"
    assert water["species"] == tpl["species"]
    assert water["rule_hits"] == [], water["rule_hits"]

    # 「预测结果」：趋势方向 + 外推时域 + 逐时投影都要在
    assert water["trends"]["dissolved_oxygen"]["direction"] in {"上升", "下降", "平稳"}
    assert water["extrapolation_horizon_hours"] > 0
    assert water["projections"], "没给逐时投影"
    assert water["horizon_hours"] == 72


def test_hypoxia_template_raises_an_alert_with_a_projection(client, templates):
    """凌晨缺氧模板：等级升到「高危」，并给出「几小时后跌破危险线」的预测。"""
    tpl, readings = readings_from_template(client, templates, "dawn_hypoxia")
    assert tpl["expected_risk"] == "高危"

    payload = post_assess(
        client,
        pond_id="IT-WATER-2",
        species=tpl["species"],
        readings=readings,
        mock_source="hash",
    )

    assert payload["risk_level"] == "高危", payload["reason"]
    water = payload["water"]
    assert water["limiting_factor"] == "dissolved_oxygen"
    assert water["risk_score"] > 30.0

    # 预测结果：跌到危险线（鲤科档案是 1.5 mg/L）的预计时间，单位是小时
    eta = water["projected_do_alert_eta_hours"]
    assert eta is not None and 0 < eta < 24, eta
    assert water["projected_do_alert_threshold"] == 1.5

    codes = [e["code"] for e in payload["evidence"]]
    assert "DO_REDLINE_PROJECTED" in codes
    assert "SEV_DISSOLVED_OXYGEN" in codes
    # 单模态时融合层会补一条「等级下限」证据
    assert any(e["source"] == "fusion" for e in payload["evidence"])

    # 告警文案 + 可执行动作
    assert "高危" in payload["headline"]
    assert payload["advisories"], "判了高危却没给处置建议"
    for item in payload["advisories"]:
        for key in ("priority", "target", "action", "timing"):
            assert key in item


def test_the_same_water_is_judged_differently_for_shrimp_and_carp(client, templates):
    """同一个溶氧水平：对虾蟹正常、对鲤科已预警——品种档案真的在生效。"""
    _, readings = readings_from_template(client, templates, "normal_shrimp")

    shrimp = post_assess(
        client, pond_id="IT-WATER-3", species="虾蟹", readings=readings, mock_source="hash"
    )
    carp = post_assess(
        client, pond_id="IT-WATER-3", species="鲤科", readings=readings, mock_source="hash"
    )

    assert shrimp["risk_level"] == "正常", shrimp["reason"]
    assert level_index(carp["risk_level"]) > level_index(shrimp["risk_level"]), (
        "同一份水换品种结论没变，说明品种没进阈值计算"
    )
    assert carp["water"]["risk_score"] > shrimp["water"]["risk_score"]
    assert carp["water"]["species"] == "鲤科"
    assert shrimp["water"]["species"] == "虾蟹"
