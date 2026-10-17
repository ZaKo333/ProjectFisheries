"""P0 主链路④：双模态融合——同一口塘「缺氧 + 烂鳃」的结论要重于单模态。

融合规则最容易出「乘积调不准」的错：单看水质是「预警」、单看图像是「高危」，
合起来既不能塌回「预警」（漏报），也不该凭空变成 100（吓人）。
单元测试里的融合函数是隔离的，只有真的把两条模态喂进 ``/assess`` 才看得出接线对不对。
"""

from __future__ import annotations

import pytest
from conftest import level_index, post_assess, readings_from_template

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def hypoxia_readings(client, templates):
    tpl, readings = readings_from_template(client, templates, "dawn_hypoxia")
    return tpl, readings


def test_both_modalities_together_score_higher_than_water_alone(
    client, gill_image, hypoxia_readings
):
    """水质单模态 vs 图像 + 水质：融合等级不降、分数更高、理由说明协同。"""
    tpl, readings = hypoxia_readings

    water_only = post_assess(
        client,
        pond_id="IT-FUSION-1",
        species=tpl["species"],
        readings=readings,
        mock_source="hash",
    )
    fused = post_assess(
        client,
        pond_id="IT-FUSION-1",
        species=tpl["species"],
        readings=readings,
        image_path=str(gill_image),
        mock_source="hash",
    )

    # 融合等级不低于任一单模态
    assert level_index(fused["risk_level"]) >= level_index(water_only["risk_level"])
    assert fused["risk_level"] == "高危", fused["reason"]

    # 双模态的分数要显著高于只喂水质，否则融合等于没做
    assert float(fused["fused_score"]) > float(water_only["fused_score"]) + 10.0, (
        water_only["fused_score"],
        fused["fused_score"],
    )
    assert float(fused["fused_score"]) <= 100.0

    # 两条模态各自都留了证据，融合层还要单独记一笔
    sources = {e["source"] for e in fused["evidence"]}
    assert {"water", "vision", "fusion"} <= sources, sources
    assert any(e["source"] == "fusion" for e in fused["evidence"])

    # 病害与水质的处置动作都要给：只有一边说明另一边的证据被吞了
    report = fused["report"]
    assert report["water_actions"], "双模态却没给水质处置动作"
    assert report["disease_actions"], "双模态却没给病害处置动作"
    assert "协同" in fused["reason"] or "叠加" in fused["reason"], fused["reason"]


def test_synergy_evidence_codes_are_reported(client, gill_image, hypoxia_readings):
    """「缺氧 + 烂鳃」这种组合本身要有名字（可追溯，而不是只体现在分数上）。"""
    tpl, readings = hypoxia_readings
    fused = post_assess(
        client,
        pond_id="IT-FUSION-2",
        species=tpl["species"],
        readings=readings,
        image_path=str(gill_image),
        mock_source="hash",
    )

    codes = [e["code"] for e in fused["evidence"]]
    assert "SYN_LOW_DO_WITH_GILL" in codes, codes
    # 水质侧的证据也还在，没被融合层覆盖
    assert "DO_REDLINE_PROJECTED" in codes

    for item in fused["advisories"]:
        for key in ("priority", "target", "action", "timing"):
            assert key in item, f"融合后的处置建议缺字段 {key}"

    # 观测窗口与话术：双模态要给出比单模态更短的复核间隔（可选但应存在）
    assert fused["headline"]
    assert fused["observation_window"]
