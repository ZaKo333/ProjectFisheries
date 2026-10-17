"""Agent 层（融合 M4 + 调度 M6）单元测试。

覆盖重点：
    1. 融合层的**确定性**与**单模态权重归一化**（最容易写错、后果最严重的一处）
    2. 四条协同规则的独立触发条件
    3. 模态失败必须显式上报（静默丢模态是最危险的失败模式）
    4. 调度层的护栏补调、防重复调用、工具异常不扩散
    5. LLM 配置的优先级与离线降级
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from fisheries import datasets as _datasets
from fisheries.agent.fusion import (
    DISEASE_MIN_CONF_FOR_WARNING,
    DISEASE_MIN_SEVERITY_FOR_WARNING,
    SCORE_CRITICAL,
    SCORE_WARNING,
    SEVERE_DISEASE_MIN_CONF,
    SEVERE_DISEASE_WEIGHT,
    SYNERGY_RULES,
    fuse_assessment,
)
from fisheries.agent.llm import (
    ChatResponse,
    LLMUnavailableError,
    MockLLMClient,
    OpenAICompatibleClient,
    ToolCall,
    load_llm_config,
)
from fisheries.agent.orchestrator import (
    KNOWLEDGE_DISABLE_ENV,
    MAX_ROUNDS,
    AgentInput,
    AquacultureAgent,
)
from fisheries.agent.prompts import (
    ADVISOR_SYSTEM_PROMPT,
    DOMAIN_CONSTRAINTS,
    build_advisor_user_message,
    build_confidence_facts,
    format_confidence_block,
    render_confidence_note,
)


# 单元测试默认关闭 RAG 检索。
# 理由：本文件测的是**调度与融合逻辑**，不该依赖 `knowledge/chroma` 是否已构建，
# 也不该为了跑 69 个毫秒级用例去加载 90 MB 的向量模型。
# RAG 本身的行为由 tests/unit/test_rag.py 单独覆盖。
@pytest.fixture(autouse=True, scope="module")
def _disable_knowledge():
    import os

    old = os.environ.get(KNOWLEDGE_DISABLE_ENV)
    os.environ[KNOWLEDGE_DISABLE_ENV] = "1"
    yield
    if old is None:
        os.environ.pop(KNOWLEDGE_DISABLE_ENV, None)
    else:
        os.environ[KNOWLEDGE_DISABLE_ENV] = old

# --------------------------------------------------------------------------- #
# 构造器
# --------------------------------------------------------------------------- #

GOOD_READINGS = [
    {"timestamp": f"2026-05-11T{i:02d}:00:00", "water_temp": 26.0, "ph": 7.6,
     "dissolved_oxygen": 6.0, "ammonia_n": 0.10, "nitrite_n": 0.03}
    for i in range(24)
]


def W(**over: Any) -> dict[str, Any]:
    """构造一个 water_risk_predict 的返回值。"""
    base: dict[str, Any] = {
        "tool": "water_risk_predict", "status": "ok", "pond_id": "P01",
        "risk_level": "正常", "risk_score": 0.0,
        "severity_map": {}, "redline_hits": [], "rule_hits": [],
        "trends": {"water_temp": {"current": 26.0, "label": "水温", "unit": "℃",
                                  "direction": "平稳"},
                   "dissolved_oxygen": {"current": 6.0, "label": "溶氧", "unit": "mg/L",
                                        "direction": "平稳"}},
        "projected_do_alert_eta_hours": None,
        "degraded_mode": False,
    }
    base.update(over)
    return base


def V(**over: Any) -> dict[str, Any]:
    """构造一个 fish_disease_detect 的返回值。"""
    base: dict[str, Any] = {
        "tool": "fish_disease_detect", "status": "ok", "image_path": "f.jpg",
        "disease_type": "健康", "disease_group": "健康", "confidence": 0.9,
        "is_healthy": True, "severity_weight": 0.0, "is_low_confidence": False,
        "raw_class": "Healthy Fish", "model_version": "mock-v0.1",
        "degraded_mode": False,
    }
    base.update(over)
    return base


def bacterial_gill(conf: float = 0.9) -> dict[str, Any]:
    return V(disease_type="烂鳃", disease_group="细菌性", confidence=conf,
             is_healthy=False, severity_weight=0.85, raw_class="Bacterial gill disease")


def septicemia(conf: float = 0.9) -> dict[str, Any]:
    return V(disease_type="出血", disease_group="细菌性", confidence=conf,
             is_healthy=False, severity_weight=0.90,
             raw_class="Bacterial diseases - Aeromoniasis")


def sapro(conf: float = 0.85) -> dict[str, Any]:
    return V(disease_type="水霉", disease_group="真菌性", confidence=conf,
             is_healthy=False, severity_weight=0.70,
             raw_class="Fungal diseases Saprolegniasis")


def W_cold(**over: Any) -> dict[str, Any]:
    """22℃ 低温水质——水霉高发区间，用于触发 SYN_SAPRO_ON_COLD。

    注意 `W()` 默认水温 26℃，**不在** 15–25℃ 窗口内，用它测这条规则会失败。
    """
    over.setdefault("trends", {"water_temp": {"current": 22.0, "label": "水温",
                                              "unit": "℃", "direction": "平稳"},
                               "dissolved_oxygen": {"current": 6.0, "label": "溶氧",
                                                    "unit": "mg/L", "direction": "平稳"}})
    return W(**over)


# --------------------------------------------------------------------------- #
# 一、融合层：权重与归一化
# --------------------------------------------------------------------------- #

class TestFusionWeights:

    def test_water_only_weight_normalised(self):
        """只有水质数据时权重必须是 1.0，否则单模态永远进不了高危档。"""
        r = fuse_assessment(W(risk_level="预警", risk_score=60.0), None)
        assert r.effective_weights == {"water": 1.0}
        assert r.modalities == ["water"]
        assert r.fused_score == pytest.approx(60.0, abs=0.01)

    def test_vision_only_weight_normalised(self):
        """只有图片时同理——严重病害单靠图像就该能判高危。"""
        r = fuse_assessment(None, septicemia(0.95))
        assert r.effective_weights == {"vision": 1.0}
        assert r.modalities == ["vision"]
        # 0.90 × 0.95 = 0.855 → 85.5 分
        assert r.fused_score == pytest.approx(85.5, abs=0.6)

    def test_both_modalities_use_configured_weights(self):
        r = fuse_assessment(W(risk_level="预警", risk_score=50.0), bacterial_gill(0.8))
        assert r.effective_weights == {"water": 0.6, "vision": 0.4}
        # 0.6×0.50 + 0.4×(0.85×0.8) = 0.30 + 0.272 = 0.572 → 57.2，再加细菌协同 12.0
        assert r.weighted_score == pytest.approx(57.2 + r.synergy_score, abs=0.6)
        # 同一份输入同时命中了 SEVERE_DISEASE（0.85 严重度 × 0.8 置信度），
        # 等级被下限抬到高危，fused_score 必须跟着抬到档线，否则报告自相矛盾。
        assert r.risk_level == "高危"
        assert r.fused_score == pytest.approx(70.0)
        assert r.weighted_score < r.fused_score

    def test_vision_score_is_confidence_weighted(self):
        """同样检出烂鳃，置信度 0.9 与 0.3 不能给同一个分。"""
        hi = fuse_assessment(None, bacterial_gill(0.9))
        lo = fuse_assessment(None, bacterial_gill(0.3))
        assert hi.fused_score > lo.fused_score
        assert lo.fused_score == pytest.approx(0.85 * 0.3 * 100, abs=0.5)

    def test_healthy_vision_scores_zero(self):
        r = fuse_assessment(W(risk_score=0.0), V())
        assert r.vision_score == 0.0
        assert r.fused_score == 0.0
        assert r.risk_level == "正常"

    def test_no_modality_gives_zero_and_explains(self):
        r = fuse_assessment(None, None)
        assert r.modalities == []
        assert r.fused_score == 0.0
        assert "无法定级" in r.reason


# --------------------------------------------------------------------------- #
# 二、融合层：协同规则
# --------------------------------------------------------------------------- #

class TestSynergyRules:

    def test_no_synergy_with_single_modality(self):
        """协同是"两个模态之间"的关系，单模态谈不上协同。"""
        r = fuse_assessment(W(risk_level="高危", risk_score=90.0), None)
        assert r.synergy_score == 0.0
        assert r.synergy_hits == []

    def test_bacterial_on_warning(self):
        r = fuse_assessment(W(risk_level="预警", risk_score=45.0), bacterial_gill())
        assert "SYN_BACTERIAL_ON_WARNING" in r.synergy_hits

    def test_bacterial_synergy_not_triggered_when_water_normal(self):
        r = fuse_assessment(W(risk_level="正常", risk_score=10.0), bacterial_gill())
        assert "SYN_BACTERIAL_ON_WARNING" not in r.synergy_hits

    def test_bacterial_synergy_not_triggered_by_fungal(self):
        """水霉是真菌，不该吃到细菌协同的加成。"""
        r = fuse_assessment(W(risk_level="预警", risk_score=45.0), sapro())
        assert "SYN_BACTERIAL_ON_WARNING" not in r.synergy_hits

    def test_nitrogen_with_any_disease(self):
        r = fuse_assessment(W(severity_map={"ammonia_n": 0.5}), sapro())
        assert "SYN_NITROGEN_WITH_DISEASE" in r.synergy_hits

    def test_nitrogen_synergy_not_triggered_for_healthy_fish(self):
        r = fuse_assessment(W(severity_map={"nitrite_n": 0.5}), V())
        assert "SYN_NITROGEN_WITH_DISEASE" not in r.synergy_hits

    def test_low_do_with_gill_disease(self):
        r = fuse_assessment(W(severity_map={"dissolved_oxygen": 1.0}), bacterial_gill())
        assert "SYN_LOW_DO_WITH_GILL" in r.synergy_hits

    def test_low_do_synergy_also_fires_on_projected_do_drop(self):
        """溶氧当前还行、但预测要跌破，同样该算协同。"""
        r = fuse_assessment(
            W(severity_map={}, projected_do_alert_eta_hours=5.0), bacterial_gill())
        assert "SYN_LOW_DO_WITH_GILL" in r.synergy_hits

    def test_sapro_on_cold_water(self):
        r = fuse_assessment(W_cold(), sapro())
        assert "SYN_SAPRO_ON_COLD" in r.synergy_hits

    def test_sapro_not_triggered_when_warm(self):
        """水霉在 15–25℃ 高发；30℃ 时不该给这个加成。"""
        w = W(trends={"water_temp": {"current": 30.0, "label": "水温", "unit": "℃"}})
        r = fuse_assessment(w, sapro())
        assert "SYN_SAPRO_ON_COLD" not in r.synergy_hits

    def test_synergy_bonus_value_matches_table(self):
        """22℃ + 水霉 → 水霉协同；同时氨氮超阈 → 氮胁迫协同。两条都要命中。"""
        r = fuse_assessment(W_cold(severity_map={"ammonia_n": 0.5}), sapro())
        assert set(r.synergy_hits) == {"SYN_SAPRO_ON_COLD", "SYN_NITROGEN_WITH_DISEASE"}
        expected = SYNERGY_RULES["SYN_SAPRO_ON_COLD"][1] + \
            SYNERGY_RULES["SYN_NITROGEN_WITH_DISEASE"][1]
        assert r.synergy_score == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 三、融合层：定级与红线
# --------------------------------------------------------------------------- #

class TestFusionLevel:

    def test_redline_forces_critical_even_with_low_score(self):
        r = fuse_assessment(W(risk_level="预警", risk_score=20.0,
                              redline_hits=["NH3_REDLINE"]), None)
        assert r.risk_level == "高危"
        assert any(a.code == "NH3_REDLINE" for a in r.alerts)

    def test_severe_disease_alone_forces_critical(self):
        r = fuse_assessment(None, septicemia(0.9))
        assert r.risk_level == "高危"
        assert any(a.code == "SEVERE_DISEASE" for a in r.alerts)

    def test_severe_disease_low_confidence_does_not_force_critical(self):
        """0.3 置信度的败血症不该直接拉满告警——那会把人训练到忽略告警。"""
        r = fuse_assessment(None, septicemia(0.3))
        assert not any(a.code == "SEVERE_DISEASE" for a in r.alerts)

    def test_mild_disease_does_not_force_critical(self):
        r = fuse_assessment(None, sapro(0.95))
        assert not any(a.code == "SEVERE_DISEASE" for a in r.alerts)

    @pytest.mark.parametrize("score,expected", [
        (0.0, "正常"), (39.9, "正常"), (40.0, "预警"),
        (69.9, "预警"), (70.0, "高危"), (100.0, "高危"),
    ])
    def test_score_thresholds(self, score, expected):
        r = fuse_assessment(W(risk_level="正常", risk_score=score), None)
        assert r.risk_level == expected

    def test_fused_score_is_capped_at_100(self):
        r = fuse_assessment(W(risk_level="高危", risk_score=100.0,
                              severity_map={"ammonia_n": 1.0}), septicemia(1.0))
        assert r.fused_score == 100.0

    def test_confidence_thresholds_are_sane(self):
        assert 0.0 < SEVERE_DISEASE_MIN_CONF <= 1.0
        assert 0.0 < SEVERE_DISEASE_WEIGHT <= 1.0


# --------------------------------------------------------------------------- #
# 三之二、融合层：等级下限（单调性 —— 融合不得低于任一单模态的判定）
# --------------------------------------------------------------------------- #

class TestLevelFloors:
    """加权平均会把"一个模态没发现问题"当成"问题减半"的证据，这是错的。

    下面三组用例分别对应实测中发现的三处缺陷，输入取自 6 组边界演示案例
    （跑 scripts/demo_cases.py 能原样重现）：

        A. 水质独立判高危（100 分）+ 健康鱼 → 曾被稀释成 60 分/预警
        B. 等级被红线上调为高危，fused_score 却仍是 12 分 → 报告自相矛盾
        C. 水质满分 + 确诊水霉 0.62 / 烂鳃 0.42 → 曾被判"正常，照常管理"
    """

    # ---- 缺陷 A：健康照片把水质高危稀释成预警（P0，单调性违反） ----

    def test_healthy_photo_cannot_downgrade_critical_water(self):
        """水质工具独立判高危/100 分时，一张"鱼看着还行"的照片不得把结论压回预警。"""
        w = W(risk_level="高危", risk_score=100.0, redline_hits=[],
              severity_map={"dissolved_oxygen": 1.0, "ammonia_n": 1.0, "nitrite_n": 1.0})
        r = fuse_assessment(w, V())          # V() 是健康鱼
        assert r.risk_level == "高危"
        assert "WATER_LEVEL:高危" in r.level_floors
        # 没有这条下限时这里正好是 60.0（0.6 × 1.0 × 100），落在预警档
        assert r.weighted_score == pytest.approx(60.0, abs=0.01)
        assert r.fused_score >= SCORE_CRITICAL
        assert any(e.code == "LEVEL_FLOOR" for e in r.evidence)
        assert any(a.code == "LEVEL_FLOOR" for a in r.alerts)

    def test_warning_water_floors_to_warning(self):
        """水质预警 + 健康鱼同样不该被拉回正常。"""
        r = fuse_assessment(W(risk_level="预警", risk_score=45.0), V())
        assert r.risk_level == "预警"
        assert "WATER_LEVEL:预警" in r.level_floors

    def test_water_level_floor_does_not_apply_to_failed_water(self):
        """水质工具失败时没有"水质档位"可继承，不能凭错误码抬等级。"""
        r = fuse_assessment({"status": "error", "error_code": "INVALID_INPUT"}, V())
        assert r.level_floors == []
        assert r.risk_level == "正常"

    def test_normal_water_healthy_fish_is_untouched(self):
        """下限只抬不降：一切正常时不得凭空产生等级、分数或告警。"""
        r = fuse_assessment(W(), V())
        assert r.level_floors == []
        assert r.score_floor is None
        assert r.fused_score == r.weighted_score == 0.0
        assert r.risk_level == "正常"

    # ---- 缺陷 B：等级被抬升而分数没抬，报告自相矛盾（P1，一致性） ----

    def test_severe_disease_score_is_consistent_with_level(self):
        """纯公式值本来就够高危时，不该被误标成"被下限抬升过"。"""
        r = fuse_assessment(None, septicemia(0.9))
        assert r.risk_level == "高危"
        assert r.weighted_score == pytest.approx(81.0, abs=0.6)
        assert r.fused_score == r.weighted_score
        assert r.score_floor is None, "81.0 已高于 70.0 档线，不该再抬"

    def test_score_floor_is_recorded_when_it_actually_raised(self):
        """红线把 20 分抬到高危时必须留下痕迹，且原始分仍可审计。"""
        r = fuse_assessment(W(risk_level="预警", risk_score=20.0,
                              redline_hits=["NH3_REDLINE"]), None)
        assert r.risk_level == "高危"
        # 单模态水质的权重被归一化到 1.0，所以纯公式值就是 20.0
        assert r.weighted_score == pytest.approx(20.0, abs=0.01)
        assert r.score_floor == pytest.approx(SCORE_CRITICAL)
        assert r.fused_score == pytest.approx(SCORE_CRITICAL)
        assert any(a.code == "LEVEL_FLOOR" for a in r.alerts)
        assert "REDLINE" in r.level_floors

    def test_fused_score_never_below_weighted_score_or_conflicts_with_level(self):
        """不变式：fused_score ≥ weighted_score，且等级与 fused_score 同档。"""
        cases = [
            (W(risk_level="高危", risk_score=100.0), V()),
            (W(risk_level="预警", risk_score=45.0), bacterial_gill(0.8)),
            (None, sapro(0.62)),
            (W(), bacterial_gill(0.42)),
            (None, septicemia(0.3)),
            (None, None),
        ]
        bands = {"正常": 0.0, "预警": SCORE_WARNING, "高危": SCORE_CRITICAL}
        for w, v in cases:
            r = fuse_assessment(w, v)
            assert r.fused_score >= r.weighted_score - 1e-9
            assert r.fused_score >= bands[r.risk_level] - 1e-9, r.reason

    # ---- 缺陷 C：水质全优时确认病害被 0.4 权重压到"正常"（P1，漏报） ----

    def test_confirmed_disease_with_excellent_water_is_at_least_warning(self):
        """水质满分时图像只占 0.4 权重，确诊水霉/烂鳃会被压到 40 分线以下。"""
        for v in (sapro(0.62), bacterial_gill(0.42)):
            r = fuse_assessment(W(), v)
            assert r.risk_level == "预警", f"{v['disease_type']} 被判成了 {r.risk_level}"
            assert "DISEASE_DETECTED" in r.level_floors
            # 加权公式值确实低于预警线——是下限把它抬上来的
            assert r.weighted_score < SCORE_WARNING
            assert r.fused_score == pytest.approx(SCORE_WARNING)

    def test_noise_confidence_disease_does_not_raise_level(self):
        """0.40 以下是纯噪声区间，不能被抬进预警——否则告警会被训成背景音。"""
        r = fuse_assessment(W(), bacterial_gill(DISEASE_MIN_CONF_FOR_WARNING - 0.01))
        assert "DISEASE_DETECTED" not in r.level_floors
        assert r.risk_level == "正常"

    def test_low_severity_disease_does_not_raise_level(self):
        """轻微病变（严重度低于门槛）不该触发下限。"""
        mild = V(disease_type="赤皮", disease_group="细菌性", confidence=0.95,
                 is_healthy=False, raw_class="Bacterial Red disease",
                 severity_weight=DISEASE_MIN_SEVERITY_FOR_WARNING - 0.05)
        assert "DISEASE_DETECTED" not in fuse_assessment(W(), mild).level_floors
        r = fuse_assessment(W(risk_level="正常", risk_score=5.0), mild)
        assert r.risk_level == "正常"

    def test_healthy_fish_never_triggers_disease_floor(self):
        assert "DISEASE_DETECTED" not in fuse_assessment(W(), V()).level_floors

    def test_floor_thresholds_are_sane(self):
        assert 0.0 < DISEASE_MIN_SEVERITY_FOR_WARNING <= 1.0
        assert 0.0 < DISEASE_MIN_CONF_FOR_WARNING <= 1.0
        # 病害下限必须比高致病性红线更宽松，否则两条规则会退化成同一条
        assert DISEASE_MIN_CONF_FOR_WARNING < SEVERE_DISEASE_MIN_CONF
        assert DISEASE_MIN_SEVERITY_FOR_WARNING < SEVERE_DISEASE_WEIGHT


# --------------------------------------------------------------------------- #
# 四、融合层：模态失败必须显式上报
# --------------------------------------------------------------------------- #

class TestFailedModality:

    def test_failed_vision_is_recorded(self):
        """用户传了图片但工具报错——绝不能静默当成"没有图片"。"""
        r = fuse_assessment(W(), {"status": "error", "error_code": "IMAGE_NOT_FOUND"})
        assert r.failed_modalities == {"vision": "IMAGE_NOT_FOUND"}
        assert any(a.code == "MODALITY_UNAVAILABLE" for a in r.alerts)
        assert "鱼体图片分析失败" in r.reason
        assert r.modalities == ["water"]

    def test_failed_water_is_recorded(self):
        r = fuse_assessment({"status": "error", "error_code": "INVALID_INPUT"}, V())
        assert r.failed_modalities == {"water": "INVALID_INPUT"}
        assert "水质数据分析失败" in r.reason

    def test_all_modalities_failed_explains_why(self):
        r = fuse_assessment({"status": "error", "error_code": "INVALID_INPUT"},
                            {"status": "error", "error_code": "IMAGE_NOT_FOUND"})
        assert r.modalities == []
        assert "无法定级" in r.reason
        assert len(r.failed_modalities) == 2

    def test_no_false_alarm_when_modality_absent(self):
        """压根没传图片，不该报"图片分析失败"。"""
        r = fuse_assessment(W(), None)
        assert r.failed_modalities == {}
        assert not any(a.code == "MODALITY_UNAVAILABLE" for a in r.alerts)


# --------------------------------------------------------------------------- #
# 五、LLM 配置
# --------------------------------------------------------------------------- #

class TestLLMConfig:

    def test_default_is_mock(self):
        cfg = load_llm_config("scheduler", env={})
        assert cfg.is_mock
        assert cfg.provider == "mock"

    def test_role_specific_env_beats_generic(self):
        env = {
            "FISHERIES_LLM_PROVIDER": "openai",
            "FISHERIES_LLM_ADVISOR_PROVIDER": "aqua-7b-ollama",
        }
        assert load_llm_config("scheduler", env=env).provider == "openai"
        assert load_llm_config("advisor", env=env).provider == "aqua-7b-ollama"

    def test_preset_fills_base_url_and_model(self):
        cfg = load_llm_config("advisor", env={"FISHERIES_LLM_PROVIDER": "aqua-7b-ollama"})
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.model == "aqua-7b"
        assert not cfg.is_mock, "配了 base_url 就不该判为 mock"

    def test_api_key_can_come_from_provider_specific_env(self):
        cfg = load_llm_config("scheduler", env={
            "FISHERIES_LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "sk-test-123",
        })
        assert cfg.api_key == "sk-test-123"

    def test_bad_number_falls_back_to_default(self):
        cfg = load_llm_config("scheduler", env={"FISHERIES_LLM_TEMPERATURE": "abc"})
        assert cfg.temperature == 0.2


# --------------------------------------------------------------------------- #
# 六、OpenAI 兼容响应解析
# --------------------------------------------------------------------------- #

class TestResponseParsing:

    def test_parses_tool_calls(self):
        body = {
            "model": "m", "choices": [{
                "finish_reason": "tool_calls",
                "message": {"content": None, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "water_risk_predict",
                                 "arguments": '{"pond_id": "P1"}'},
                }]},
            }],
        }
        resp = OpenAICompatibleClient._parse(body)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "water_risk_predict"
        assert resp.tool_calls[0].arguments == {"pond_id": "P1"}

    def test_broken_arguments_are_preserved_not_silently_emptied(self):
        """模型把参数写坏了，必须留下痕迹，不能当成空参数继续跑。"""
        body = {"choices": [{"message": {"tool_calls": [{
            "id": "c1", "function": {"name": "f", "arguments": "{不是JSON"}}]}}]}
        call = OpenAICompatibleClient._parse(body).tool_calls[0]
        assert "__parse_error__" in call.arguments

    def test_empty_choices_raises_unavailable(self):
        with pytest.raises(LLMUnavailableError):
            OpenAICompatibleClient._parse({"choices": []})


# --------------------------------------------------------------------------- #
# 七、MockLLM 客户端的工具决策
# --------------------------------------------------------------------------- #

class TestMockLLM:

    def _ctx_msg(self, ctx: dict[str, Any]) -> dict[str, str]:
        from fisheries.agent.prompts import CTX_MARKER
        return {"role": "system", "content": "sys\n" + CTX_MARKER + json.dumps(ctx,
                                                                              ensure_ascii=False)}

    def test_calls_water_tool_when_readings_present(self):
        client = MockLLMClient()
        msgs = [self._ctx_msg({"water_readings": GOOD_READINGS, "image_path": None})]
        resp = client.chat(msgs, tools=[{"function": {"name": "water_risk_predict"}}])
        assert [c.name for c in resp.tool_calls] == ["water_risk_predict"]

    def test_calls_both_tools_when_both_present(self):
        client = MockLLMClient()
        msgs = [self._ctx_msg({"water_readings": GOOD_READINGS, "image_path": "f.jpg"})]
        resp = client.chat(msgs, tools=[{"function": {"name": "water_risk_predict"}},
                                        {"function": {"name": "fish_disease_detect"}}])
        assert {c.name for c in resp.tool_calls} == {"water_risk_predict", "fish_disease_detect"}

    def test_stops_after_tools_called(self):
        client = MockLLMClient()
        msgs = [
            self._ctx_msg({"water_readings": GOOD_READINGS}),
            {"role": "tool", "name": "water_risk_predict", "content": "{}"},
        ]
        resp = client.chat(msgs, tools=[{"function": {"name": "water_risk_predict"}}])
        assert resp.tool_calls == []
        assert resp.content

    def test_no_tool_call_when_no_data(self):
        client = MockLLMClient()
        resp = client.chat([self._ctx_msg({"water_readings": None, "image_path": None})],
                           tools=[{"function": {"name": "water_risk_predict"}}])
        assert resp.tool_calls == []


# --------------------------------------------------------------------------- #
# 八、调度层
# --------------------------------------------------------------------------- #

class SilentClient:
    """一个永远不调工具的调度客户端——用来验证护栏是否生效。"""

    def chat(self, messages, tools=None):
        return ChatResponse(content="我不打算调用任何工具。", finish_reason="stop")


class BrokenClient:
    def chat(self, messages, tools=None):
        raise LLMUnavailableError("模拟：后端未启动")


class BadArgsClient:
    """第一轮传错参数，第二轮不再调工具。"""

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return ChatResponse(finish_reason="tool_calls", tool_calls=[
                ToolCall(id="c1", name="water_risk_predict",
                         arguments={"readings": "不是数组", "不存在的参数": 1}),
            ])
        return ChatResponse(content="好", finish_reason="stop")


class LoopClient:
    """永远重复调同一个工具——验证防死循环。"""

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        return ChatResponse(finish_reason="tool_calls", tool_calls=[
            ToolCall(id=f"c{self.n}", name="water_risk_predict",
                     arguments={"readings": GOOD_READINGS}),
        ])


class TestOrchestrator:

    def test_full_run_calls_both_tools(self):
        agent = AquacultureAgent()
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS,
                                 image_path="__no_such__.jpg"))
        called = [t.title for t in r.trace if t.kind == "tool_call"]
        assert any("water_risk_predict" in t for t in called)
        assert any("fish_disease_detect" in t for t in called)
        assert r.risk_level in ("正常", "预警", "高危")
        assert r.trace  # 轨迹非空

    def test_guardrail_fills_missing_tool(self):
        """LLM 一个工具都不调，护栏必须补上，且要如实记录。"""
        agent = AquacultureAgent(scheduler_client=SilentClient(),
                                 advisor_client=MockLLMClient())
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        assert r.water is not None and r.water.get("status") == "ok"
        guard = [t for t in r.trace if t.kind == "guardrail"]
        assert guard, "护栏补调必须留痕，不能悄悄补"
        assert any("漏调" in g.title for g in guard)
        assert r.degraded_mode is True

    def test_guardrail_also_fills_image_tool(self):
        agent = AquacultureAgent(scheduler_client=SilentClient(),
                                 advisor_client=MockLLMClient())
        r = agent.run(AgentInput(pond_id="P1", image_path="__no_such__.jpg"))
        assert r.vision is not None
        assert r.vision.get("status") == "error"
        assert r.fusion["failed_modalities"] == {"vision": "IMAGE_NOT_FOUND"}

    def test_unknown_tool_becomes_structured_error(self):
        agent = AquacultureAgent()
        payload = agent._execute_tool("not_a_tool", {})
        assert payload["error_code"] == "UNKNOWN_TOOL"
        assert "not_a_tool" in payload["error_message"]

    def test_bad_arguments_become_structured_error(self):
        agent = AquacultureAgent()
        payload = agent._execute_tool("water_risk_predict", {"nope": 1})
        assert payload["error_code"] == "BAD_ARGUMENTS"

    def test_bad_args_from_llm_do_not_crash_agent(self):
        agent = AquacultureAgent(scheduler_client=BadArgsClient(),
                                 advisor_client=MockLLMClient())
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        assert r.risk_level in ("正常", "预警", "高危")
        assert any(t.kind == "tool_result" and "error" in t.title for t in r.trace)

    def test_duplicate_call_is_rejected(self):
        agent = AquacultureAgent(scheduler_client=LoopClient(),
                                 advisor_client=MockLLMClient())
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        assert any("拒绝重复调用" in t.title for t in r.trace)
        # 循环被轮数上限强制中断
        assert any("上限" in t.title for t in r.trace)

    def test_llm_unavailable_falls_back_and_still_answers(self):
        agent = AquacultureAgent(scheduler_client=BrokenClient(),
                                 advisor_client=BrokenClient())
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        assert r.water is not None, "LLM 挂了，护栏仍要保证工具被调用"
        assert r.advisories, "LLM 挂了，仍要有规则模板建议"
        assert any("不可用" in x for x in r.degraded_reasons)

    def test_max_rounds_is_bounded(self):
        assert 1 <= MAX_ROUNDS <= 20

    def test_single_modality_runs(self):
        agent = AquacultureAgent()
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        assert r.fusion["effective_weights"] == {"water": 1.0}

    def test_result_is_json_serialisable(self):
        agent = AquacultureAgent()
        r = agent.run(AgentInput(pond_id="P1", water_readings=GOOD_READINGS))
        dumped = r.model_dump(mode="json")
        assert json.dumps(dumped, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 九、输入契约
# --------------------------------------------------------------------------- #

class TestAgentInput:

    def test_requires_at_least_one_modality(self):
        with pytest.raises(ValueError):
            AgentInput(pond_id="P1")

    def test_rejects_empty_reading_list(self):
        """空列表是"传了但没数据"，与 None（没传）语义不同，要拦住。"""
        with pytest.raises(ValueError):
            AgentInput(pond_id="P1", water_readings=[])

    def test_accepts_image_only(self):
        assert AgentInput(image_path="f.jpg").image_path == "f.jpg"

    def test_summary_mentions_both(self):
        s = AgentInput(pond_id="P9", water_readings=GOOD_READINGS,
                       image_path="f.jpg").summary()
        assert "P9" in s and "水质" in s and "图片" in s


# --------------------------------------------------------------------------- #
# 十、建议输出解析
# --------------------------------------------------------------------------- #

class TestAdvisorParsing:

    def test_parses_plain_json(self):
        got = AquacultureAgent._parse_advisor_output('{"headline": "x", "advisories": []}')
        assert got == {"headline": "x", "advisories": []}

    def test_parses_json_in_markdown_fence(self):
        got = AquacultureAgent._parse_advisor_output(
            '```json\n{"headline": "x", "advisories": []}\n```')
        assert got is not None and got["headline"] == "x"

    def test_returns_none_for_prose(self):
        assert AquacultureAgent._parse_advisor_output("建议开增氧机。") is None

    def test_returns_none_for_broken_json(self):
        assert AquacultureAgent._parse_advisor_output('{"headline": ') is None


class TestFallbackAdvisories:
    """规则模板建议器 —— 现在返回 AdvisorReport（结构化四块）。"""

    def test_healthy_case_says_maintain_routine(self):
        f = fuse_assessment(W(), V())
        rep = AquacultureAgent._fallback_advisories(f, W(), V())
        assert len(rep.water_actions) == 1
        assert "常规" in rep.water_actions[0].action
        # 健康鱼不该产生任何病害处置建议
        assert rep.disease_actions == []

    def test_redline_advice_is_first_priority(self):
        f = fuse_assessment(W(redline_hits=["DO_REDLINE"]), None)
        rep = AquacultureAgent._fallback_advisories(
            f, W(redline_hits=["DO_REDLINE"]), None)
        assert rep.water_actions[0].priority == 1
        # 面向养殖户的文案不能出现 DO_REDLINE 这种规则编号
        assert "泛塘" in rep.water_actions[0].target
        assert "DO_REDLINE" not in rep.water_actions[0].target

    def test_disease_advice_includes_confidence(self):
        f = fuse_assessment(None, bacterial_gill(0.87))
        rep = AquacultureAgent._fallback_advisories(f, None, bacterial_gill(0.87))
        assert any("0.87" in a.target for a in rep.disease_actions)

    def test_low_confidence_adds_recheck_advice(self):
        v = bacterial_gill(0.3)
        v["is_low_confidence"] = True
        f = fuse_assessment(None, v)
        rep = AquacultureAgent._fallback_advisories(f, None, v)
        assert any("把握不大" in a.target for a in rep.disease_actions)

    def test_farmer_facing_text_stays_short(self):
        """面向养殖户的硬约束：动作 ≤40 字、时间 ≤20 字。"""
        f = fuse_assessment(W(redline_hits=["DO_REDLINE_PROJECTED"]), bacterial_gill(0.87))
        rep = AquacultureAgent._fallback_advisories(
            f, W(redline_hits=["DO_REDLINE_PROJECTED"]), bacterial_gill(0.87))
        assert len(rep.headline) <= 30
        for a in rep.all_actions():
            assert len(a.action) <= 40, a.action
            assert len(a.timing) <= 20, a.timing

    def test_advisory_schema_accepts_template_output(self):
        f = fuse_assessment(W(severity_map={"dissolved_oxygen": 1.0}), None)
        rep = AquacultureAgent._fallback_advisories(
            f, W(severity_map={"dissolved_oxygen": 1.0}), None)
        for item in rep.all_actions():
            assert item.priority >= 1
            assert item.action

    def test_template_risk_level_and_alerts_come_from_fusion(self):
        """模板路径的结构化字段必须与融合层同源，不能凭空造。"""
        f = fuse_assessment(W(redline_hits=["DO_REDLINE"]), None)
        rep = AquacultureAgent._fallback_advisories(
            f, W(redline_hits=["DO_REDLINE"]), None)
        assert rep.risk_level == f.risk_level
        assert [a.code for a in rep.alerts] == [a.code for a in f.alerts]

    def test_headline_names_the_disease_when_water_is_fine(self):
        """水质满分、只查出病害时，标题不能写「水质开始不对」。"""
        v = sapro(0.62)
        w = W()
        f = fuse_assessment(w, v)
        assert f.risk_level == "预警"
        rep = AquacultureAgent._fallback_advisories(f, w, v)
        assert "水霉" in rep.headline
        assert "水质" not in rep.headline

    def test_headline_names_the_water_factor_when_water_is_bad(self):
        """水质真有问题时，标题要点名水质侧的限值因子。"""
        w = W(risk_level="高危", risk_score=100.0, limiting_factor_label="溶氧")
        v = V()
        f = fuse_assessment(w, v)
        rep = AquacultureAgent._fallback_advisories(f, w, v)
        assert "溶氧" in rep.headline
        assert "水质" not in rep.headline

    def test_headline_for_disease_only_critical_names_the_disease(self):
        v = septicemia(0.9)
        f = fuse_assessment(None, v)
        assert f.risk_level == "高危"
        rep = AquacultureAgent._fallback_advisories(f, None, v)
        assert "出血" in rep.headline
        assert len(rep.headline) <= 30

    def test_disease_case_does_not_tell_farmer_to_feed_normally(self):
        """水质满分、确诊病害时，第 1 条动作必须是停料，而不是「照常喂料」。

        项目自己的知识库把「发病期停料」列为硬规则
        （knowledge/corpus/05_投料管理规范.md:71），
        模板若因为水质正常就回一句「按常规量喂料」，等于让养殖户给一塘病鱼正常投饵。
        """
        v = bacterial_gill(0.87)
        w = W()
        f = fuse_assessment(w, v)
        rep = AquacultureAgent._fallback_advisories(f, w, v)
        assert "停料" in rep.water_actions[0].action
        assert all("常规量喂料" not in a.action for a in rep.water_actions)

    def test_disease_case_observation_window_watches_the_disease(self):
        """只查出病害时，观察窗口该盯死鱼数量和病灶，而不是复测溶氧氨氮。"""
        v = bacterial_gill(0.87)
        w = W()
        f = fuse_assessment(w, v)
        rep = AquacultureAgent._fallback_advisories(f, w, v)
        assert "死鱼" in rep.observation_window
        assert "溶氧" not in rep.observation_window

    def test_two_action_lists_are_merged_by_priority(self):
        """all_actions() 把水质与病害两块合并后按 priority 排序。"""
        f = fuse_assessment(W(severity_map={"dissolved_oxygen": 1.0}),
                            bacterial_gill(0.87))
        rep = AquacultureAgent._fallback_advisories(
            f, W(severity_map={"dissolved_oxygen": 1.0}), bacterial_gill(0.87))
        merged = rep.all_actions()
        assert len(merged) == len(rep.water_actions) + len(rep.disease_actions)
        assert [a.priority for a in merged] == sorted(a.priority for a in merged)


# --------------------------------------------------------------------------- #
# 九、建议报告护栏（大模型只能改措辞，不能改定级）
# --------------------------------------------------------------------------- #

class ScriptedAdvisor:
    """返回固定 JSON 的建议角色替身，用来测试护栏而不是测试模型。"""

    def __init__(self, payload: dict[str, Any] | str) -> None:
        self.payload = payload
        self.last_messages: list[dict[str, Any]] = []

    def chat(self, messages, tools=None):
        self.last_messages = messages
        content = (self.payload if isinstance(self.payload, str)
                   else json.dumps(self.payload, ensure_ascii=False))
        return ChatResponse(content=content, finish_reason="stop")


def _agent_with_advisor(payload: dict[str, Any] | str) -> AquacultureAgent:
    """调度角色用离线 Mock（会正常调工具），只把建议角色换成脚本化替身。"""
    return AquacultureAgent(advisor_client=ScriptedAdvisor(payload))


def bad_do_readings() -> list[dict[str, Any]]:
    """溶氧 24 小时内从 6.2 掉到 1.8 —— 命中 DO_REDLINE，必判高危。

    数据是造出来的，但形状与真实传感器一致（分钟级递增时间戳 + 五个必测指标）。
    """
    out: list[dict[str, Any]] = []
    for i in range(24):
        ratio = i / 23
        out.append({
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": 28.0 + 1.5 * ratio,
            "ph": 7.8,
            "dissolved_oxygen": round(6.2 - 4.4 * ratio, 3),
            "ammonia_n": round(0.15 + 0.10 * ratio, 3),
        })
    return out


#: 真实数据集里的烂鳃图片，用于验证"水质 + 图像"双模态链路。
#: 数据集不入库，所以这里用 default_path（找不到也不抛）——否则一台没挂
#: 数据集的机器连 import 这个测试模块都会失败。是否跳过由下面的常量控制。
GILL_IMAGE = (
    _datasets.default_path("fish_disease")
    / "Test" / "Bacterial gill disease"
    / "Bacterial gill disease (100).jpg"
)
GILL_IMAGE_AVAILABLE = GILL_IMAGE.is_file()


class TestConfidenceNote:
    """可信度说明：事实由规则算，措辞照模板给。

    这一组测试守的是同一件事——**可信度说明里出现的每一句话，
    都必须能在输入里找到出处**。它既不能是一句放之四海皆准的套话
    （"本建议可靠不可靠请结合实际"），也不能由模型凭印象判断
    （"本次结论可靠"而图像其实分析失败了）。
    """

    def _facts(self, water=None, vision=None, fusion=None, knowledge=None):
        return build_confidence_facts(water, vision, fusion, knowledge)

    def test_note_has_three_fixed_clauses(self):
        note = render_confidence_note(self._facts(W(), V(), {"risk_level": "正常"}))
        assert note.startswith("依据：")
        assert "不确定：" in note and "复核：" in note
        assert note.endswith("。")

    def test_no_double_period_and_no_stray_space_before_etc(self):
        """标点细节也要钉住：枚举以 etc. 结尾时不能变成 "etc.."。

        中文侧同理——" 等" 前那个空格在 "甲；乙 等。" 里看着像排版事故。
        这两处都是渲染层的手工拼接，不测就会在某个分支上悄悄退化。
        """
        facts = self._facts(
            W(data_quality={"n_points": 12, "interval_minutes": 30},
              degraded_mode=True, extrapolation_horizon_hours=4.0),
            None,
            {"risk_level": "预警", "modalities": ["water"],
             "failed_modalities": {"vision": "IMAGE_NOT_FOUND"}},
        )
        zh = render_confidence_note(facts)
        en = render_confidence_note(facts, "en")
        assert "。。" not in zh and ".." not in en
        assert " 等" not in zh
        assert "。。" not in en and "。." not in zh
        # 中文句读一个都不能少
        assert zh.count("。") == 3

    def test_confident_case_says_so_explicitly(self):
        """没有已知不确定项时必须明说——留空的话读者会以为我们忘了写。"""
        facts = self._facts(
            W(data_quality={"n_points": 48, "interval_minutes": 60}),
            None,
            {"risk_level": "正常", "modalities": ["water"]},
        )
        note = render_confidence_note(facts)
        assert "无明显不确定项" in note
        assert "无需人工复核" in note

    def test_too_few_water_points_is_declared(self):
        facts = self._facts(
            W(data_quality={"n_points": 12, "interval_minutes": 60}),
            None,
            {"risk_level": "预警", "modalities": ["water"]},
        )
        note = render_confidence_note(facts)
        assert "12 点" in note
        assert "趋势偏弱" in note

    def test_enough_points_is_not_flagged(self):
        """点数够了就不该出现"点少"这条——出现过一次就说明阈值判错了。"""
        facts = self._facts(
            W(data_quality={"n_points": 24, "interval_minutes": 60}),
            None,
            {"risk_level": "正常", "modalities": ["water"]},
        )
        assert "趋势偏弱" not in render_confidence_note(facts)

    def test_low_confidence_disease_asks_for_human_review(self):
        facts = self._facts(
            W(data_quality={"n_points": 24}),
            bacterial_gill(0.5) | {"is_low_confidence": True},
            {"risk_level": "预警", "modalities": ["water", "vision"]},
        )
        note = render_confidence_note(facts)
        assert "病害置信度偏低" in note
        assert "建议人工复看图片" in note

    def test_failed_modality_is_declared_in_the_note(self):
        """图像分析失败这件事必须写进可信度说明，而不是只留在 trace 里。"""
        facts = self._facts(
            W(data_quality={"n_points": 24}),
            None,
            {"risk_level": "预警", "modalities": ["water"],
             "failed_modalities": {"vision": "IMAGE_NOT_FOUND"}},
        )
        note = render_confidence_note(facts)
        assert "图像分析失败" in note
        assert "建议人工重做图像分析" in note

    def test_unverified_knowledge_is_counted(self):
        class _Chunk:
            def __init__(self, verified: bool) -> None:
                self.verified = verified

        class _Ctx:
            chunks: ClassVar[list[_Chunk]] = [_Chunk(True), _Chunk(False), _Chunk(False)]

        facts = self._facts(
            W(data_quality={"n_points": 24}),
            None,
            {"risk_level": "正常", "modalities": ["water"]},
            knowledge=_Ctx(),
        )
        note = render_confidence_note(facts)
        assert "引用 3 条资料中 2 条未与标准原文核对" in note

    def test_missing_knowledge_is_not_reported_as_unverified(self):
        """没检索到资料 ≠ 检索到未核对的资料，这两件事不能混。"""
        facts = self._facts(
            W(data_quality={"n_points": 24}), None,
            {"risk_level": "正常", "modalities": ["water"]},
        )
        assert "标准原文" not in render_confidence_note(facts)

    def test_extrapolation_is_declared(self):
        facts = self._facts(
            W(data_quality={"n_points": 24}, extrapolation_horizon_hours=6.9),
            None,
            {"risk_level": "预警", "modalities": ["water"]},
        )
        note = render_confidence_note(facts)
        assert "6.9 小时外推" in note
        assert "建议尽快补测一次水质" in note

    def test_vision_only_run_says_water_was_not_measured(self):
        facts = self._facts(
            None, bacterial_gill(0.9),
            {"risk_level": "高危", "modalities": ["vision"]},
        )
        assert "仅鱼体图片" in render_confidence_note(facts)

    def test_block_gives_the_model_a_filled_template(self):
        """提示词里给模型的必须是"已填好事实的三栏 + 固定句式"，不是空白模板。"""
        facts = self._facts(
            W(data_quality={"n_points": 24}), bacterial_gill(0.87),
            {"risk_level": "高危", "modalities": ["water", "vision"]},
        )
        block = format_confidence_block(facts)
        assert "- 依据：水质 24 点 + 图片检出烂鳃（把握 0.87）" in block
        assert "句式固定为：依据：…。不确定：…。复核：…。" in block

    def test_english_block_is_english(self):
        """英文提示词里不能塞中文素材——那等于让英文底座先做一次阅读理解的翻译。"""
        facts = self._facts(
            W(data_quality={"n_points": 24}), bacterial_gill(0.87),
            {"risk_level": "高危", "modalities": ["water", "vision"]},
        )
        block = format_confidence_block(facts, "en")
        assert "Evidence:" in block
        assert "bacterial gill disease" in block
        assert "烂鳃" not in block
        assert "依据" not in block

    def test_advisor_message_carries_the_block(self):
        msg = build_advisor_user_message(
            {"risk_level": "高危", "fused_score": 88.0, "reason": "…",
             "evidence": [], "alerts": []},
            W(data_quality={"n_points": 24}), bacterial_gill(0.87),
        )
        assert "## 可信度素材" in msg
        assert "- 依据：水质 24 点 + 图片检出烂鳃（把握 0.87）" in msg

    def test_fallback_note_states_the_evidence_not_a_platitude(self):
        """规则模板路径写出来的可信度说明也得有信息量，不能是一句套话。"""
        fusion = fuse_assessment(W(data_quality={"n_points": 24}), bacterial_gill(0.87))
        report = AquacultureAgent._fallback_advisories(
            fusion, W(data_quality={"n_points": 24}), bacterial_gill(0.87),
        )
        assert report.confidence_note.startswith("依据：")
        assert "24 点" in report.confidence_note
        assert "烂鳃" in report.confidence_note

    def test_model_omitting_the_note_falls_back_to_the_template(self):
        """模型漏写这一栏时补模板句，而不是在报告里留一个空栏。"""
        from fisheries.agent.orchestrator import AquacultureAgent as _Agent

        agent = _Agent()
        agent._confidence_note_default = "依据：水质 24 点。不确定：无明显不确定项。复核：无需人工复核。"
        fusion = fuse_assessment(W(data_quality={"n_points": 24}), None)
        report = agent._report_from_parsed(
            {"risk_level": fusion.risk_level, "headline": "照常巡塘。",
             "confidence_note": "   ", "alerts": []},
            fusion, W(data_quality={"n_points": 24}), None,
        )
        assert report.confidence_note == agent._confidence_note_default

    def test_model_written_note_is_kept(self):
        agent = AquacultureAgent()
        agent._confidence_note_default = "模板句。"
        fusion = fuse_assessment(W(data_quality={"n_points": 24}), None)
        report = agent._report_from_parsed(
            {"risk_level": fusion.risk_level, "headline": "照常巡塘。",
             "confidence_note": "依据：水质 24 点。不确定：无。复核：无需。",
             "alerts": []},
            fusion, W(data_quality={"n_points": 24}), None,
        )
        assert report.confidence_note.startswith("依据：水质 24 点")


class TestAdvisorReportGuardrail:
    """建议报告的结构化字段必须由融合层决定，大模型无权改写。"""

    def test_wrong_risk_level_discards_whole_output(self):
        """等级写错说明模型误读了输入，其 headline 也不能信——必须整份作废。"""
        agent = _agent_with_advisor({
            "risk_level": "正常",
            "headline": "水质正常，一切良好。",
            "water_actions": [
                {"priority": 1, "target": "溶氧", "action": "建议加强管理。", "timing": "随时"},
            ],
        })
        result = agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        assert result.risk_level == "高危"
        assert result.report["risk_level"] == "高危"
        # 自相矛盾的报告比没有报告更危险：不能出现"高危 + 水质正常，一切良好"
        assert "一切良好" not in result.report["headline"]
        assert "建议加强管理" not in [a["action"] for a in result.report["water_actions"]]
        # 模型试图降级必须留下痕迹，且被记录为降级原因
        assert any("风险等级" in r for r in result.degraded_reasons)
        assert any(s.kind == "guardrail" for s in result.trace)
        # 退回模板后仍然给出可执行建议，而不是空报告
        assert result.report["water_actions"]

    def test_alerts_come_from_fusion_not_from_model(self):
        agent = _agent_with_advisor({
            "risk_level": "高危",
            "headline": "溶氧危险",
            "alerts": [{"level": "info", "title": "模型自己编的告警", "detail": "x"}],
        })
        result = agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        titles = [a["title"] for a in result.report["alerts"]]
        assert "模型自己编的告警" not in titles
        assert titles == [a["title"] for a in result.alerts]
        assert any("告警" in r for r in result.degraded_reasons)

    @pytest.mark.skipif(not GILL_IMAGE_AVAILABLE, reason="未导入鱼类病害图像数据集")
    def test_actions_are_split_into_water_and_disease_blocks(self):
        agent = _agent_with_advisor({
            "risk_level": "高危",
            "headline": "溶氧和烂鳃都要处理",
            "water_actions": [
                {"priority": 1, "target": "溶氧", "action": "马上开增氧机", "timing": "现在"},
            ],
            "disease_actions": [
                {"priority": 1, "target": "烂鳃", "action": "捞出病鱼隔离", "timing": "今天"},
                {"priority": 2, "target": "用药", "action": "按说明书泼洒", "timing": "明天"},
            ],
        })
        result = agent.run(AgentInput(
            pond_id="P01", water_readings=bad_do_readings(),
            image_path=str(GILL_IMAGE),
        ))
        rep = result.report
        assert [a["target"] for a in rep["water_actions"]] == ["溶氧"]
        assert [a["target"] for a in rep["disease_actions"]] == ["烂鳃", "用药"]
        # 展平的 advisories 必须是两块合并的结果
        assert len(result.advisories) == 3

    def test_illegal_action_items_are_dropped_not_fatal(self):
        """一条 priority 非法的建议不该炸掉整个 Agent 循环。"""
        agent = _agent_with_advisor({
            "risk_level": "高危",
            "water_actions": [
                {"priority": 0, "target": "溶氧", "action": "马上开增氧机", "timing": "现在"},
                {"target": "氨氮", "action": "停料", "timing": "今天"},
                "这不是一个对象",
            ],
        })
        result = agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        targets = [a["target"] for a in result.report["water_actions"]]
        assert "溶氧" not in targets      # priority=0 违反 ge=1
        assert "氨氮" in targets          # 缺 priority 时按序号补
        assert len(targets) == 1

    def test_legacy_advisories_field_still_accepted(self):
        """模型换版本退回旧字段名时，建议不该整份丢失。"""
        agent = _agent_with_advisor({
            "risk_level": "高危",
            "advisories": [
                {"priority": 1, "target": "溶氧", "action": "马上开增氧机", "timing": "现在"},
            ],
        })
        result = agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        assert [a["target"] for a in result.report["water_actions"]] == ["溶氧"]

    def test_broken_json_falls_back_to_template(self):
        agent = _agent_with_advisor("这不是 JSON，是模型在胡说")
        result = agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        assert result.report["water_actions"]          # 模板兜底，不是空报告
        assert result.report["risk_level"] == result.risk_level
        assert any("规则模板" in r for r in result.degraded_reasons)

    def test_domain_prompt_forbids_feeding_a_diseased_pond(self):
        """知识库把「发病期停料」列为硬规则，LLM 路径的提示词也必须写进去。

        模板路径已由 test_disease_case_does_not_tell_farmer_to_feed_normally 钉住；
        这条钉 LLM 路径——否则接上大模型反而可能建议「按常规量喂料」。
        """
        assert "停料" in DOMAIN_CONSTRAINTS
        assert "不是「健康」" in DOMAIN_CONSTRAINTS
        # 建议角色要按风险来源挑观察窗口，不能一律复测水质
        assert "observation_window" in ADVISOR_SYSTEM_PROMPT
        assert "数死鱼" in ADVISOR_SYSTEM_PROMPT

    def test_advisor_prompt_injects_all_four_blocks(self):
        """提示词里必须同时出现四个业务块的字段名，否则模型没机会输出它们。"""
        advisor = ScriptedAdvisor({"risk_level": "高危"})
        agent = AquacultureAgent(advisor_client=advisor)
        agent.run(AgentInput(pond_id="P01", water_readings=bad_do_readings()))
        system = advisor.last_messages[0]["content"]
        user = advisor.last_messages[1]["content"]
        for key in ("risk_level", "alerts", "water_actions", "disease_actions"):
            assert key in system or key in user
        # 面向养殖户的篇幅与通俗性要求必须写进系统提示词
        assert "40 个字" in system
        assert "养殖户" in system


