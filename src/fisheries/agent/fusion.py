"""M4 多源融合与风险决策 —— 确定性规则实现。

设计原则（这一条比代码本身更重要）
====================================
**融合决策绝不交给大模型。**

原因有三：
1. **可复现**：同一份传感器数据 + 同一张图片，必须永远得出同一个风险等级。
   如果定级由 LLM 采样决定，两次告警可能不一致，养殖户无法信任。
2. **可审计**：评委或监管方问"为什么判高危"，我们必须能给出规则 ID 和数值依据，
   而不是"模型觉得"。
3. **可测试**：只有确定性逻辑才能写单元测试钉住边界行为。

因此本模块的分工是：
- **本模块**：算风险等级、算融合分、生成结构化告警（alerts）、产出证据链（evidence）。
- **大模型**：只负责把这些结构化结论"翻译"成自然语言养殖建议（advisories）。

一句话：**数字和告警由规则算，措辞由模型写。模型不得改动数字。**

融合公式（来源于《技术方案设计.md》第 7 节）
==============================================
    weighted_score = w_water × S_water + w_vision × S_vision + synergy_bonus

    S_water  = water.risk_score / 100          （0~1）
    S_vision = severity_weight × confidence     （0~1，按置信度加权）
    w_water  = 0.6，w_vision = 0.4

当只有单一模态输入时（例如只传了图片、没有传感器数据），
缺失模态的权重会被剔除并**在剩余模态上重新归一化**——
否则"只有图片"的满分级病害也只能得 40 分，永远进不了高危，
这是典型的"权重配置把单模态场景锁死"的缺陷。

等级下限（level floors）——为什么加权平均不够
================================================
加权平均隐含一个假设：两个模态在测量**同一个**潜在量，所以一个模态"没看到问题"
可以抵消另一个模态看到的问题。这里不成立：

    S_water  衡量的是**环境危害**（溶氧 1.9 mg/L，今晚就可能泛塘）
    S_vision 衡量的是**已见病征**（照片上还没烂鳃）

"水质高危但照片上鱼看着还行"不等于风险减半——那只是**还没出现症状**。
同理"水质全优但检出烂鳃"也不是"照常管理"：确诊病害就要用药、隔离、算停药期。

因此本模块的定级规则是：
    level = max(加权分档, 水质模态自身档位, 病害检出档位, 红线, 高致病性病害)
并把最终判级所依据的分数记为 `fused_score`（≥ 纯公式值 `weighted_score`），
抬升幅度与原因分别记录在 `score_floor` / `level_floors` 中，绝不隐瞒。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# 一、融合参数（全部集中在顶部，便于标定与答辩时解释）
# --------------------------------------------------------------------------- #

#: 水质模态权重
W_WATER: float = 0.60
#: 图像模态权重
W_VISION: float = 0.40

#: 等级判定阈值，与 water_risk 工具保持一致，避免同一套系统出现两种分档口径
SCORE_CRITICAL: float = 70.0
SCORE_WARNING: float = 40.0

#: 协同加成规则表：(规则ID, 说明, 加成值)
#: 这些规则编码的是"单看水质不严重、单看鱼病也不严重，但两者同时出现就很危险"的领域知识，
#: 也是本项目"多源融合"相对单模态方案的核心增量。
SYNERGY_RULES: dict[str, tuple[str, float]] = {
    "SYN_BACTERIAL_ON_WARNING": (
        "水质已达预警及以上，同时检出细菌性病害，细菌繁殖条件与宿主抵抗力下降叠加",
        12.0,
    ),
    "SYN_NITROGEN_WITH_DISEASE": (
        "氨氮或亚硝酸盐超警戒，同时检出任何病害，毒物与病原双重胁迫",
        8.0,
    ),
    "SYN_LOW_DO_WITH_GILL": (
        "溶氧处于低位/预测将持续下降，同时检出烂鳃，鳃组织摄氧能力已受损",
        15.0,
    ),
    "SYN_SAPRO_ON_COLD": (
        "水温处于 15–25℃ 水霉易发区间，同时检出水霉病，符合水霉低温高发规律",
        10.0,
    ),
}

#: 触发红线的病害严重度门槛：严重度达到该值的病害，即使水质完全正常也直接高危
#: （病毒性白尾 0.95 / 细菌性败血症 0.90 会命中；水霉 0.70、赤皮 0.65 不会）
SEVERE_DISEASE_WEIGHT: float = 0.85
#: 高置信度门槛：只有够确信的严重病害才允许单独触发高危，避免误检直接拉满告警
SEVERE_DISEASE_MIN_CONF: float = 0.70

#: 病害下限规则："检出病害"升级为"至少预警"的严重度门槛。
#: 只有严重度 ≥ 该值的病害（烂鳃 0.85 / 败血症 0.90 / 赤皮 0.65 不达标）才享受该下限，
#: 避免轻微皮肤病变也被拉进预警。
DISEASE_MIN_SEVERITY_FOR_WARNING: float = 0.70
#: 病害下限规则的置信度门槛。
#: 取 0.40 而不是工具自身的低置信线 0.60（`fisheries.tools.fish_disease.LOW_CONFIDENCE_THRESHOLD`），
#: 是因为两者的语义不同：0.60 以下代表"识别结果需要人工复核"，
#: 但"需要复核"不等于"可以按正常处理"——确诊流程仍要启动，只是同时附上复核提示。
#: 0.40 以下是纯噪声区间，放过它以免把告警训成背景音。
DISEASE_MIN_CONF_FOR_WARNING: float = 0.40


# --------------------------------------------------------------------------- #
# 二、输出契约
# --------------------------------------------------------------------------- #

RiskLevel = Literal["正常", "预警", "高危"]


class Evidence(BaseModel):
    """一条证据：结论 + 数值依据。用于答辩展示与人工复核。"""

    source: Literal["water", "vision", "fusion"]
    code: str = Field(..., description="证据编码，例如 DO_REDLINE_PROJECTED")
    label: str = Field(..., description="人类可读的说明")
    value: float | str | None = Field(default=None, description="支撑该证据的数值")
    unit: str = Field(default="", description="数值单位")


class Alert(BaseModel):
    """一条告警。结构化输出，可直接送短信/看板。"""

    code: str
    level: Literal["info", "warning", "critical"]
    title: str
    detail: str


class FusionResult(BaseModel):
    """融合决策结果。"""

    risk_level: RiskLevel
    fused_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="最终判级所依据的分数 = max(weighted_score, 等级下限对应的分数线)",
    )
    weighted_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="纯加权公式值（含协同加成），保留原始信息供审计与标定",
    )
    score_floor: float | None = Field(
        default=None,
        description="等级下限把分数抬升到的值；None 表示未被抬升（fused_score == weighted_score）",
    )
    level_floors: list[str] = Field(
        default_factory=list,
        description="触发的等级下限规则 ID，例如 WATER_LEVEL:高危 / DISEASE_DETECTED",
    )
    water_score: float | None = Field(default=None, description="水质模态归一化得分 0~1")
    vision_score: float | None = Field(default=None, description="图像模态归一化得分 0~1")
    effective_weights: dict[str, float] = Field(default_factory=dict, description="实际生效的模态权重")
    synergy_score: float = Field(default=0.0, description="协同加成总分")
    synergy_hits: list[str] = Field(default_factory=list, description="命中的协同规则 ID")
    alerts: list[Alert] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str = Field(..., description="定级理由，一句话，由规则拼装")
    modalities: list[str] = Field(default_factory=list, description="本次实际参与融合的模态")
    failed_modalities: dict[str, str] = Field(
        default_factory=dict,
        description="用户提供了数据但工具执行失败的模态：{模态: 错误码}",
    )
    degraded_mode: bool = Field(default=False, description="true 表示存在模态降级（如模型未训练）")


# --------------------------------------------------------------------------- #
# 三、单模态归一化
# --------------------------------------------------------------------------- #

def _water_score(water: dict[str, Any] | None) -> tuple[float | None, list[Evidence]]:
    """把 water_risk_predict 的输出压成 0~1 的分数，并抽取水质侧证据。"""
    if not water or water.get("status") != "ok":
        return None, []

    ev: list[Evidence] = []
    score = float(water.get("risk_score") or 0.0) / 100.0

    for hit in water.get("redline_hits") or []:
        ev.append(Evidence(source="water", code=hit, label=f"水质红线：{hit}"))

    sev_map = water.get("severity_map") or {}
    for key, sev in sev_map.items():
        if sev and sev > 0:
            trend = (water.get("trends") or {}).get(key) or {}
            ev.append(
                Evidence(
                    source="water",
                    code=f"SEV_{key.upper()}",
                    label=f"{trend.get('label', key)}处于{'危险' if sev >= 1.0 else '警戒'}区间",
                    value=trend.get("current"),
                    unit=trend.get("unit", ""),
                )
            )

    eta = water.get("projected_do_alert_eta_hours")
    if eta is not None:
        ev.append(
            Evidence(
                source="water",
                code="DO_ALERT_ETA",
                label=f"预计 {eta:.1f} 小时后溶氧跌破 {water.get('projected_do_alert_threshold')} mg/L"
                      f"{water.get('projected_do_alert_label', '')}",
                value=eta,
                unit="小时",
            )
        )
    return score, ev


def _vision_score(vision: dict[str, Any] | None) -> tuple[float | None, list[Evidence]]:
    """把 fish_disease_detect 的输出压成 0~1 的分数，并抽取图像侧证据。

    注意用 **severity_weight × confidence** 而不是直接取 severity_weight：
    模型说"烂鳃，但只有 0.3 的把握"和"烂鳃，0.95 的把握"不该给同一个分。
    """
    if not vision or vision.get("status") != "ok":
        return None, []

    ev: list[Evidence] = []
    if vision.get("is_healthy"):
        ev.append(Evidence(source="vision", code="VISION_HEALTHY", label="未检出明显病害特征"))
        return 0.0, ev

    sev = float(vision.get("severity_weight") or 0.0)
    conf = float(vision.get("confidence") or 0.0)
    score = sev * conf

    ev.append(
        Evidence(
            source="vision",
            code="VISION_DISEASE",
            label=f"检出{vision.get('disease_type')}（{vision.get('disease_group')}），"
                  f"置信度 {conf:.2f}",
            value=round(conf, 4),
        )
    )
    if vision.get("is_low_confidence"):
        ev.append(
            Evidence(
                source="vision",
                code="VISION_LOW_CONF",
                label="识别置信度偏低，建议补拍清晰照片或人工复核",
                value=round(conf, 4),
            )
        )
    if vision.get("raw_class"):
        ev.append(
            Evidence(source="vision", code="VISION_RAW_CLASS",
                     label=f"底层模型原始判定：{vision['raw_class']}")
        )
    return score, ev


# --------------------------------------------------------------------------- #
# 四、协同规则
# --------------------------------------------------------------------------- #

def _evaluate_synergy(water: dict[str, Any] | None,
                      vision: dict[str, Any] | None) -> tuple[float, list[str], list[Evidence]]:
    """评估所有跨模态协同规则，返回 (总分, 命中规则ID, 证据)。"""
    if not water or not vision:
        return 0.0, [], []          # 单模态谈不上"协同"
    if water.get("status") != "ok" or vision.get("status") != "ok":
        return 0.0, [], []

    total = 0.0
    hits: list[str] = []
    ev: list[Evidence] = []

    sev_map = water.get("severity_map") or {}
    water_level = water.get("risk_level")
    disease_type = vision.get("disease_type")
    disease_group = vision.get("disease_group")
    is_healthy = bool(vision.get("is_healthy"))

    # ① 水质 ≥ 预警 ∧ 细菌性病害
    if water_level in ("预警", "高危") and not is_healthy and disease_group == "细菌性":
        rid, desc, bonus = "SYN_BACTERIAL_ON_WARNING", *SYNERGY_RULES["SYN_BACTERIAL_ON_WARNING"]
        total += bonus
        hits.append(rid)
        ev.append(Evidence(source="fusion", code=rid, label=desc, value=bonus))

    # ② 氨氮或亚硝酸盐超警戒 ∧ 检出任何病害
    nitrogen_bad = (sev_map.get("ammonia_n", 0) or 0) > 0 or (sev_map.get("nitrite_n", 0) or 0) > 0
    if nitrogen_bad and not is_healthy:
        rid, desc, bonus = "SYN_NITROGEN_WITH_DISEASE", *SYNERGY_RULES["SYN_NITROGEN_WITH_DISEASE"]
        total += bonus
        hits.append(rid)
        ev.append(Evidence(source="fusion", code=rid, label=desc, value=bonus))

    # ③ 溶氧低位或预测下降 ∧ 烂鳃
    do_stressed = ((sev_map.get("dissolved_oxygen", 0) or 0) > 0
                   or water.get("projected_do_alert_eta_hours") is not None)
    if do_stressed and disease_type == "烂鳃":
        rid, desc, bonus = "SYN_LOW_DO_WITH_GILL", *SYNERGY_RULES["SYN_LOW_DO_WITH_GILL"]
        total += bonus
        hits.append(rid)
        ev.append(Evidence(source="fusion", code=rid, label=desc, value=bonus))

    # ④ 水温 15–25℃ ∧ 水霉（水霉在低温高发，高温反而少见）
    temp = ((water.get("trends") or {}).get("water_temp") or {}).get("current")
    if isinstance(temp, (int, float)) and 15.0 <= temp <= 25.0 and disease_type == "水霉":
        rid, desc, bonus = "SYN_SAPRO_ON_COLD", *SYNERGY_RULES["SYN_SAPRO_ON_COLD"]
        total += bonus
        hits.append(rid)
        ev.append(Evidence(source="fusion", code=rid, label=desc, value=bonus))

    return total, hits, ev


# --------------------------------------------------------------------------- #
# 五、主入口
# --------------------------------------------------------------------------- #

def fuse_assessment(water: dict[str, Any] | None = None,
                    vision: dict[str, Any] | None = None) -> FusionResult:
    """融合水质风险与病害识别结果，输出确定的等级、告警与证据链。

    Args:
        water: `water_risk_predict` 的返回值（可为 None，表示本次没有水质数据）。
        vision: `fish_disease_detect` 的返回值（可为 None）。

    Returns:
        FusionResult，可直接 `.model_dump(mode="json")` 序列化。

    Examples:
        >>> w = {"status": "ok", "risk_level": "预警", "risk_score": 53.6,
        ...      "severity_map": {"dissolved_oxygen": 0.5}, "redline_hits": [],
        ...      "trends": {"water_temp": {"current": 22.0, "label": "水温", "unit": "℃"}},
        ...      "projected_do_alert_eta_hours": None}
        >>> v = {"status": "ok", "disease_type": "水霉", "disease_group": "真菌性",
        ...      "confidence": 0.88, "severity_weight": 0.70, "is_healthy": False}
        >>> r = fuse_assessment(w, v)
        >>> r.risk_level in ("正常", "预警", "高危")
        True
    """
    w_score, w_ev = _water_score(water)
    v_score, v_ev = _vision_score(vision)

    # ---- 显式记录"用户给了数据但工具失败"的模态 ----
    # 不记录这个就会出现最危险的失败模式：用户明明传了照片，图片工具报错，
    # 融合层静默丢掉图像模态，理由里还写"本次仅有水质单模态输入"，
    # 用户以为照片被分析了 —— 结论看起来完全正常，实际少了一整路证据。
    failed_modalities: dict[str, str] = {}
    if water is not None and water.get("status") != "ok":
        failed_modalities["water"] = str(water.get("error_code") or "UNKNOWN")
    if vision is not None and vision.get("status") != "ok":
        failed_modalities["vision"] = str(vision.get("error_code") or "UNKNOWN")

    modalities: list[str] = []
    if w_score is not None:
        modalities.append("water")
    if v_score is not None:
        modalities.append("vision")

    # ---- 单模态时权重重新归一化（否则单模态永远进不了高危） ----
    if modalities == ["water"]:
        weights = {"water": 1.0}
    elif modalities == ["vision"]:
        weights = {"vision": 1.0}
    elif len(modalities) == 2:
        weights = {"water": W_WATER, "vision": W_VISION}
    else:
        weights = {}

    base = 0.0
    if "water" in weights:
        base += weights["water"] * (w_score or 0.0)
    if "vision" in weights:
        base += weights["vision"] * (v_score or 0.0)

    synergy, syn_hits, syn_ev = _evaluate_synergy(water, vision)
    weighted = min(100.0, max(0.0, base * 100.0 + synergy))

    evidence = w_ev + v_ev + syn_ev
    alerts: list[Alert] = []

    # ---- 模态失败告警：优先级仅次于红线，必须让人看见 ----
    for mod, code in failed_modalities.items():
        name = "水质数据" if mod == "water" else "鱼体图片"
        alerts.append(Alert(
            code="MODALITY_UNAVAILABLE",
            level="warning",
            title=f"{name}分析失败",
            detail=f"已接收{name}，但对应工具执行失败（{code}），"
                   f"本次结论**未包含该模态证据**，请修正输入后重新分析。",
        ))

    # ---- 红线优先级最高：命中即高危 ----
    redline_hits = (water or {}).get("redline_hits") or []
    for hit in redline_hits:
        alerts.append(Alert(
            code=hit, level="critical",
            title="水质红线告警",
            detail=f"触发规则 {hit}，需立即人工干预并启动应急预案。",
        ))

    # ---- 严重病害单模态红线：够确信的高危病害直接升级 ----
    severe_disease = False
    if vision and vision.get("status") == "ok" and not vision.get("is_healthy"):
        sw = float(vision.get("severity_weight") or 0.0)
        cf = float(vision.get("confidence") or 0.0)
        if sw >= SEVERE_DISEASE_WEIGHT and cf >= SEVERE_DISEASE_MIN_CONF:
            severe_disease = True
            alerts.append(Alert(
                code="SEVERE_DISEASE",
                level="critical",
                title="高致病性病害告警",
                detail=f"检出{vision.get('disease_type')}（严重度 {sw:.2f}，置信度 {cf:.2f}），"
                       f"即使水质正常也需立即隔离处置。",
            ))

    # ---- 等级下限（level floors）：融合等级不得低于任一单模态自身的档位 ----
    # 这一段是本模块唯一的"非线性"部分，也是最容易被写成加权平均而犯错的地方：
    #   · 水质工具独立判高危，照片上鱼却看着正常 → 加权平均会把高危稀释成预警，
    #     等于"没看到病鱼"被当成了"水质不危险"的反证。
    #   · 水质全优时图像模态只占 0.4 权重，任何非"高致病性"的确认病害都进不了 40 分线，
    #     结论会写成"照常管理"，而确诊病害本该立刻用药。
    level_floors: list[str] = []

    if redline_hits:
        level_floors.append("REDLINE")
    if severe_disease:
        level_floors.append("SEVERE_DISEASE")

    water_ok = bool(water) and water.get("status") == "ok"
    water_level = water.get("risk_level") if water_ok else None
    if water_level in ("预警", "高危"):
        level_floors.append(f"WATER_LEVEL:{water_level}")

    if (vision and vision.get("status") == "ok" and not vision.get("is_healthy")
            and float(vision.get("severity_weight") or 0.0) >= DISEASE_MIN_SEVERITY_FOR_WARNING
            and float(vision.get("confidence") or 0.0) >= DISEASE_MIN_CONF_FOR_WARNING):
        level_floors.append("DISEASE_DETECTED")

    if "REDLINE" in level_floors or "SEVERE_DISEASE" in level_floors \
            or "WATER_LEVEL:高危" in level_floors:
        floor_level: RiskLevel = "高危"
    elif level_floors:
        floor_level = "预警"
    else:
        floor_level = "正常"

    # ---- 定级：加权分档与等级下限取较大者 ----
    if floor_level == "高危" or weighted >= SCORE_CRITICAL:
        level: RiskLevel = "高危"
    elif floor_level == "预警" or weighted >= SCORE_WARNING:
        level = "预警"
    else:
        level = "正常"

    # fused_score 表示"最终判级所依据的分数"。被下限抬升时必须抬到对应分数线上，
    # 否则会出现"高危 / 32.3 分"这种自相矛盾的报告——按分数筛阈值的调用方会漏掉它。
    # 抬升前的纯公式值保留在 weighted_score 里，信息不丢。
    floor_score = {"高危": SCORE_CRITICAL, "预警": SCORE_WARNING}.get(level, 0.0)
    fused = max(weighted, floor_score)
    score_floor = round(floor_score, 2) if floor_score > weighted + 1e-9 else None

    if score_floor is not None:
        raised_by = ", ".join(level_floors)
        evidence.append(Evidence(
            source="fusion",
            code="LEVEL_FLOOR",
            label=f"加权公式得分 {weighted:.1f} 低于【{level}】档线，"
                  f"已按等级下限（{raised_by}）抬升至 {fused:.1f}",
            value=round(weighted, 2),
        ))
        alerts.append(Alert(
            code="LEVEL_FLOOR",
            level="critical" if level == "高危" else "warning",
            title=f"等级按下限抬升至【{level}】",
            detail=f"加权公式得分仅 {weighted:.1f}，但 {raised_by} 要求等级不低于【{level}】，"
                   f"故按 {fused:.1f} 处理。**请以下限规则为准，不要只看分数**。",
        ))

    # ---- 协同规则产生的告警 ----
    for rid in syn_hits:
        desc = SYNERGY_RULES[rid][0]
        alerts.append(Alert(
            code=rid,
            level="warning" if level != "高危" else "critical",
            title="多源协同风险",
            detail=desc,
        ))

    # ---- 定级理由（规则拼装，不由模型生成） ----
    reason = _build_reason(level, fused, weights, w_score, v_score, synergy,
                           syn_hits, redline_hits, severe_disease, modalities,
                           failed_modalities, weighted, score_floor, level_floors)

    degraded = bool(
        (water or {}).get("degraded_mode") or (vision or {}).get("degraded_mode")
    )

    return FusionResult(
        risk_level=level,
        fused_score=round(fused, 2),
        weighted_score=round(weighted, 2),
        score_floor=score_floor,
        level_floors=level_floors,
        water_score=round(w_score, 4) if w_score is not None else None,
        vision_score=round(v_score, 4) if v_score is not None else None,
        effective_weights=weights,
        synergy_score=round(synergy, 2),
        synergy_hits=syn_hits,
        alerts=alerts,
        evidence=evidence,
        reason=reason,
        modalities=modalities,
        failed_modalities=failed_modalities,
        degraded_mode=degraded,
    )


def _build_reason(level: str, fused: float, weights: dict[str, float],
                  w_score: float | None, v_score: float | None, synergy: float,
                  syn_hits: list[str], redline_hits: list[str],
                  severe_disease: bool, modalities: list[str],
                  failed_modalities: dict[str, str] | None = None,
                  weighted: float | None = None,
                  score_floor: float | None = None,
                  level_floors: list[str] | None = None) -> str:
    """用规则拼装一句话定级理由。"""
    failed = failed_modalities or {}
    if not modalities:
        if failed:
            names = "、".join("水质数据" if m == "water" else "鱼体图片" for m in failed)
            return f"{names}对应的工具执行失败，未获得任何有效证据，无法定级。"
        return "未获得有效的水质数据或图像数据，无法定级。"

    floors = level_floors or []
    parts: list[str] = []
    seg: list[str] = []
    if "water" in weights:
        seg.append(f"水质模态 {w_score:.2f}×{weights['water']:.2f}")
    if "vision" in weights:
        seg.append(f"图像模态 {v_score:.2f}×{weights['vision']:.2f}")
    formula = " + ".join(seg) + (f" + 协同加成 {synergy:.0f}" if synergy else "")

    if score_floor is not None and weighted is not None:
        parts.append(f"融合分 {fused:.1f}（加权公式 {weighted:.1f}：{formula}；"
                     f"等级下限抬升至 {fused:.1f}）")
    else:
        parts.append(f"融合分 {fused:.1f}（{formula}）")
    parts.append(f"判定等级【{level}】")

    why: list[str] = []
    if redline_hits:
        why.append(f"命中水质红线 {('、'.join(redline_hits))}")
    if severe_disease:
        why.append("检出高致病性病害")
    # 红线与严重病害已有专门措辞，这里只补充新增的两条下限，避免同义重复
    extra_floors = [f for f in floors
                    if f not in ("REDLINE", "SEVERE_DISEASE")
                    and not (redline_hits and f.startswith("WATER_LEVEL"))]
    if extra_floors:
        why.append(f"等级下限 {'、'.join(extra_floors)}")
    if syn_hits:
        why.append(f"命中 {len(syn_hits)} 条多源协同规则")
    if not why:
        why.append("无红线命中，按融合分落入该档")

    if len(modalities) == 1:
        parts.append(f"本次仅有{'水质' if modalities[0] == 'water' else '图像'}单模态输入，"
                     f"权重已归一化")
    if failed:
        names = "、".join("水质数据" if m == "water" else "鱼体图片" for m in failed)
        parts.append(f"⚠ {names}分析失败，结论未包含该模态证据")
    return "，".join(parts) + "；" + "；".join(why) + "。"


__all__ = [
    "Alert",
    "DISEASE_MIN_CONF_FOR_WARNING",
    "DISEASE_MIN_SEVERITY_FOR_WARNING",
    "Evidence",
    "FusionResult",
    "SCORE_CRITICAL",
    "SCORE_WARNING",
    "SEVERE_DISEASE_MIN_CONF",
    "SEVERE_DISEASE_WEIGHT",
    "SYNERGY_RULES",
    "W_VISION",
    "W_WATER",
    "fuse_assessment",
]
