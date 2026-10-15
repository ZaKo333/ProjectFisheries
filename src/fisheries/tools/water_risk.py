"""工具 1：water_risk_predict —— 水质时序风险预测。

定位
----
本模块是 Agent 的「水质预测工具」，对外只暴露一个函数 `water_risk_predict`。
当前实现是 **规则基线版**（阈值判定 + 线性趋势外推），承担两个职责：

1. 在 LSTM 时序模型训练完成前，让 Agent 全链路可跑通、可测试、可演示；
2. 模型上线后作为**降级兜底**——模型服务不可用时自动回落到本实现，接口不变。

与模型版的接口约定
------------------
后续接入 LSTM 时，只需替换 `_predict_series()` 的内部实现，对外函数签名与
返回 JSON 结构完全不变，因此 Agent 调度层与后端 API 无需改动。
返回值中的 `degraded_mode` 字段用于标记"当前是否走的是降级基线"。

风险分算法（与《技术方案设计.md》第 7 节一致）
---------------------------------------------
    risk_score = clip( Σ(w_i × severity_i) × persistence, 0, 100 )
    severity_i ∈ {0.0 适宜, 0.5 警戒, 1.0 危险}
    persistence = min(1.5, 1.0 + 0.1 × (最长连续超阈点数 − 1))

权重在"实际可用指标"上重新归一化——这一点很关键：真实采集里亚硝酸盐探头
经常缺失，若不做归一化，缺一项就会系统性低估风险分。

等级判定不只看分数，而是「分数 + 阈值带 + 红线规则」三者联合：
    高危 ← 命中任一红线规则，或 risk_score ≥ 70，
           或**任一指标落入危险区间**（单项越界即高危，不做平均稀释）
    预警 ← risk_score ≥ 40，或任一指标落入警戒区间
    正常 ← 其余

这里刻意**不用加权分直接定级**。加权平均有个致命缺陷：5 项指标里 4 项正常、
1 项溶氧跌到 1.8 mg/L，平均下来分数可能只有 40 出头，看起来"还好"——
但塘里的鱼已经在浮头了。因此定级走"任一越界即升级"的保守逻辑，
加权分只用来表征**风险量级**，用于多塘排序与看板展示。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# --------------------------------------------------------------------------- #
# 一、指标元信息与阈值配置
# --------------------------------------------------------------------------- #

#: 指标 → 中文名 / 单位 / 物理量程（用于外推结果裁剪，避免出现负溶氧这类荒谬值）
INDICATOR_META: dict[str, dict[str, Any]] = {
    "water_temp": {"label": "水温", "unit": "℃", "phys_range": (-5.0, 45.0)},
    "ph": {"label": "pH", "unit": "", "phys_range": (0.0, 14.0)},
    "dissolved_oxygen": {"label": "溶氧", "unit": "mg/L", "phys_range": (0.0, 25.0)},
    "ammonia_n": {"label": "氨氮", "unit": "mg/L", "phys_range": (0.0, 50.0)},
    "nitrite_n": {"label": "亚硝酸盐", "unit": "mg/L", "phys_range": (0.0, 50.0)},
}

#: 风险权重（原始值，未归一化）。来源：《技术方案设计.md》7.1。
#: 其中"投料 0.08"不在本工具入参内，将在 Agent 融合层（M4）单独计入。
BASE_WEIGHTS: dict[str, float] = {
    "dissolved_oxygen": 0.35,
    "ammonia_n": 0.20,
    "nitrite_n": 0.15,
    "ph": 0.12,
    "water_temp": 0.10,
}

#: 品种阈值档案（工程默认值 v0.1：标准限值对到 GB 11607-1989 原文，
#: 警戒 / 危险 / 红线是在标准之上的工程扩展，须由有资质的水产技术人员复核后冻结）
#: 命名约定：
#:   temp_opt / ph_opt   —— 适宜区间 (lo, hi)，含端点
#:   temp_warn / ph_warn —— 警戒区间 (lo, hi)，含端点；超出即危险
#:   *_opt / *_warn      —— 单侧指标：do_* 越大越好，ammonia_*/nitrite_* 越小越好
SPECIES_PROFILES: dict[str, dict[str, Any]] = {
    "鲤科": {
        "temp_opt": (20.0, 30.0), "temp_warn": (15.0, 33.0),
        "ph_opt": (6.5, 8.5), "ph_warn": (6.0, 9.0),
        "do_opt": 5.0, "do_warn": 3.0,
        "ammonia_opt": 0.2, "ammonia_warn": 0.5,
        "nitrite_opt": 0.1, "nitrite_warn": 0.3,
    },
    "罗非鱼": {
        "temp_opt": (22.0, 32.0), "temp_warn": (18.0, 35.0),
        "ph_opt": (6.5, 8.5), "ph_warn": (6.0, 9.0),
        "do_opt": 5.0, "do_warn": 3.0,
        "ammonia_opt": 0.2, "ammonia_warn": 0.5,
        "nitrite_opt": 0.1, "nitrite_warn": 0.3,
    },
    "虾蟹": {
        "temp_opt": (25.0, 32.0), "temp_warn": (20.0, 34.0),
        "ph_opt": (7.5, 8.5), "ph_warn": (7.0, 9.0),
        "do_opt": 4.0, "do_warn": 2.5,
        "ammonia_opt": 0.1, "ammonia_warn": 0.3,
        "nitrite_opt": 0.05, "nitrite_warn": 0.2,
    },
}

#: 可选品种的顺序列表。**从 :data:`SPECIES_PROFILES` 派生**，
#: 保证「界面上能选的品种」与「有阈值档案的品种」永远不会分叉 ——
#: 手抄一份字面量迟早会漏掉新加的品种，然后下拉框里选了一个没有档案的值。
SPECIES_CHOICES: tuple[str, ...] = tuple(SPECIES_PROFILES)

#: 红线阈值：命中任一即直接判「高危」，不再看加权分。
#:
#: 注意 ``PH_LOW_REDLINE`` / ``PH_HIGH_REDLINE`` 只是本表的键：pH 越界时
#: :func:`_check_redlines` 发出的是 **``PH_REDLINE``** 这一个码。
#: 前端字典如果照着这张表去列红线码，就会多出两个永远收不到的键、
#: 同时漏掉真正会收到的码。
REDLINE = {
    "DO_REDLINE": 1.5,        # 溶氧 < 1.5 mg/L：泛塘临界
    "NH3_REDLINE": 1.0,       # 氨氮 > 1.0 mg/L
    "NO2_REDLINE": 0.5,       # 亚硝酸盐 > 0.5 mg/L
    "PH_LOW_REDLINE": 5.5,    # pH < 5.5
    "PH_HIGH_REDLINE": 9.5,   # pH > 9.5
    "DO_CRASH_DROP": 2.0,     # 3 小时内溶氧跌幅 > 2.0 mg/L
    "DO_CRASH_WINDOW_H": 3.0,
}

#: 等级判定阈值
SCORE_CRITICAL = 70.0
SCORE_WARNING = 40.0
PERSISTENCE_CAP = 1.5

#: 「还有多久要动手」的行动阈值，按危险程度由近到远排列。
#: 外推时取**第一条尚未跌破的线**，这样无论当前处于哪一档，都能给出一个可执行的倒计时：
#: 溶氧 6.0 → 报"还有多久跌破 3.0"（该开增氧机）；溶氧 2.6 → 报"还有多久跌破 1.5"（马上要翻塘）。
ACTION_THRESHOLDS: dict[str, tuple[tuple[float, str], ...]] = {
    "dissolved_oxygen": ((3.0, "危险线"), (1.5, "泛塘临界值")),
    "ammonia_n": ((1.0, "红线"),),
    "nitrite_n": ((0.5, "红线"),),
}

#: 低置信度提示：有效样本少于该条数时，趋势外推不可靠
MIN_RELIABLE_POINTS = 24

#: 线性外推的最大时长。**这是本模块最重要的一个约束。**
#: 水体指标随时间的演化是饱和/衰减型曲线，不是直线：真实池塘不会让溶氧线性跌到 0，
#: 而是跌到某个谷值后因增氧机启动、藻类恢复而反弹。把 24 小时的斜率外推到 72 小时，
#: 会算出"72 小时后溶氧 0.00 mg/L、水温 34℃"这种一看就假的结果，
#: 直接损害告警的可信度。因此数值外推只报最近 12 小时，
#: 更远的未来交给 LSTM 模型；horizon_hours 仍然参与趋势判定与风险打分。
MAX_EXTRAPOLATION_HOURS = 12.0


# --------------------------------------------------------------------------- #
# 二、入参 / 出参数据契约（Pydantic v2）
# --------------------------------------------------------------------------- #

class WaterQualityPoint(BaseModel):
    """单条水质采样记录。

    Examples:
        >>> p = WaterQualityPoint(water_temp=28.4, ph=8.1,
        ...                       dissolved_oxygen=2.6, ammonia_n=0.62, nitrite_n=0.15)
        >>> p.dissolved_oxygen
        2.6
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None = Field(
        default=None, description="采样时间，ISO 8601；缺失时按 interval_minutes 等间隔处理"
    )
    water_temp: float = Field(..., description="水温 ℃")
    ph: float = Field(..., description="pH 值")
    dissolved_oxygen: float = Field(..., description="溶解氧 mg/L")
    ammonia_n: float = Field(..., description="氨氮（以 N 计）mg/L")
    nitrite_n: float | None = Field(
        default=None, description="亚硝酸盐氮 mg/L；探头缺失时为 None，该指标不参与打分"
    )

    @field_validator("water_temp", "ph", "dissolved_oxygen", "ammonia_n", "nitrite_n")
    @classmethod
    def _must_be_finite(cls, v: float | None) -> float | None:
        """拦截 NaN / inf：传感器掉线常产生这些值，必须挡在打分之外。"""
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError("数值必须为有限数（不允许 NaN / inf）")
        return v

    @field_validator("ph")
    @classmethod
    def _ph_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 14.0:
            raise ValueError(f"pH 超出物理量程 [0, 14]：{v}")
        return v


class WaterRiskInput(BaseModel):
    """`water_risk_predict` 的完整入参。"""

    model_config = ConfigDict(extra="forbid")

    pond_id: str = Field(default="unknown", description="池塘编号")
    readings: list[WaterQualityPoint] = Field(
        ..., min_length=3, description="按时间升序的水质序列，至少 3 条"
    )
    interval_minutes: int = Field(default=60, ge=1, description="采样间隔（分钟）")
    horizon_hours: int = Field(default=72, ge=1, le=168, description="预测窗口（小时）")
    species: Literal["鲤科", "罗非鱼", "虾蟹"] = Field(default="鲤科", description="养殖品种")


class IndicatorTrend(BaseModel):
    """单个指标的趋势统计量。"""

    indicator: str = Field(..., description="指标英文键")
    label: str = Field(..., description="指标中文名")
    unit: str = Field(..., description="单位")
    current: float = Field(..., description="最新一条读数")
    mean: float = Field(..., description="窗口均值")
    minimum: float = Field(..., description="窗口最小值")
    maximum: float = Field(..., description="窗口最大值")
    slope_per_hour: float = Field(..., description="最小二乘拟合斜率，单位/小时；负值代表下降")
    direction: Literal["上升", "下降", "平稳"] = Field(..., description="趋势方向")
    change_pct: float = Field(..., description="窗口内相对首值的变化百分比")


class IndicatorProjection(BaseModel):
    """单个指标的线性外推结果（规则基线，非模型预测）。"""

    indicator: str = Field(..., description="指标英文键")
    label: str = Field(..., description="指标中文名")
    unit: str = Field(..., description="单位")
    horizon_used_hours: float = Field(..., description="本次外推实际使用的时长（已被 MAX_EXTRAPOLATION_HOURS 截断）")
    projected_at_horizon: float = Field(..., description="外推时长后的预测值")
    projected_extreme: float = Field(..., description="外推区间内的极值（溶氧取最小，氨氮等取最大）")
    extreme_at_hour: float = Field(..., description="极值出现的时间点，单位小时")
    hours_to_threshold: float | None = Field(
        default=None,
        description="预计多久后触及该指标的下一条管控阈值；未触及为 null",
    )
    threshold_used: float | None = Field(
        default=None, description="hours_to_threshold 对应的阈值数值"
    )
    threshold_label: str | None = Field(
        default=None, description="阈值名称，例如「危险线」「泛塘临界值」「红线」"
    )


class DataQuality(BaseModel):
    """数据质量摘要，供 Agent 判断结论可信度。"""

    n_points: int = Field(..., description="有效记录条数")
    interval_minutes: int = Field(..., description="实际使用的采样间隔")
    interval_source: Literal["timestamp", "parameter"] = Field(..., description="间隔来源")
    span_hours: float = Field(..., description="序列覆盖时长（小时）")
    missing_indicators: list[str] = Field(
        default_factory=list, description="缺失的指标（不参与打分）"
    )
    warnings: list[str] = Field(default_factory=list, description="数据质量告警")


class WaterRiskOutput(BaseModel):
    """`water_risk_predict` 的返回体。Agent 与后端均以此为准。"""

    model_config = ConfigDict(protected_namespaces=())

    tool: str = Field(default="water_risk_predict", description="工具名，便于链路审计")
    status: Literal["ok", "error"] = Field(default="ok", description="执行状态")
    pond_id: str = Field(default="unknown", description="池塘编号")
    risk_level: Literal["正常", "预警", "高危"] = Field(..., description="综合风险等级")
    risk_score: float = Field(..., ge=0, le=100, description="综合风险分 0–100")
    horizon_hours: int = Field(..., description="风险评估窗口")
    extrapolation_horizon_hours: float = Field(
        ..., description="数值外推实际使用的时长（≤ MAX_EXTRAPOLATION_HOURS）"
    )
    species: str = Field(..., description="使用的品种阈值档案")
    degradation_summary: str = Field(..., description="水质恶化预测的自然语言描述（规则生成，非 LLM 编造）")
    limiting_factor: str | None = Field(default=None, description="主控风险因子（英文键）")
    limiting_factor_label: str | None = Field(default=None, description="主控风险因子（中文名）")
    projected_do_alert_eta_hours: float | None = Field(
        default=None,
        description="预计多少小时后溶氧跌破下一条管控线；未触及为 null。这是最直接可执行的一个数",
    )
    projected_do_alert_threshold: float | None = Field(
        default=None, description="上述倒计时所对应的溶氧阈值数值"
    )
    projected_do_alert_label: str | None = Field(
        default=None, description="上述阈值的名称（危险线 / 泛塘临界值）"
    )
    exceeded_indicators: list[str] = Field(
        default_factory=list, description="处于警戒或危险区间的指标（中文名）"
    )
    rule_hits: list[str] = Field(default_factory=list, description="命中的规则编号列表")
    redline_hits: list[str] = Field(default_factory=list, description="命中的红线规则编号列表")
    severity_map: dict[str, float] = Field(
        default_factory=dict, description="各指标严重度 0/0.5/1.0"
    )
    effective_weights: dict[str, float] = Field(
        default_factory=dict, description="归一化后的实际权重"
    )
    trends: dict[str, IndicatorTrend] = Field(default_factory=dict, description="各指标趋势")
    projections: dict[str, IndicatorProjection] = Field(
        default_factory=dict, description="各指标外推"
    )
    data_quality: DataQuality | None = Field(default=None, description="数据质量摘要")
    degraded_mode: bool = Field(
        default=True, description="true 表示走的是规则基线而非 LSTM 模型"
    )
    projection_method: str = Field(
        default="linear_extrapolation_baseline", description="外推方法标识"
    )
    error_code: str | None = Field(default=None, description="status=error 时的错误码")
    error_message: str | None = Field(default=None, description="status=error 时的错误说明")


# --------------------------------------------------------------------------- #
# 三、阈值判定（三个基础函数，覆盖全部 5 个指标的语义）
# --------------------------------------------------------------------------- #

def _double_side(v: float, opt_lo: float, opt_hi: float,
                 warn_lo: float, warn_hi: float) -> tuple[float, str]:
    """双侧指标（水温、pH）：适宜区间内正常，警戒区间内预警，其余危险。"""
    if v < warn_lo or v > warn_hi:
        return 1.0, "危险"
    if v < opt_lo or v > opt_hi:
        return 0.5, "警戒"
    return 0.0, "适宜"


def _upper_side(v: float, opt_max: float, warn_max: float) -> tuple[float, str]:
    """越小越好的指标（氨氮、亚硝酸盐）。"""
    if v > warn_max:
        return 1.0, "危险"
    if v > opt_max:
        return 0.5, "警戒"
    return 0.0, "适宜"


def _lower_side(v: float, opt_min: float, warn_min: float) -> tuple[float, str]:
    """越大越好的指标（溶解氧）。"""
    if v < warn_min:
        return 1.0, "危险"
    if v < opt_min:
        return 0.5, "警戒"
    return 0.0, "适宜"


def _severity(indicator: str, value: float, p: dict[str, Any]) -> tuple[float, str]:
    """按指标分派到对应的判定函数，返回 (严重度, 中文状态)。

    Examples:
        >>> _severity("ph", 8.5, SPECIES_PROFILES["鲤科"])
        (0.0, '适宜')
        >>> _severity("ammonia_n", 0.2, SPECIES_PROFILES["鲤科"])
        (0.0, '适宜')
    """
    if indicator == "water_temp":
        return _double_side(value, *p["temp_opt"], *p["temp_warn"])
    if indicator == "ph":
        return _double_side(value, *p["ph_opt"], *p["ph_warn"])
    if indicator == "dissolved_oxygen":
        return _lower_side(value, p["do_opt"], p["do_warn"])
    if indicator == "ammonia_n":
        return _upper_side(value, p["ammonia_opt"], p["ammonia_warn"])
    if indicator == "nitrite_n":
        return _upper_side(value, p["nitrite_opt"], p["nitrite_warn"])
    raise KeyError(f"未知指标：{indicator}")


# --------------------------------------------------------------------------- #
# 四、趋势与外推
# --------------------------------------------------------------------------- #

def _slope_per_hour(values: list[float], interval_hours: float) -> float:
    """最小二乘拟合斜率，单位：指标值 / 小时。

    用回归而非"首末两点之差"，是为了抑制单点抖动造成的误判。
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = [i * interval_hours for i in range(n)]
    mx = sum(xs) / n
    my = sum(values) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / denom


def _determine_direction(slope: float, mean: float, horizon_hours: int) -> str:
    """判断趋势方向；预测窗口内的累计变化不足均值的 5% 视为平稳。"""
    if mean == 0:
        return "上升" if slope > 0 else ("下降" if slope < 0 else "平稳")
    projected_change = abs(slope) * horizon_hours
    if projected_change < 0.05 * abs(mean):
        return "平稳"
    return "上升" if slope > 0 else "下降"


def _project(indicator: str, current: float, slope: float,
             horizon_hours: int) -> IndicatorProjection:
    """线性外推，裁剪到物理量程，并把外推时长限制在 MAX_EXTRAPOLATION_HOURS 内。

    另外计算 `hours_to_threshold`：按当前速率还要多久才会触及管控阈值。
    对溶氧用的是**危险线（3.0 mg/L）**而非红线（1.5），因为养殖户需要的是
    "还有多久要开增氧机"，而不是"还有多久要翻塘"。
    """
    lo, hi = INDICATOR_META[indicator]["phys_range"]
    h = min(float(horizon_hours), MAX_EXTRAPOLATION_HOURS)

    raw_end = current + slope * h
    end = min(max(raw_end, lo), hi)

    # 线性函数在区间端点上取极值；判定取哪个端点
    if indicator == "dissolved_oxygen":          # 越小越危险 → 取最小
        if raw_end <= current:
            extreme, at_hour = max(end, lo), h
        else:
            extreme, at_hour = current, 0.0
    else:                                         # 其余指标越大越危险 → 取最大
        if raw_end >= current:
            extreme, at_hour = min(end, hi), h
        else:
            extreme, at_hour = current, 0.0

    # ---- 预计多久触及"下一条"管控阈值 ----
    # 溶氧按由高到低的顺序找第一条尚未跌破的线：当前若在 3.0 以上就报"还有多久跌破 3.0"，
    # 若已在 3.0 以下但高于 1.5，就报"还有多久跌破 1.5 泛塘线"。氨氮/亚硝酸盐同理只看向上的红线。
    eta: float | None = None
    eta_target: float | None = None
    eta_label: str | None = None
    if abs(slope) > 1e-9:
        for target, label in ACTION_THRESHOLDS.get(indicator, ()):
            if indicator == "dissolved_oxygen":
                if slope < 0 and current > target:
                    eta, eta_target, eta_label = (current - target) / -slope, target, label
                    break
            else:
                if slope > 0 and current < target:
                    eta, eta_target, eta_label = (target - current) / slope, target, label
                    break
    if eta is not None and (eta < 0 or eta > MAX_EXTRAPOLATION_HOURS * 4):
        eta, eta_target, eta_label = None, None, None   # 太远的外推值没有决策价值

    return IndicatorProjection(
        indicator=indicator,
        label=INDICATOR_META[indicator]["label"],
        unit=INDICATOR_META[indicator]["unit"],
        horizon_used_hours=round(h, 2),
        projected_at_horizon=round(end, 3),
        projected_extreme=round(extreme, 3),
        extreme_at_hour=at_hour,
        hours_to_threshold=round(eta, 2) if eta is not None else None,
        threshold_used=eta_target,
        threshold_label=eta_label,
    )


# --------------------------------------------------------------------------- #
# 五、红线规则与普通规则
# --------------------------------------------------------------------------- #

def _longest_exceed_run(sev_matrix: list[dict[str, float]]) -> int:
    """最长连续"存在超阈指标"的采样点数，用于 persistence 放大。"""
    best = run = 0
    for point in sev_matrix:
        if any(s > 0 for s in point.values()):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _check_redlines(current: dict[str, float], series: dict[str, list[float]],
                    interval_hours: float,
                    projections: dict[str, IndicatorProjection]) -> list[str]:
    """检查红线规则。命中任一即直接判高危。

    判定基准是**当前状态**（最近 3 点均值）与**外推极值**，而不是整段历史的最大值。
    原因：3 天前的一次氨氮尖峰不该让今天的告警一直挂着"高危"，
    那样会让养殖户对告警脱敏。
    """
    hits: list[str] = []

    do_now = current.get("dissolved_oxygen")
    if do_now is not None:
        if do_now < REDLINE["DO_REDLINE"]:
            hits.append("DO_REDLINE")
        elif "dissolved_oxygen" in projections and \
                projections["dissolved_oxygen"].projected_extreme < REDLINE["DO_REDLINE"]:
            hits.append("DO_REDLINE_PROJECTED")

    if current.get("ammonia_n") is not None and \
            current["ammonia_n"] > REDLINE["NH3_REDLINE"]:
        hits.append("NH3_REDLINE")

    if current.get("nitrite_n") is not None and \
            current["nitrite_n"] > REDLINE["NO2_REDLINE"]:
        hits.append("NO2_REDLINE")

    ph_now = current.get("ph")
    if ph_now is not None and (ph_now < REDLINE["PH_LOW_REDLINE"]
                               or ph_now > REDLINE["PH_HIGH_REDLINE"]):
        hits.append("PH_REDLINE")

    # 溶氧骤降：只看**最近**一个 3 小时窗口，历史下跌不算数
    do_series = series.get("dissolved_oxygen", [])
    steps = max(1, round(REDLINE["DO_CRASH_WINDOW_H"] / interval_hours))
    tail_start = max(0, len(do_series) - 2 * steps)
    for i in range(tail_start + steps, len(do_series)):
        if do_series[i - steps] - do_series[i] > REDLINE["DO_CRASH_DROP"]:
            hits.append("DO_CRASH_TREND")
            break

    return hits


def _check_normal_rules(series: dict[str, list[float]], slopes: dict[str, float],
                        sev: dict[str, float], p: dict[str, Any]) -> list[str]:
    """检查非红线的提示性规则，用于丰富"为什么报警"的解释。"""
    hits: list[str] = []

    if sev.get("dissolved_oxygen", 0) > 0:
        hits.append("DO_BELOW_OPTIMAL")
    if sev.get("ammonia_n", 0) > 0:
        hits.append("NH3N_ABOVE_WARNING")
    if sev.get("nitrite_n", 0) > 0:
        hits.append("NO2N_ABOVE_WARNING")
    if sev.get("ph", 0) > 0:
        hits.append("PH_OUT_OF_OPTIMAL")
    if sev.get("water_temp", 0) > 0:
        hits.append("TEMP_OUT_OF_OPTIMAL")

    if slopes.get("dissolved_oxygen", 0.0) <= -0.10:
        hits.append("DO_TREND_DOWN_FAST")
    if slopes.get("ammonia_n", 0.0) >= 0.02:
        hits.append("NH3N_TREND_UP_FAST")
    if slopes.get("nitrite_n", 0.0) >= 0.01:
        hits.append("NO2N_TREND_UP_FAST")

    ph_series = series.get("ph", [])
    if ph_series and (max(ph_series) - min(ph_series) > 1.0):
        hits.append("PH_FLUCTUATION_HIGH")

    temp_series = series.get("water_temp", [])
    if temp_series and (max(temp_series) - min(temp_series) > 5.0):
        hits.append("TEMP_FLUCTUATION_HIGH")

    # 协同规则：pH 偏高 + 水温偏高 → 分子氨占比升高，氨氮毒性放大
    if series.get("ph") and series.get("water_temp"):
        if max(series["ph"]) > 8.5 and max(series["water_temp"]) > 28.0 \
                and sev.get("ammonia_n", 0) > 0:
            hits.append("UNIONIZED_AMMONIA_RISK")

    return hits


# --------------------------------------------------------------------------- #
# 六、自然语言恶化描述生成
# --------------------------------------------------------------------------- #

def _fmt(value: float, unit: str) -> str:
    """把数值和单位拼成可读字符串（pH 不带单位，其余用空格分隔）。"""
    return f"{value:.2f} {unit}" if unit else f"{value:.2f}"


def _build_summary(*, level: str, score: float, horizon_hours: int,
                   trends: dict[str, IndicatorTrend],
                   projections: dict[str, IndicatorProjection],
                   sev: dict[str, float], limiting: str | None,
                   points: list[WaterQualityPoint],
                   redline_hits: list[str]) -> str:
    """用规则拼装"水质恶化预测描述"。

    为什么不用 LLM 直接生成：这段文字会进入告警短信和看板，
    如果由模型自由发挥，数字可能与真实数据不一致（幻觉）。
    这里全部由已计算的字段拼装，保证与数值严格一致；
    Agent 后续可以在**不改动任何数字**的前提下把它改写成更口语化的表述。
    """
    parts: list[str] = []

    # 1) 总体趋势
    worsening = [t for t in trends.values()
                 if t.direction in ("上升", "下降")]
    do_trend = trends.get("dissolved_oxygen")
    if do_trend and do_trend.direction == "下降":
        trend_word = "呈持续恶化趋势"
    elif len(worsening) >= 2:
        trend_word = "存在波动恶化风险"
    elif any(t.direction != "平稳" for t in trends.values()):
        trend_word = "基本稳定、局部指标有变化"
    else:
        trend_word = "总体平稳"

    parts.append(
        f"未来 {horizon_hours} 小时内水质{trend_word}，"
        f"综合风险分 {score:.1f}，判定等级【{level}】。"
    )

    # 2) 主控因子
    if limiting and limiting in trends:
        t = trends[limiting]
        status = {1.0: "已进入危险区间", 0.5: "处于警戒区间", 0.0: "仍在适宜区间"}[sev[limiting]]
        parts.append(f"主控风险因子为{t.label}：当前 {_fmt(t.current, t.unit)}，{status}。")
    else:
        parts.append("各项常规指标均处于适宜区间。")

    # 3) 危险 / 警戒项枚举
    danger = [trends[k].label for k, v in sev.items() if v == 1.0 and k in trends]
    warning = [trends[k].label for k, v in sev.items() if v == 0.5 and k in trends]
    if danger:
        parts.append(f"其中{('、'.join(danger))}已达到危险级别。")
    if warning:
        parts.append(f"{('、'.join(warning))}处于警戒区间。")

    # 4) 外推预测：只讲截断后的窗口，避免给出"72 小时后溶氧 0.00"这类荒谬数字
    do_proj = projections.get("dissolved_oxygen")
    do_falling = do_trend is not None and do_trend.direction == "下降"
    if do_proj and do_falling:
        if do_proj.hours_to_threshold is not None:
            parts.append(
                f"按近段回归速率 {do_trend.slope_per_hour:+.3f} mg/L/小时外推，"
                f"预计 {do_proj.hours_to_threshold:.1f} 小时后溶氧将跌破 "
                f"{do_proj.threshold_used:.2f} mg/L {do_proj.threshold_label}。"
            )
        else:
            parts.append(
                f"未来 {do_proj.horizon_used_hours:.0f} 小时内溶氧预计降至 "
                f"{do_proj.projected_extreme:.2f} mg/L。"
            )
    elif do_proj and do_trend is not None and do_trend.direction == "上升":
        parts.append(
            f"未来 {do_proj.horizon_used_hours:.0f} 小时内溶氧预计回升至 "
            f"{do_proj.projected_at_horizon:.2f} mg/L。"
        )

    am_trend = trends.get("ammonia_n")
    am_proj = projections.get("ammonia_n")
    if am_proj and am_trend and am_trend.direction == "上升" and sev.get("ammonia_n", 0) > 0:
        if am_proj.hours_to_threshold is not None:
            parts.append(
                f"氨氮按当前速率预计 {am_proj.hours_to_threshold:.1f} 小时后触及 "
                f"{am_proj.threshold_used:.2f} mg/L {am_proj.threshold_label}。"
            )
        else:
            parts.append(
                f"氨氮在 {am_proj.horizon_used_hours:.0f} 小时内预计升至 "
                f"{am_proj.projected_extreme:.2f} mg/L。"
            )

    # 5) 协同风险提示（这是"多因子联合"的价值点，不是单指标复读）
    de = sev.get("dissolved_oxygen", 0.0)
    # 只点名本次**真正测到且超标**的指标：缺失的探头不能在文案里被说成"超标"
    offenders = [
        INDICATOR_META[k]["label"]
        for k in ("ammonia_n", "nitrite_n")
        if sev.get(k, 0.0) > 0
    ]
    offender_txt = "、".join(offenders)
    if de > 0 and offenders:
        if do_falling:
            parts.append(
                f"溶氧持续下降与{offender_txt}超标叠加，会显著削弱鱼类鳃组织的耐受力；"
                "夜间至凌晨（约 2–6 时）藻类停止产氧，泛塘风险将明显升高。"
            )
        else:
            parts.append(
                f"溶氧已偏低且{offender_txt}超标，会从呼吸与解毒两个环节同时加重鱼类负担，"
                "应优先处理。"
            )
    elif de > 0 and do_falling:
        parts.append("溶氧呈下降趋势，低值多出现在夜间至凌晨，需重点关注 2–6 时的溶氧谷值。")
    elif de > 0:
        parts.append("溶氧当前偏低但未见明显下降趋势，需保持增氧设备待命。")

    # 6) 红线说明
    if redline_hits:
        parts.append(f"已触发红线规则：{('、'.join(redline_hits))}，需立即人工干预。")

    # 7) 数据置信度声明
    if len(points) < MIN_RELIABLE_POINTS:
        parts.append(
            f"（置信度提示：本次仅 {len(points)} 条有效记录，少于 {MIN_RELIABLE_POINTS} 条，"
            f"趋势与外推结果仅供粗略参考。）"
        )
    if projections:
        parts.append(
            f"（说明：数值外推基于线性回归，仅前 {MAX_EXTRAPOLATION_HOURS:.0f} 小时具有参考意义，"
            f"接入 LSTM 时序模型后将替换为真实预测曲线。）"
        )

    return "".join(parts)


# --------------------------------------------------------------------------- #
# 七、核心计算
# --------------------------------------------------------------------------- #

def _resolve_interval(points: list[WaterQualityPoint],
                      fallback_minutes: int) -> tuple[float, str, list[str]]:
    """确定采样间隔：优先由 timestamp 推断，否则使用入参。"""
    warnings: list[str] = []
    stamps = [p.timestamp for p in points]
    if all(s is not None for s in stamps):
        deltas = [
            (stamps[i + 1] - stamps[i]).total_seconds() / 60.0  # type: ignore[operator]
            for i in range(len(stamps) - 1)
        ]
        deltas = [d for d in deltas if d > 0]
        if deltas:
            median = sorted(deltas)[len(deltas) // 2]
            if max(deltas) - min(deltas) > 0.5 * median:
                warnings.append(
                    f"采样间隔不均匀（{min(deltas):.0f}–{max(deltas):.0f} 分钟），"
                    f"已按中位数 {median:.0f} 分钟处理，趋势斜率可能有偏差。"
                )
            return median / 60.0, "timestamp", warnings
    return fallback_minutes / 60.0, "parameter", warnings


def _predict_series(payload: WaterRiskInput) -> WaterRiskOutput:
    """规则基线核心：阈值判定 + 回归趋势 + 线性外推。

    >>> 接入 LSTM 时，本函数是唯一需要替换的地方。
    """
    points = payload.readings
    profile = SPECIES_PROFILES[payload.species]

    interval_hours, interval_source, quality_warnings = _resolve_interval(
        points, payload.interval_minutes
    )

    # ---- 1. 组装序列（缺失指标直接跳过，后续权重自动归一化） ----
    candidate: dict[str, list[float]] = {}
    for key in BASE_WEIGHTS:
        vals = [getattr(p, key) for p in points]
        if any(v is None for v in vals):
            quality_warnings.append(f"指标 {INDICATOR_META[key]['label']} 存在缺失值，已整列剔除。")
            continue
        candidate[key] = [float(v) for v in vals]  # type: ignore[arg-type]

    missing = [INDICATOR_META[k]["label"] for k in BASE_WEIGHTS if k not in candidate]
    series = candidate

    # ---- 2. 逐指标：当前状态 = 最近 3 点均值（抑制单点抖动） ----
    sev: dict[str, float] = {}
    sev_label: dict[str, str] = {}
    current: dict[str, float] = {}
    for key, vals in series.items():
        tail = vals[-3:] if len(vals) >= 3 else vals
        current[key] = sum(tail) / len(tail)
        sev[key], sev_label[key] = _severity(key, current[key], profile)

    # ---- 3. 趋势 ----
    trends: dict[str, IndicatorTrend] = {}
    slopes: dict[str, float] = {}
    for key, vals in series.items():
        slope = _slope_per_hour(vals, interval_hours)
        slopes[key] = slope
        meta = INDICATOR_META[key]
        first = vals[0]
        trends[key] = IndicatorTrend(
            indicator=key,
            label=meta["label"],
            unit=meta["unit"],
            current=round(vals[-1], 3),
            mean=round(sum(vals) / len(vals), 3),
            minimum=round(min(vals), 3),
            maximum=round(max(vals), 3),
            slope_per_hour=round(slope, 5),
            direction=_determine_direction(slope, sum(vals) / len(vals), payload.horizon_hours),
            change_pct=round((vals[-1] - first) / first * 100.0, 2) if first else 0.0,
        )

    # ---- 4. 外推 ----
    projections = {
        key: _project(key, vals[-1], slopes[key], payload.horizon_hours)
        for key, vals in series.items()
    }

    # ---- 5. 权重归一化 ----
    total_w = sum(BASE_WEIGHTS[k] for k in series)
    weights = {k: BASE_WEIGHTS[k] / total_w for k in series} if total_w else {}

    # ---- 6. 风险分 ----
    base = sum(weights[k] * sev[k] for k in series)
    sev_matrix = [
        {k: _severity(k, series[k][i], profile)[0] for k in series}
        for i in range(len(points))
    ]
    run = _longest_exceed_run(sev_matrix)
    persistence = min(PERSISTENCE_CAP, 1.0 + 0.1 * (run - 1)) if run >= 1 else 1.0
    score = max(0.0, min(100.0, base * persistence * 100.0))

    # ---- 7. 规则 ----
    redline_hits = _check_redlines(current, series, interval_hours, projections)
    rule_hits = redline_hits + _check_normal_rules(series, slopes, sev, profile)

    # ---- 8. 主控因子 ----
    limiting = max(series, key=lambda k: weights[k] * sev[k]) if series else None
    if limiting is not None and sev[limiting] == 0.0:
        limiting = None  # 全部适宜时不宣称存在主控因子

    # ---- 9. 等级判定：任一越界即升级，加权分只表征量级 ----
    any_danger = any(v == 1.0 for v in sev.values())
    any_warning = any(v == 0.5 for v in sev.values())
    if redline_hits or any_danger or score >= SCORE_CRITICAL:
        level: str = "高危"
    elif any_warning or score >= SCORE_WARNING:
        level = "预警"
    else:
        level = "正常"

    # ---- 10. 组装输出 ----
    span_hours = (len(points) - 1) * interval_hours
    if len(points) < MIN_RELIABLE_POINTS:
        quality_warnings.append(
            f"有效记录 {len(points)} 条，少于建议的 {MIN_RELIABLE_POINTS} 条，趋势结论置信度低。"
        )

    summary = _build_summary(
        level=level, score=score, horizon_hours=payload.horizon_hours,
        trends=trends, projections=projections, sev=sev, limiting=limiting,
        points=points, redline_hits=redline_hits,
    )

    do_proj = projections.get("dissolved_oxygen")
    return WaterRiskOutput(
        tool="water_risk_predict",
        status="ok",
        pond_id=payload.pond_id,
        risk_level=level,  # type: ignore[arg-type]
        risk_score=round(score, 2),
        horizon_hours=payload.horizon_hours,
        extrapolation_horizon_hours=round(
            min(float(payload.horizon_hours), MAX_EXTRAPOLATION_HOURS), 2
        ),
        species=payload.species,
        degradation_summary=summary,
        limiting_factor=limiting,
        limiting_factor_label=INDICATOR_META[limiting]["label"] if limiting else None,
        projected_do_alert_eta_hours=do_proj.hours_to_threshold if do_proj else None,
        projected_do_alert_threshold=do_proj.threshold_used if do_proj else None,
        projected_do_alert_label=do_proj.threshold_label if do_proj else None,
        exceeded_indicators=[trends[k].label for k in series if sev[k] > 0],
        rule_hits=rule_hits,
        redline_hits=redline_hits,
        severity_map={k: sev[k] for k in series},
        effective_weights={k: round(v, 6) for k, v in weights.items()},
        trends=trends,
        projections=projections,
        data_quality=DataQuality(
            n_points=len(points),
            interval_minutes=int(round(interval_hours * 60)),
            interval_source=interval_source,  # type: ignore[arg-type]
            span_hours=round(span_hours, 2),
            missing_indicators=missing,
            warnings=quality_warnings,
        ),
        degraded_mode=True,
        projection_method="linear_extrapolation_baseline",
        error_code=None,
        error_message=None,
    )


# --------------------------------------------------------------------------- #
# 八、对外工具函数
# --------------------------------------------------------------------------- #

def water_risk_predict(
    readings: list[dict[str, Any] | WaterQualityPoint],
    pond_id: str = "unknown",
    interval_minutes: int = 60,
    horizon_hours: int = 72,
    species: str = "鲤科",
) -> dict[str, Any]:
    """【工具】水质时序风险预测。

    入参
    ----
    readings : list[dict | WaterQualityPoint]
        按时间升序的水质记录，至少 3 条。每条支持的键：
        timestamp（可选，ISO 8601）、water_temp（℃）、ph、
        dissolved_oxygen（mg/L）、ammonia_n（mg/L）、nitrite_n（mg/L，可缺失）。
    pond_id : str
        池塘编号，默认 "unknown"。
    interval_minutes : int
        采样间隔（分钟），默认 60。若记录自带 timestamp 会被自动覆盖。
    horizon_hours : int
        预测窗口（小时），默认 72，上限 168。
    species : str
        品种，取值 "鲤科" / "罗非鱼" / "虾蟹"，默认 "鲤科"。

    返回
    ----
    dict
        可直接 JSON 序列化的结果字典。`status` 为 "ok" 或 "error"；
        出错时不抛异常，而是在 error_code / error_message 中说明，
        这样 Agent 循环不会因为一次工具失败而中断。

    Examples:
        >>> series = [{"water_temp": 28.0, "ph": 7.5, "dissolved_oxygen": 6.0,
        ...            "ammonia_n": 0.1, "nitrite_n": 0.05} for _ in range(24)]
        >>> out = water_risk_predict(series, pond_id="POND-01")
        >>> out["risk_level"]
        '正常'
    """
    try:
        payload = WaterRiskInput(
            pond_id=pond_id,
            readings=readings,  # type: ignore[arg-type]
            interval_minutes=interval_minutes,
            horizon_hours=horizon_hours,
            species=species,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        return {
            "tool": "water_risk_predict",
            "status": "error",
            "pond_id": pond_id,
            "risk_level": None,
            "risk_score": None,
            "error_code": "INVALID_INPUT",
            "error_message": _format_validation_error(exc),
            "degraded_mode": True,
        }

    try:
        return _predict_series(payload).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 —— 工具层必须兜住所有异常
        return {
            "tool": "water_risk_predict",
            "status": "error",
            "pond_id": pond_id,
            "risk_level": None,
            "risk_score": None,
            "error_code": "COMPUTE_FAILED",
            "error_message": f"{type(exc).__name__}: {exc}",
            "degraded_mode": True,
        }


def _format_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 的报错压成一句人话，便于回传给 LLM 让它自我修正。"""
    items = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"])
        items.append(f"{loc}: {err['msg']}")
    return "入参校验失败 → " + "；".join(items)
