"""水质模板库 —— 一组「能直接导入、也能直接跑」的水体场景数据。

**为什么要有这个东西。** 现场演示真正卡住的从来不是算法，是"手上没数据"：
评委想看氨氮超标会怎么判，你总不能现场编 24 行数字；农户想比较自己塘口和
"正常塘"差多少，也没有基准可对。所以这里把常见水体形态固化成一册模板，
每个模板都**同时**满足两个用途：

1. **当数据用** —— ``templates/<key>.csv`` 是真实文件，可以直接上传、
   粘贴、或用 Excel 打开改几格再传回来；
2. **当基准用** —— 每个模板都写死了一个 ``expected_risk``，
   ``tests/unit/test_water_templates.py`` 会拿规则引擎真跑一遍并断言等级相符。
   模板不是"随手编的数字"，而是**被测试钉住的、规则引擎必然判成该等级的样例**。

**表头风格刻意不统一。** 真实导出工具的表头五花八门（中文、全大写、
带单位、甚至把 Dissolved Oxygen 拼成 Disolved），``webui/table.py`` 的别名层
就是为这些坑写的。模板里有五种风格，其中 ``upper_en`` 保留了数据集里
原样存在的拼写错误 —— 让"导入"这条路在演示时真的被走到。

**数值全部确定性生成**：没有随机数、没有时钟依赖，时间基点是每个模板
自带的历史日期。同样的代码在谁的机器上跑出来都是同一份字节，
``scripts/build_water_templates.py --check`` 才能拿它做一致性校验。
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .tools.water_risk import WaterQualityPoint

__all__ = [
    "HEADER_STYLES",
    "TEMPLATES_DIR",
    "TEMPLATE_INTERVAL_MINUTES",
    "WaterTemplate",
    "TemplateSpec",
    "all_templates",
    "catalog_markdown",
    "catalog_rows",
    "get_template",
    "render_csv",
    "template_keys",
    "template_path",
    "write_all",
]


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 模板 CSV 的落地目录。与 ``webui/table.py`` 的 ``EXAMPLE_TABLE`` 不同，
#: 这里是一整个目录：模板要能被用户当成"一批可选文件"看待。
TEMPLATES_DIR: Path = Path(__file__).resolve().parents[2] / "templates"

#: 全部模板统一 1 小时一个点、24 个点（08:00 到次日 07:00）。
#: 24 点正好等于 ``water_risk.MIN_RELIABLE_POINTS``，趋势结论不会被判低置信度。
TEMPLATE_INTERVAL_MINUTES = 60
_POINTS = 24
_START_HOUR = 8.0

#: 表头风格 → (时间列, 水温, pH, 溶氧, 氨氮, 亚硝酸盐)。
#:
#: ``upper_en`` 里的 ``DISOLVED OXYGEN`` 是数据集 ``IoTPond10.csv`` 里
#: 原样存在的拼写错误（少一个 s），别名表认得它 —— 故意留着，用来验证
#: "导入别人的导出文件"这条路真的走得通。
HEADER_STYLES: dict[str, tuple[str, ...]] = {
    "canonical": (
        "timestamp", "water_temp", "ph", "dissolved_oxygen", "ammonia_n", "nitrite_n",
    ),
    "units_en": (
        "timestamp", "water_temp_c", "ph",
        "dissolved_oxygen_mg_l", "ammonia_n_mg_l", "nitrite_n_mg_l",
    ),
    "upper_en": (
        "TIMESTAMP", "TEMPERATURE", "PH",
        "DISOLVED OXYGEN", "AMMONIA_N", "NITRITE_N",
    ),
    "zh": (
        "采集时间", "水温", "酸碱度", "溶解氧", "氨氮", "亚硝酸盐",
    ),
    "zh_units": (
        "采集时间", "水温(C)", "pH值", "溶氧(mg/L)", "氨氮(mg/L)", "亚硝酸盐(mg/L)",
    ),
}

_INDICATORS: tuple[str, ...] = (
    "water_temp", "ph", "dissolved_oxygen", "ammonia_n", "nitrite_n",
)

#: 各指标的保留小数位。溶氧/pH/水温写两位就够（探头精度也就这样），
#: 氨氮与亚硝酸盐保留三位 —— 标准限值本身就落在小数点后第三位。
_DECIMALS: dict[str, int] = {
    "water_temp": 2, "ph": 2, "dissolved_oxygen": 2,
    "ammonia_n": 3, "nitrite_n": 3,
}


# --------------------------------------------------------------------------- #
# 曲线工具：所有模板的数值都由这三个函数拼出来，保证可复现
# --------------------------------------------------------------------------- #

def _curve(base: float, amp: float, peak_hour: float,
           start_hour: float = _START_HOUR, n: int = _POINTS,
           drift: float = 0.0) -> list[float]:
    """一条以 24 小时为周期的余弦日变化曲线，叠加一个全天线性漂移。

    水体指标的日内形态基本就是这个形状：水温、pH、溶氧都在午后（14–16 点）
    见顶，凌晨（3–5 点）见底。``peak_hour`` 控制峰值出现的钟点。

    ``drift`` 是"这一天本身在往哪走"——比如暴雨降温、氨氮持续累积，
    用日变化叠漂移比纯线性更像真实记录。
    """
    span = max(1, n - 1)
    out: list[float] = []
    for i in range(n):
        hour = start_hour + i
        value = base + amp * math.cos(2.0 * math.pi * (hour - peak_hour) / 24.0)
        out.append(value + drift * (i / span))
    return out


def _ramp(start: float, end: float, n: int = _POINTS,
          ease: float = 1.0) -> list[float]:
    """从 ``start`` 单调走到 ``end``。``ease > 1`` 表示"后段涨（跌）得更快"。"""
    span = max(1, n - 1)
    return [start + (end - start) * ((i / span) ** ease) for i in range(n)]


def _add(*series: Sequence[float]) -> list[float]:
    """逐点相加。用来把"日变化"和"趋势"叠在一起。"""
    return [sum(part[i] for part in series) for i in range(len(series[0]))]


# --------------------------------------------------------------------------- #
# 模板定义
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TemplateSpec:
    """一个模板的元信息 + 它的数值生成器。

    与 :class:`WaterTemplate` 分开，是因为"元信息"是给人看、给接口暴露的，
    而 ``build`` 是一次性的生成逻辑；把生成逻辑塞进 pydantic 模型会让
    模型序列化出一堆函数引用。
    """

    key: str
    name: str
    scenario: str
    species: Literal["鲤科", "罗非鱼", "虾蟹"]
    expected_risk: Literal["正常", "预警", "高危"]
    #: 这个模板想演示什么。写进 README 与页面提示，也是测试失败时的线索。
    focus: str
    tags: tuple[str, ...]
    header_style: str
    #: 数据起点。每个模板自带日期，让季节与数值对得上（冰封期就该在一月）。
    start: str
    build: Callable[[], dict[str, list[float]]]


def _specs() -> tuple[TemplateSpec, ...]:
    """全部模板。顺序 = 页面下拉框顺序 = README 表格顺序，从"正常"排到"高危"。"""

    def normal_summer() -> dict[str, list[float]]:
        return {
            "water_temp": _curve(27.5, 2.0, 15.0),
            "ph": _curve(8.00, 0.20, 16.0),
            "dissolved_oxygen": _curve(6.40, 0.70, 15.0),
            "ammonia_n": _ramp(0.060, 0.150, ease=1.3),
            "nitrite_n": _ramp(0.020, 0.060),
        }

    def normal_shrimp() -> dict[str, list[float]]:
        # 溶氧刻意压在 4.35~5.45：对虾蟹（档案下限 4.0）算正常，
        # 换成鲤科档案（下限 5.0）就是警戒。这一个模板同时演示了
        # "同一份水、换个品种档案结论就不同"，是品种下拉框存在的理由。
        return {
            "water_temp": _curve(28.0, 1.5, 16.0),
            "ph": _curve(8.05, 0.20, 15.0),
            "dissolved_oxygen": _curve(4.90, 0.55, 15.0),
            "ammonia_n": _ramp(0.040, 0.090),
            "nitrite_n": _ramp(0.010, 0.040),
        }

    def dawn_hypoxia() -> dict[str, list[float]]:
        # 溶氧从午后 6.5 一路耗到黎明前 2.8：夜间藻类只呼吸不光合，
        # 这是池塘最典型的缺氧形态。故意不用曲线拟合，逐点写死，
        # 免得公式一改就把"凌晨跌破危险线"这个关键事实改没了。
        do = [
            6.50, 6.42, 6.30, 6.15, 5.98, 5.80, 5.62, 5.45,
            5.28, 5.10, 4.92, 4.74, 4.55, 4.36, 4.18, 4.00,
            3.83, 3.66, 3.50, 3.34, 3.19, 3.05, 2.92, 2.80,
        ]
        return {
            "water_temp": _curve(28.5, 1.6, 16.0),
            "ph": _curve(7.95, 0.15, 15.0),
            "dissolved_oxygen": do,
            "ammonia_n": _ramp(0.100, 0.190),
            "nitrite_n": _ramp(0.040, 0.090),
        }

    def do_redline_crash() -> dict[str, list[float]]:
        # 跌破红线 1.5：泛塘临界。3 小时最大跌幅约 1.1 mg/L，
        # 刻意**不**触发"3 小时跌 2.0"那条趋势红线，好让这个模板单独验证
        # "当前值破红线"这一条规则。
        do = [
            5.20, 5.02, 4.82, 4.58, 4.30, 3.98, 3.62, 3.24,
            2.86, 2.50, 2.18, 1.92, 1.72, 1.58, 1.48, 1.42,
            1.38, 1.34, 1.31, 1.28, 1.26, 1.24, 1.22, 1.20,
        ]
        return {
            "water_temp": _curve(29.0, 1.5, 16.0),
            "ph": _curve(7.60, 0.10, 15.0),
            "dissolved_oxygen": do,
            "ammonia_n": _ramp(0.100, 0.180),
            "nitrite_n": _ramp(0.040, 0.090),
        }

    def ammonia_accumulation() -> dict[str, list[float]]:
        # 高温 + 高 pH + 氨氮累积：这是"分子氨占比升高"的教科书组合，
        # 会命中 UNIONIZED_AMMONIA_RISK 协同规则。
        return {
            "water_temp": _curve(31.0, 1.2, 16.0),
            "ph": _curve(8.65, 0.20, 16.0),
            "dissolved_oxygen": _curve(5.80, 0.50, 15.0),
            "ammonia_n": _ramp(0.280, 0.720, ease=1.2),
            "nitrite_n": _ramp(0.050, 0.090),
        }

    def nitrite_spike() -> dict[str, list[float]]:
        # 亚硝酸盐落在危险带（>0.3）但没破红线（0.5）：验证"危险带"这一档。
        return {
            "water_temp": _curve(25.5, 1.5, 16.0),
            "ph": _curve(7.85, 0.15, 15.0),
            "dissolved_oxygen": _curve(6.00, 0.50, 15.0),
            "ammonia_n": _ramp(0.100, 0.190),
            "nitrite_n": _ramp(0.120, 0.440, ease=1.1),
        }

    def ph_high_alkaline() -> dict[str, list[float]]:
        # 蓝藻旺发/碱度大的塘，午后 pH 能上 9。振幅压到 0.15，
        # 保证峰值 9.40 不碰 9.5 那条红线 —— 这里要演示的是"危险带"不是红线。
        return {
            "water_temp": _curve(29.5, 1.3, 16.0),
            "ph": _curve(9.25, 0.15, 16.0),
            "dissolved_oxygen": _curve(7.20, 1.10, 15.0),
            "ammonia_n": _ramp(0.060, 0.120),
            "nitrite_n": _ramp(0.020, 0.050),
        }

    def winter_ice() -> dict[str, list[float]]:
        # 冰封期：水温 4℃ 上下，早已跌出水温危险线（15℃ 以下即危险）；
        # 但低温溶氧反而充足，所以"高危"来自水温而不是缺氧。
        return {
            "water_temp": _curve(4.80, 1.00, 14.0),
            "ph": _curve(7.60, 0.10, 15.0),
            "dissolved_oxygen": _curve(8.50, 0.60, 14.0),
            "ammonia_n": _ramp(0.080, 0.160),
            "nitrite_n": _ramp(0.030, 0.070),
        }

    def low_do_warning() -> dict[str, list[float]]:
        # 溶氧全程在 3.0–5.0 之间：够不上危险（<3.0），但已经不是适宜（≥5.0）。
        return {
            "water_temp": _curve(26.0, 1.5, 16.0),
            "ph": _curve(7.80, 0.15, 15.0),
            "dissolved_oxygen": _curve(4.30, 0.80, 16.0),
            "ammonia_n": _ramp(0.090, 0.170),
            "nitrite_n": _ramp(0.040, 0.080),
        }

    def temp_drop_after_rain() -> dict[str, list[float]]:
        # 暴雨后 24 小时内水温从 29.4 掉到 18.4、pH 从 8.25 掉到 7.25。
        # 水温跌进警戒带（15–20），pH 波动幅度超过 1.0 个单位。
        return {
            "water_temp": _add(_ramp(29.4, 18.4, ease=0.85),
                               _curve(0.0, 0.60, 15.0)),
            "ph": _add(_ramp(8.25, 7.25),
                       _curve(0.0, 0.10, 16.0)),
            "dissolved_oxygen": _curve(5.60, 0.70, 15.0),
            "ammonia_n": _ramp(0.100, 0.190),
            "nitrite_n": _ramp(0.030, 0.070),
        }

    def ammonia_climbing() -> dict[str, list[float]]:
        # 氨氮从 0.12 爬到 0.38：越过适宜上限（0.2）但没到危险（0.5）。
        return {
            "water_temp": _curve(28.5, 1.4, 16.0),
            "ph": _curve(8.25, 0.20, 16.0),
            "dissolved_oxygen": _curve(5.90, 0.60, 15.0),
            "ammonia_n": _ramp(0.120, 0.380, ease=1.6),
            "nitrite_n": _ramp(0.040, 0.090),
        }

    def spring_low_temp() -> dict[str, list[float]]:
        # 早春：水温 16.6，落在警戒带（15–20）里，吃食与免疫力都上不来。
        return {
            "water_temp": _curve(17.5, 1.8, 15.0),
            "ph": _curve(7.90, 0.15, 15.0),
            "dissolved_oxygen": _curve(7.00, 0.60, 15.0),
            "ammonia_n": _ramp(0.070, 0.150),
            "nitrite_n": _ramp(0.020, 0.060),
        }

    return (
        TemplateSpec(
            key="normal_summer",
            name="夏季正常塘（鲤科）",
            scenario="7 月中旬，晴，水温 26–30℃，溶氧充沛，氨氮与亚硝酸盐都在适宜区间。",
            species="鲤科",
            expected_risk="正常",
            focus="基准模板：用来对比「我的塘口比正常差多少」。",
            tags=("正常", "鲤科", "夏季"),
            header_style="units_en",
            start="2026-07-14 08:00:00",
            build=normal_summer,
        ),
        TemplateSpec(
            key="normal_shrimp",
            name="虾蟹塘正常（对虾/河蟹）",
            scenario="8 月上旬，水温 26.5–29.5℃，溶氧 4.3–5.5 mg/L。这是个「对虾蟹正常、"
            "对鲤科偏紧」的溶氧水平——换品种档案结论就会变。",
            species="虾蟹",
            expected_risk="正常",
            focus="品种维度：同一份水，换品种档案结论可能不同（这里溶氧 4.35 对虾蟹正常，对鲤科已警戒）。",
            tags=("正常", "虾蟹", "夏季"),
            header_style="zh",
            start="2026-08-05 08:00:00",
            build=normal_shrimp,
        ),
        TemplateSpec(
            key="dawn_hypoxia",
            name="黎明前缺氧（夜间耗氧）",
            scenario="7 月下旬晴热无风，溶氧从午后 6.5 一路耗到黎明前 2.8 mg/L。",
            species="鲤科",
            expected_risk="高危",
            focus="最典型的泛塘前兆：白天看着没事，天亮前最危险。",
            tags=("高危", "鲤科", "溶氧", "夜间"),
            header_style="canonical",
            start="2026-07-20 08:00:00",
            build=dawn_hypoxia,
        ),
        TemplateSpec(
            key="do_redline_crash",
            name="溶氧破红线（泛塘临界）",
            scenario="溶氧在 18 小时内从 5.2 掉到 1.2 mg/L，已跌破 1.5 的泛塘临界值。",
            species="鲤科",
            expected_risk="高危",
            focus="红线规则：任何一项破红线即判高危，不看加权分。",
            tags=("高危", "鲤科", "溶氧", "红线"),
            header_style="upper_en",
            start="2026-07-21 08:00:00",
            build=do_redline_crash,
        ),
        TemplateSpec(
            key="ammonia_accumulation",
            name="高温高 pH 氨氮累积",
            scenario="7 月末高温，pH 8.5–8.9，氨氮爬到 0.72 mg/L（危险带，未破 1.0 红线）。",
            species="鲤科",
            expected_risk="高危",
            focus="分子氨协同：pH 与水溫同时偏高时氨氮毒性放大。",
            tags=("高危", "鲤科", "氨氮", "高温"),
            header_style="zh_units",
            start="2026-07-25 08:00:00",
            build=ammonia_accumulation,
        ),
        TemplateSpec(
            key="nitrite_spike",
            name="亚硝酸盐超标",
            scenario="9 月中旬，亚硝酸盐从 0.12 升到 0.44 mg/L（危险带，未破 0.5 红线）。",
            species="鲤科",
            expected_risk="高危",
            focus="亚硝酸盐单独超标：它能独立把等级推到高危。",
            tags=("高危", "鲤科", "亚硝酸盐"),
            header_style="units_en",
            start="2026-09-10 08:00:00",
            build=nitrite_spike,
        ),
        TemplateSpec(
            key="ph_high_alkaline",
            name="午后 pH 偏高（碱度大）",
            scenario="8 月中旬蓝藻旺发，pH 全天 9.1–9.4，溶氧因光合作用过饱和。",
            species="鲤科",
            expected_risk="高危",
            focus="pH 单指标进危险带（>9.0）；溶氧高不等于安全。",
            tags=("高危", "鲤科", "pH"),
            header_style="canonical",
            start="2026-08-18 08:00:00",
            build=ph_high_alkaline,
        ),
        TemplateSpec(
            key="winter_ice",
            name="冬季冰封期",
            scenario="1 月中旬封冰，水温 4℃ 上下；低温溶氧充足，但水温远低于危险线。",
            species="鲤科",
            expected_risk="高危",
            focus="反直觉案例：溶氧漂亮但水温致命，等级仍为高危。",
            tags=("高危", "鲤科", "水温", "冬季"),
            header_style="zh",
            start="2027-01-12 08:00:00",
            build=winter_ice,
        ),
        TemplateSpec(
            key="low_do_warning",
            name="溶氧偏低（警戒）",
            scenario="9 月初，溶氧全天在 3.5–5.1 mg/L，够不上危险但已离开适宜区。",
            species="鲤科",
            expected_risk="预警",
            focus="警戒带：还没到要抢救的程度，但必须开始干预。",
            tags=("预警", "鲤科", "溶氧"),
            header_style="units_en",
            start="2026-09-02 08:00:00",
            build=low_do_warning,
        ),
        TemplateSpec(
            key="temp_drop_after_rain",
            name="暴雨降温后",
            scenario="9 月下旬一场暴雨，24 小时内水温 29.4→18.4℃、pH 8.25→7.25。",
            species="鲤科",
            expected_risk="预警",
            focus="应激场景：水温跌进警戒带 + pH 剧烈波动。",
            tags=("预警", "鲤科", "水温", "pH", "天气"),
            header_style="zh_units",
            start="2026-09-22 08:00:00",
            build=temp_drop_after_rain,
        ),
        TemplateSpec(
            key="ammonia_climbing",
            name="氨氮缓慢爬升",
            scenario="8 月末投料量大，氨氮从 0.12 一路爬到 0.38 mg/L，尚未破危险线。",
            species="鲤科",
            expected_risk="预警",
            focus="趋势预警：数值还好看，但方向不对。",
            tags=("预警", "鲤科", "氨氮", "趋势"),
            header_style="canonical",
            start="2026-08-28 08:00:00",
            build=ammonia_climbing,
        ),
        TemplateSpec(
            key="spring_low_temp",
            name="早春水温偏低",
            scenario="3 月中旬，水温 15.7–19.3℃，落在警戒带；其它指标干净。",
            species="鲤科",
            expected_risk="预警",
            focus="季节性警戒：水瘦温低，投料与用药都要改口径。",
            tags=("预警", "鲤科", "水温", "春季"),
            header_style="zh",
            start="2027-03-18 08:00:00",
            build=spring_low_temp,
        ),
    )


_SPECS: tuple[TemplateSpec, ...] = _specs()
_SPEC_BY_KEY: dict[str, TemplateSpec] = {s.key: s for s in _SPECS}


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

class WaterTemplate(BaseModel):
    """一个水质模板的完整内容：元信息 + 可直接喂给工具层的读数。"""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(..., description="模板标识，同时是 CSV 文件名（不含扩展名）")
    name: str = Field(..., description="给人看的短名")
    scenario: str = Field(..., description="这个模板描述的水体形态")
    species: str = Field(..., description="建议的养殖品种档案")
    expected_risk: Literal["正常", "预警", "高危"] = Field(
        ..., description="规则引擎对该模板应给出的等级；由单元测试钉住",
    )
    focus: str = Field(..., description="这个模板用来演示什么")
    tags: list[str] = Field(default_factory=list)
    header_style: str = Field(..., description="CSV 表头风格，见 HEADER_STYLES")
    start: datetime = Field(..., description="数据起点")
    interval_minutes: int = Field(default=TEMPLATE_INTERVAL_MINUTES)
    readings: list[WaterQualityPoint] = Field(default_factory=list)

    @property
    def csv_filename(self) -> str:
        return f"{self.key}.csv"

    @property
    def n_points(self) -> int:
        return len(self.readings)

    def to_csv(self) -> str:
        """渲染成 CSV 文本（与 ``templates/<key>.csv`` 逐字节一致）。"""
        return render_csv(self)

    def summary_line(self) -> str:
        """下拉框里那一行。带上期望等级，选之前就知道会看到什么。"""
        return f"{self.name}（{self.species} · 预期{self.expected_risk}）"


# --------------------------------------------------------------------------- #
# 生成与渲染
# --------------------------------------------------------------------------- #

def _round(indicator: str, value: float) -> float:
    return round(value, _DECIMALS.get(indicator, 2))


def _readings(spec: TemplateSpec) -> list[WaterQualityPoint]:
    """把生成器吐出的列式数据组装成逐点读数。"""
    columns = spec.build()
    missing = [k for k in _INDICATORS if k not in columns]
    if missing:  # pragma: no cover —— 模板定义错了才会走到
        raise ValueError(f"模板 {spec.key} 缺少指标列：{missing}")
    lengths = {len(columns[k]) for k in _INDICATORS}
    if lengths != {_POINTS}:  # pragma: no cover —— 同上
        raise ValueError(f"模板 {spec.key} 的点数不一致：{sorted(lengths)}，应为 {_POINTS}")

    start = datetime.fromisoformat(spec.start)
    points: list[WaterQualityPoint] = []
    for i in range(_POINTS):
        points.append(
            WaterQualityPoint(
                timestamp=start + timedelta(minutes=TEMPLATE_INTERVAL_MINUTES * i),
                **{k: _round(k, columns[k][i]) for k in _INDICATORS},
            )
        )
    return points


def _template(spec: TemplateSpec) -> WaterTemplate:
    return WaterTemplate(
        key=spec.key,
        name=spec.name,
        scenario=spec.scenario,
        species=spec.species,
        expected_risk=spec.expected_risk,
        focus=spec.focus,
        tags=list(spec.tags),
        header_style=spec.header_style,
        start=datetime.fromisoformat(spec.start),
        interval_minutes=TEMPLATE_INTERVAL_MINUTES,
        readings=_readings(spec),
    )


def template_keys() -> tuple[str, ...]:
    """全部模板 key，顺序与页面下拉框一致。"""
    return tuple(s.key for s in _SPECS)


def all_templates() -> tuple[WaterTemplate, ...]:
    """全部模板（每次调用重新构造；模板是纯函数输出，不存在"陈旧"问题）。"""
    return tuple(_template(s) for s in _SPECS)


def get_template(key: str) -> WaterTemplate:
    """按 key 取模板。取不到时抛 :class:`KeyError`，消息里列出全部可用 key。

    这里不返回 ``None``：调用方（接口、页面）拿到 ``None`` 只能自己编一个
    错误消息，不如让异常把可选值一并带出来。
    """
    spec = _SPEC_BY_KEY.get((key or "").strip())
    if spec is None:
        available = "、".join(template_keys())
        raise KeyError(f"没有名为 {key!r} 的水质模板。可用模板：{available}")
    return _template(spec)


def template_path(key: str, directory: str | Path | None = None) -> Path:
    """模板 CSV 的落地路径（不保证文件已存在，写盘见 :func:`write_all`）。"""
    get_template(key)  # 校验 key
    root = Path(directory) if directory is not None else TEMPLATES_DIR
    return root / f"{key}.csv"


def render_csv(template: WaterTemplate) -> str:
    """渲染模板 CSV。

    换行用 ``\\n``、不带 BOM、末尾留一个换行 —— 与
    ``examples/pond_demo_water.csv`` 保持一致。用 ``\\r\\n`` 会让同一份模板
    在 Windows 与 Linux 上产出不同字节，``--check`` 就没法当一致性校验用了。
    """
    header = HEADER_STYLES[template.header_style]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for point in template.readings:
        stamp = point.timestamp.strftime("%Y-%m-%d %H:%M:%S") if point.timestamp else ""
        row: list[Any] = [stamp]
        for indicator in _INDICATORS:
            value = getattr(point, indicator)
            row.append("" if value is None else f"{_round(indicator, value):.{_DECIMALS[indicator]}f}")
        writer.writerow(row)
    return buffer.getvalue()


def write_all(directory: str | Path | None = None) -> list[Path]:
    """把全部模板写成 CSV（覆盖式），返回写出的路径列表。"""
    root = Path(directory) if directory is not None else TEMPLATES_DIR
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for template in all_templates():
        path = root / template.csv_filename
        path.write_text(render_csv(template), encoding="utf-8", newline="")
        written.append(path)
    return written


def materialize(key: str, directory: str | Path | None = None) -> Path:
    """确保某个模板的 CSV 已落盘，返回它的路径。

    仓库里 ``templates/`` 是**跟着代码一起提交的**，正常情况下文件一定在。
    这里仍然做一次"缺失就现写"，是因为使用者可能只拷走了 ``src/`` 而没带
    ``templates/``（例如把这个模块搬进别的项目），此时「用该模板填充」
    不该因为一个本该存在的数据文件而失败。

    注意**只补缺失、不覆盖已存在**：磁盘上内容过期是
    ``scripts/build_water_templates.py --check`` 该报出来的事，
    在这里静默覆盖只会掩盖"忘了重跑生成脚本"。
    """
    path = template_path(key, directory)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_csv(get_template(key)), encoding="utf-8", newline="")
    return path


# --------------------------------------------------------------------------- #
# 目录（给接口与 README 用）
# --------------------------------------------------------------------------- #

def catalog_rows() -> list[dict[str, Any]]:
    """模板目录的扁平结构，供 API 序列化与 README 表格共用一处口径。"""
    rows: list[dict[str, Any]] = []
    for template in all_templates():
        rows.append({
            "key": template.key,
            "name": template.name,
            "scenario": template.scenario,
            "species": template.species,
            "expected_risk": template.expected_risk,
            "focus": template.focus,
            "tags": list(template.tags),
            "header_style": template.header_style,
            "csv_filename": template.csv_filename,
            "n_points": template.n_points,
            "interval_minutes": template.interval_minutes,
            "start": template.start.strftime("%Y-%m-%d %H:%M"),
            "end": template.readings[-1].timestamp.strftime("%Y-%m-%d %H:%M")
            if template.readings and template.readings[-1].timestamp else "",
        })
    return rows


def catalog_markdown() -> str:
    """``templates/README.md`` 的正文。由 ``scripts/build_water_templates.py`` 生成。"""
    lines: list[str] = [
        "# 水质模板",
        "",
        "> 本目录由 `scripts/build_water_templates.py` 生成，**不要手工编辑**：",
        "> 数值源头是 `src/fisheries/water_templates.py`，改模板请改代码后重跑脚本。",
        "",
        "## 怎么用",
        "",
        "三种用法，选一个就行：",
        "",
        "1. **导入文件** —— 把任意 `<key>.csv` 上传到演示页的"
        "「水质时序表格」，或直接丢进你自己的工具链；",
        "2. **用模板填充** —— 演示页左栏选好模板后点「📥 用该模板填充」，"
        "会自动填文件、切换品种、清零粘贴框；",
        "3. **调接口** —— `GET /api/v1/config/templates` 拿目录，"
        "`GET /api/v1/config/templates/{key}/csv` 直接下载 CSV。",
        "",
        "每个模板都是 24 个点、1 小时间隔（08:00 到次日 07:00），"
        "正好达到 `water_risk.MIN_RELIABLE_POINTS`，趋势外推不会被判低置信度。",
        "",
        "## 表头风格",
        "",
        "模板的表头**故意不统一**，因为真实导出文件本来就不统一。"
        "`upper_en` 风格还保留了数据集里原样的拼写错误 `DISOLVED OXYGEN`：",
        "",
        "| 风格 | 表头 | 用到的模板 |",
        "| --- | --- | --- |",
    ]
    for style, header in HEADER_STYLES.items():
        users = "、".join(
            f"`{t.key}`" for t in all_templates() if t.header_style == style
        )
        lines.append(f"| `{style}` | `{','.join(header)}` | {users} |")

    lines += [
        "",
        "## 模板清单",
        "",
        "「预期等级」不是描述，是**被单元测试钉住的断言**："
        "`tests/unit/test_water_templates.py` 会真的调用规则引擎跑一遍，"
        "等级对不上就测试失败。",
        "",
        "| 文件 | 名称 | 品种 | 预期等级 | 演示点 | 数据起点 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in catalog_rows():
        lines.append(
            f"| `{row['csv_filename']}` | {row['name']} | {row['species']} | "
            f"**{row['expected_risk']}** | {row['focus']} | {row['start']} |"
        )

    lines += [
        "",
        "## 场景说明",
        "",
    ]
    for row in catalog_rows():
        lines.append(f"- **{row['name']}**（`{row['key']}`）：{row['scenario']}")

    lines += [
        "",
        "## 免责",
        "",
        "模板数值是**构造出来的演示数据**，不是任何真实塘口的实测记录，"
        "也不是标准原文里的数值。它只保证一件事："
        "在当前这套阈值档案下，规则引擎会给出表中那个等级。",
        "真实生产决策请以现场实测与有资质的水产技术人员意见为准。",
        "",
    ]
    return "\n".join(lines)
