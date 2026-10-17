"""水质时序表格的摄入层 —— 把「用户手里的表格」变成工具层能吃的水质读数。

这是整条链路的第一道关，也是最容易当场翻车的一关。原始数据集里的表头
已经把这几种坑凑齐了：

===============  ==========================================  ==========================
文件             表头                                        坑
===============  ==========================================  ==========================
IoTpond1.csv     ``Temperature (C)`` / ``PH``               单位写在列名里，且是**错的**
IoTPond10.csv    ``TEMPERATURE`` / ``DISOLVED OXYGEN``     全大写 + 拼写错误
IoTPond11.csv    ``Lenght`` / ``AMMONIA``                 又一个拼写错误
china_water_*.csv ``Dissolved_Oxygen_mg_L``                 下划线 + 单位后缀
user_export.xlsx  ``水温`` / ``溶氧`` / ``氨氮``             中文表头
===============  ==========================================  ==========================

``IoTpond*.csv`` 的氨氮列标着 ``Ammonia(g/ml)``，数值却是 ``0.45842`` —— g/ml 的话
一升水里有 458 克氨，那是化肥不是鱼塘。**标签是错的，数值单位是 mg/L**。
本模块不替用户改单位，只把矛盾**显式报出来**（见 :func:`unit_warnings`），
因为「列名标错」和「设备真的坏了」在现场看起来一模一样，必须由人确认。

浏览器端（``web/js/parse.js``）有一份职责相同的实现：用户在页面上传文件时先在本地
解析，成功后以 ``readings`` 走接口。本模块是服务端这一侧的实现，供库调用方、脚本与
测试使用 —— 两边都要对同一批表头坑负责，**改一处必须同步另一处**。

一个刻意的设计取舍：**只做精确别名匹配，不做模糊匹配**。
``Total_Phosphorus_mg_L`` 里含有 ``ph``，``Nitrate`` 与 ``Nitrite`` 只差三个字母 ——
模糊匹配会把总磷认成 pH、把硝酸盐认成亚硝酸盐。前者只是难看，
后者会**把无毒指标当成剧毒指标去报警**。所以认不出来的列一律列进
``ignored_columns`` 交给用户看，宁可漏认也不误认。
"""

from __future__ import annotations

import bisect
import csv
import io
import itertools
import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "CANONICAL_FIELDS",
    "COLUMN_ALIASES",
    "EXAMPLE_TABLE",
    "INDICATORS",
    "MIN_POINTS",
    "MISSING_SENTINELS",
    "REQUIRED_FIELDS",
    "ColumnMapping",
    "TableError",
    "TableIngest",
    "ingest_rows",
    "ingest_table",
    "map_columns",
    "normalize_header",
    "parse_timestamp",
    "read_table_rows",
    "to_float",
    "unit_warnings",
]


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

class TableError(Exception):
    """表格无法读入（编码/格式/缺列）时抛出。消息面向现场操作的人，不是给日志看的。"""


#: 工具层 :class:`~fisheries.tools.water_risk.WaterQualityPoint` 认识的指标字段。
INDICATORS: tuple[str, ...] = (
    "water_temp",
    "ph",
    "dissolved_oxygen",
    "ammonia_n",
    "nitrite_n",
)

#: 工具层的必填项。少任何一个，``water_risk_predict`` 直接抛 pydantic 校验错，
#: 与其让它在工具层炸，不如在读表时就说清楚缺哪列。
REQUIRED_FIELDS: tuple[str, ...] = (
    "water_temp",
    "ph",
    "dissolved_oxygen",
    "ammonia_n",
)

#: 能拿去跑趋势的最小点数。与 ``WaterRiskInput.readings`` 的 ``min_length=3``
#: 保持一致——不一致的话，页面会兴冲冲地调工具然后被校验错打回来。
MIN_POINTS = 3

#: 表格里可以被识别的全部规范字段（含时间相关的伪字段）。
CANONICAL_FIELDS: tuple[str, ...] = ("timestamp", "water_temp", "ph",
                                     "dissolved_oxygen", "ammonia_n", "nitrite_n",
                                     "nitrate_n")

#: 表头别名表。键是规范字段，值是可接受的写法。
#:
#: ``_date`` / ``_time`` 是内部伪字段：有些导出工具把日期和时间拆成两列
#: （数据集里的 ``Date``），单独一列都不是合法时间戳，需要合并后再解析。
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "createdat", "created", "timestamp", "datetime", "recordedat",
        "measurementtime", "sampletime", "采集时间", "时间戳", "记录时间",
    ),
    "_date": ("date", "dates", "日期", "采样日期"),
    "_time": ("time", "时刻", "采样时间"),
    "water_temp": (
        "temperature", "temperaturec", "temperaturef", "temp", "watertemp",
        "watertemperature", "watertemperaturec", "水温", "温度", "水体温度",
    ),
    "ph": ("ph", "phvalue", "phlevel", "酸碱度", "ph值", "氢离子浓度"),
    "dissolved_oxygen": (
        "do", "dissolvedoxygen", "disolvedoxygen", "dissolvedoxygenmgl",
        "dissolvedoxygengml", "do2", "溶氧", "溶解氧", "含氧量",
    ),
    "ammonia_n": (
        "ammonia", "ammonian", "ammonianmgl", "ammoniagml", "ammoniamgl",
        "nh3", "nh3n", "nh4", "nh4n", "氨氮", "氨态氮", "非离子氨",
    ),
    "nitrite_n": (
        "nitrite", "nitriten", "nitritenmgl", "nitritemgl", "no2", "no2n",
        "亚硝酸盐", "亚硝酸盐氮",
    ),
    "nitrate_n": (
        "nitrate", "nitraten", "nitratenmgl", "nitratemgl", "no3", "no3n",
        "硝酸盐", "硝酸盐氮",
    ),
}

#: 归一化时要丢掉的字符。注意 ``/`` 也要丢：``g/ml`` 归一化后是 ``gml``。
_STRIP_CHARS = " \t\r\n_-()（）[]【】.·,，:：/／\\|'\"*#%＋+"

#: 单位后缀。列名里带单位（``Dissolved Oxygen(g/ml)``、``Ammonia_N_mg_L``）时，
#: 先原样查一次别名，查不到再去掉单位后缀查第二次。
#: 顺序按长度降序，保证 ``nmgl`` 先于 ``mgl`` 被尝试。
UNIT_SUFFIXES: tuple[str, ...] = (
    "nmgl", "mgl", "mg/l".replace("/", ""), "gml", "ugl", "ugml",
    "ntu", "uscm", "mg", "c", "f",
)

#: 设备/导出工具用来表示「这一格没有数据」的哨兵值。
#: ``IoTPond10.csv`` 里的 ``-127`` 就是典型：它不是一个温度，是「传感器没读上来」。
MISSING_SENTINELS: frozenset[float] = frozenset({-127.0, -999.0, -9999.0, -99.0})

#: 文本形式的空值。
_MISSING_TOKENS: frozenset[str] = frozenset({
    "", "-", "--", "---", "na", "n/a", "null", "none", "nil", "nan",
    "?", "无", "空", "—", "－",
})

#: 指标合理量程（下限, 上限, 单位说明）。超出即怀疑单位或列映射错了。
_PLAUSIBLE_RANGE: dict[str, tuple[float, float, str]] = {
    "water_temp": (-5.0, 45.0, "℃"),
    "ph": (0.0, 14.0, ""),
    "dissolved_oxygen": (0.0, 20.0, "mg/L"),
    "ammonia_n": (0.0, 20.0, "mg/L"),
    "nitrite_n": (0.0, 20.0, "mg/L"),
    "nitrate_n": (0.0, 500.0, "mg/L"),
}

#: 单位量级远超养殖水体可能值时，提醒「标签与数值不符」。阈值取得很宽松，
#: 只抓量级错误（差 10 倍以上），不抓正常的超标。
_UNIT_SUSPECT_ABOVE: dict[str, float] = {
    "dissolved_oxygen": 20.0,
    "ammonia_n": 20.0,
    "nitrite_n": 20.0,
}

_INDICATOR_LABEL: dict[str, str] = {
    "water_temp": "水温",
    "ph": "pH",
    "dissolved_oxygen": "溶解氧",
    "ammonia_n": "氨氮",
    "nitrite_n": "亚硝酸盐",
    "nitrate_n": "硝酸盐",
}

#: 仓库自带的示例表格，用于现场演示的「加载示例数据」。
EXAMPLE_TABLE: Path = (
    Path(__file__).resolve().parents[2] / "examples" / "pond_demo_water.csv"
)


# --------------------------------------------------------------------------- #
# 表头归一化与匹配
# --------------------------------------------------------------------------- #

def normalize_header(name: str) -> str:
    """把表头压成可比较的形式：小写、去空白、去下划线/括号/斜杠等符号。

    >>> normalize_header("Dissolved Oxygen(g/ml)")
    'dissolvedoxygengml'
    >>> normalize_header("Water_Temperature_C")
    'watertemperaturec'
    >>> normalize_header("氨氮")
    '氨氮'
    """
    text = (name or "").strip().lower()
    text = text.replace("（", "(").replace("）", ")")
    for ch in _STRIP_CHARS:
        text = text.replace(ch, "")
    return text


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            index.setdefault(normalize_header(alias), canonical)
    return index


_ALIAS_INDEX: dict[str, str] = _build_alias_index()


def _match_one(normalized: str) -> tuple[str, str] | None:
    """匹配单个表头，返回 ``(规范字段, 匹配方式)``；认不出来返回 ``None``。"""
    if not normalized:
        return None
    hit = _ALIAS_INDEX.get(normalized)
    if hit is not None:
        return hit, "alias"
    for suffix in UNIT_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            stem = normalized[: -len(suffix)]
            hit = _ALIAS_INDEX.get(stem)
            if hit is not None:
                return hit, f"alias+{suffix}"
    return None


@dataclass
class ColumnMapping:
    """表头匹配的结果。留全原始表头，出问题时能直接把这句话念给现场的人听。"""

    #: 规范字段 -> 原始表头
    columns: dict[str, str] = field(default_factory=dict)
    #: 原始表头 -> 匹配方式（``alias`` / ``alias+mgl`` / ``duplicate``）
    matched_by: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def recognized(self) -> list[str]:
        return sorted(self.columns)

    def missing(self, required: Iterable[str]) -> list[str]:
        return [f for f in required if f not in self.columns]


def map_columns(headers: Sequence[str]) -> ColumnMapping:
    """把原始表头映射到规范字段。

    ``_date`` / ``_time`` 两列若都在场且没有独立的 ``timestamp`` 列，
    会在 :func:`ingest_rows` 里被合并；这里只负责认出来。

    >>> m = map_columns(["created_at", "Temperature (C)", "PH", "Disolved Oxygen", "Ammonia(g/ml)"])
    >>> m.columns["dissolved_oxygen"]
    'Disolved Oxygen'
    >>> "created_at" in m.matched_by
    True
    """
    mapping = ColumnMapping()
    for raw in headers:
        name = (raw or "").strip()
        if not name:
            continue
        found = _match_one(normalize_header(name))
        if found is None:
            mapping.ignored.append(name)
            continue
        canonical, how = found
        if canonical in mapping.columns:
            mapping.duplicates.append(
                f"{name}（与 {mapping.columns[canonical]} 都识别为「{_INDICATOR_LABEL.get(canonical, canonical)}」，"
                f"已采用前者，后者被忽略）"
            )
            mapping.matched_by[name] = "duplicate"
            continue
        mapping.columns[canonical] = name
        mapping.matched_by[name] = how
    return mapping


# --------------------------------------------------------------------------- #
# 单元格解析
# --------------------------------------------------------------------------- #

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y.%m.%d %H:%M:%S",
    "%Y.%m.%d %H:%M",
    "%Y%m%d%H%M%S",
    "%Y%m%d%H%M",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M",
    "%m-%d-%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y",
)

#: 时间戳末尾可能跟一个时区缩写（数据集里是 ``CET``）。
_TZ_SUFFIX_RE = re.compile(r"\s+([A-Z]{2,5})$")


def parse_timestamp(raw: Any) -> datetime | None:
    """尽力解析一个时间戳，失败返回 ``None``。

    三个刻意的处理：

    1. **丢掉时区缩写**。``2021-06-19 00:00:05 CET`` 无法用 :func:`datetime.strptime`
       直接解析；猜错时区比不带时区更危险，所以只取墙上时间。
    2. **抹平 aware/naive，但已知偏移要先换算再抹**。同一个文件里可能既有
       ``+08:00`` 又有裸时间，混在一起做差会抛 ``TypeError: can't subtract
       offset-naive and offset-aware datetimes``，所以时间轴最终必须是 naive 的。
       但「naive 化」有两条路，结果差得很远：

       - ``2026-07-20T00:00:00Z`` 与 ``2026-07-20T08:00:00+08:00`` 是**同一瞬间**。
         若只把 ``tzinfo`` 抹掉，它们会变成相差 8 小时的两个时刻：混合时区的
         文件排序错乱、趋势图上的时刻整体偏移，而且**不报任何错**。
         所以带 ``±HH:MM`` / ``Z`` 的输入走 :meth:`datetime.astimezone`
         换算到本机时区再抹 ``tzinfo``，与下面纯数字分支的
         ``datetime.fromtimestamp`` 保持同一口径。
       - ``CET`` 这类**缩写**是另一回事：偏移量本身不确定（CET 含夏令时），
         猜错比不猜更危险，所以第 1 条只把它剪掉、墙上时间原样保留。

    3. **纯数字**按 Unix 秒/毫秒识别（导出工具常见）。

    >>> parse_timestamp("2021-06-19 00:00:05 CET").isoformat()
    '2021-06-19T00:00:05'
    >>> parse_timestamp("not a time") is None
    True
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in _MISSING_TOKENS:
        return None
    stripped = _TZ_SUFFIX_RE.sub("", text).strip()

    for candidate in (text, stripped):
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                return dt
            return dt.astimezone().replace(tzinfo=None)

    for fmt in _TIMESTAMP_FORMATS:
        for candidate in (text, stripped):
            try:
                return datetime.strptime(candidate, fmt)
            except ValueError:  # noqa: PERF203 - 只有两个候选格式，异常就是控制流
                continue

    # 纯数字：10 位当秒，13 位当毫秒。
    digits = re.fullmatch(r"\d{10}|\d{13}", stripped)
    if digits:
        value = int(stripped)
        if len(stripped) == 13:
            value //= 1000
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            return None
    return None


def to_float(raw: Any) -> float | None:
    """把单元格转成浮点数；空值、哨兵值、非有限数、非数字都返回 ``None``。

    哨兵值（``-127`` 之类）必须当缺失处理：``IoTPond10.csv`` 的温度列是
    ``-127``，当成真实读数会让「水温 -127℃」直接顶到危险档，
    现场演示会莫名其妙报出高危。

    ``inf`` / ``nan`` 同样必须当缺失处理，理由与
    :func:`fisheries.evaluation.ponds._to_float` 一致：``inf`` 不是一个
    读数，但它是**能通过校验**的合法浮点数 —— 放进去不会让这一格报错，
    而是让整份数据在工具层被整体拒收（``INVALID_INPUT``，因为
    :class:`~fisheries.tools.water_risk.WaterQualityPoint` 拒绝非有限数），
    用户看到的是「上传成功、分析失败」。一个单元格毁掉整张表，代价太大。

    >>> to_float("4.505")
    4.505
    >>> to_float("-127") is None
    True
    >>> to_float("N/A") is None
    True
    >>> to_float("inf") is None
    True
    >>> to_float(float("-inf")) is None
    True
    """
    if raw is None:
        return None
    if isinstance(raw, bool):  # bool 是 int 的子类，先挡掉
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if not math.isfinite(value) or value in MISSING_SENTINELS:
            return None
        return value
    text = str(raw).strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    text = text.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text) else text
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value in MISSING_SENTINELS:
        return None
    return value


def _looks_like_sentinel(raw: Any) -> bool:
    """``to_float`` 返回 ``None`` 之后，判断这是「空着」还是「填了无效哨兵」。

    两种情况都按缺失处理，但对用户的意义完全不同：空着是导出时没写，
    填 ``-127`` 是设备自己说「本次读数无效」。后者不点出来的话，用户会
    以为这一格读到了值，只是被页面吞了。
    """
    if raw is None or isinstance(raw, bool):
        return False
    if isinstance(raw, (int, float)):
        return float(raw) in MISSING_SENTINELS
    text = str(raw).strip().replace(",", "")
    try:
        return float(text) in MISSING_SENTINELS
    except ValueError:
        return False


def _looks_non_finite(raw: Any) -> bool:
    """``to_float`` 返回 ``None`` 之后，判断这是「导出写了 inf」还是别的原因。

    ``inf`` 与 ``-inf`` 被 :func:`to_float` 当缺失处理是对的，但用户需要知道
    「这一格不是空的」：pandas / numpy 导出、Excel 里除零的公式，都会把
    ``inf`` 原样写进 CSV。不点出来的话，用户会一直以为那几格是设备掉线，
    跑去查网络，而真正要修的是导出脚本。

    文本形式的 ``nan`` 落在 :data:`_MISSING_TOKENS` 里，按「空着」处理 ——
    它本来就是被当作缺失标记写出来的，不必再单独解释一次。
    """
    if raw is None or isinstance(raw, bool):
        return False
    if isinstance(raw, (int, float)):
        return not math.isfinite(float(raw))
    text = str(raw).strip().replace(",", "")
    if text.lower() in _MISSING_TOKENS:
        return False
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# 单位/量级核查
# --------------------------------------------------------------------------- #

def unit_warnings(indicator: str, values: Sequence[float]) -> list[str]:
    """对一个指标做量级核查，返回该指标的可疑说明（可能为空）。

    这里只做「这个数在这个单位下讲不讲得通」的判断，不做超标判断 ——
    超标是 :mod:`fisheries.tools.water_risk` 的事，重复实现会漂移。

    >>> unit_warnings("dissolved_oxygen", [4.5, 5.0, 4.8])
    []
    """
    if not values:
        return []
    label = _INDICATOR_LABEL.get(indicator, indicator)
    notes: list[str] = []

    low, high, unit = _PLAUSIBLE_RANGE.get(indicator, (-1e12, 1e12, ""))
    med = statistics.median(values)
    if med < low or med > high:
        notes.append(
            f"{label}中位数 {med:.4g} 超出物理量程 {low:g}–{high:g}"
            f"{'（' + unit + '）' if unit else ''}，请检查列是否映射错误。"
        )
        return notes

    suspect = _UNIT_SUSPECT_ABOVE.get(indicator)
    if suspect is not None and med > suspect:
        notes.append(
            f"{label}中位数 {med:.4g} 远超养殖水体可能值 {suspect:g} mg/L。"
            f"若列名标的是 g/ml 之类的小单位，实际数值几乎可以确定是 mg/L —— "
            f"请与设备方核对列名与量纲后再用于决策。"
        )
    negatives = sum(1 for v in values if v < 0)
    if negatives:
        notes.append(f"{label}有 {negatives} 个负值（理化指标不可能为负），已按缺失处理。")
    return notes


# --------------------------------------------------------------------------- #
# 读文件
# --------------------------------------------------------------------------- #

_CSV_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1")


def _read_csv(path: Path) -> tuple[list[str], list[list[str]], str]:
    """读 CSV/TSV，返回 ``(表头, 数据行, 说明)``。"""
    raw = path.read_bytes()
    text: str | None = None
    used = ""
    for enc in _CSV_ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        used = enc
        break
    if text is None:  # pragma: no cover —— latin-1 不会失败，留个兜底
        raise TableError(f"无法解码表格 {path.name}；请另存为 UTF-8 或 GBK 编码的 CSV。")

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","

    rows = [
        row for row in csv.reader(io.StringIO(text), delimiter=delimiter)
        if any((cell or "").strip() for cell in row)
    ]
    if not rows:
        raise TableError(f"表格 {path.name} 里没有任何数据行。")
    note = f"CSV / 编码 {used} / 分隔符 {delimiter!r}"
    return [str(c) for c in rows[0]], [[str(c) for c in r] for r in rows[1:]], note


def _read_xlsx(path: Path) -> tuple[list[str], list[list[str]], str]:
    """读 xlsx 的首个工作表。缺 openpyxl 时给一句可操作的提示。"""
    try:
        from openpyxl import load_workbook  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover —— 取决于环境
        raise TableError(
            "读取 .xlsx 需要 openpyxl。请先安装：\n"
            "    python -m pip install openpyxl\n"
            "或在 Excel 里另存为 CSV 再上传（现场演示推荐后者，无额外依赖）。"
        ) from exc

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = book[book.sheetnames[0]]
        rows: list[list[str]] = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                rows.append(cells)
    finally:
        book.close()
    if not rows:
        raise TableError(f"表格 {path.name} 的第一个工作表是空的。")
    return rows[0], rows[1:], f"XLSX / 工作表 {book.sheetnames[0]!r}"


def read_table_rows(path: str | Path) -> tuple[list[str], list[list[str]], str]:
    """读表格文件，返回 ``(表头, 数据行, 说明)``。"""
    p = Path(path)
    if not p.exists():
        raise TableError(f"找不到表格文件：{p}")
    if not p.is_file():
        raise TableError(f"这不是一个文件：{p}")
    suffix = p.suffix.lower()
    if suffix in {".csv", ".tsv", ".tab", ".txt"}:
        return _read_csv(p)
    if suffix in {".xlsx", ".xlsm"}:
        return _read_xlsx(p)
    if suffix == ".xls":
        raise TableError(
            "不支持旧版 .xls。请在 Excel 里另存为 .xlsx 或 CSV 后再上传。"
        )
    raise TableError(
        f"不认识的表格格式 {suffix or '（无扩展名）'}；支持 .csv / .tsv / .xlsx / .xlsm。"
    )


# --------------------------------------------------------------------------- #
# 摄入结果
# --------------------------------------------------------------------------- #

@dataclass
class TableIngest:
    """一次表格摄入的完整结果：既有喂给工具的数据，也有给人看的说明。"""

    #: 喂给 ``water_risk_predict`` 的读数列表（已排序、已清洗、已限长）
    readings: list[dict[str, Any]] = field(default_factory=list)
    #: 规范字段 -> 原始表头
    columns: dict[str, str] = field(default_factory=dict)
    matched_by: dict[str, str] = field(default_factory=dict)
    ignored_columns: list[str] = field(default_factory=list)
    #: 必须让用户知道的问题（不会阻断运行）
    warnings: list[str] = field(default_factory=list)
    #: 会导致跑不起来的问题（缺列等）
    errors: list[str] = field(default_factory=list)

    source: str = ""
    reader_note: str = ""
    n_rows_read: int = 0
    n_rows_used: int = 0
    n_rows_truncated: int = 0
    n_timestamps_parsed: int = 0
    n_timestamps_failed: int = 0
    n_timestamp_filled: int = 0
    n_cells_missing: dict[str, int] = field(default_factory=dict)
    #: 因为缺少必填指标而被整行丢弃的行数（见 ``ingest_rows`` 的组装读数段）
    n_rows_incomplete: int = 0
    medians: dict[str, float] = field(default_factory=dict)
    interval_minutes: int | None = None
    interval_source: str = ""
    span_hours: float | None = None
    time_range: tuple[str, str] | None = None
    #: 给界面预览用的表格（``[表头, 前若干行, ..., 末行]``）
    preview: list[list[str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """能不能拿这份数据去调水质工具。"""
        return not self.errors and len(self.readings) >= MIN_POINTS

    @property
    def n_points(self) -> int:
        return len(self.readings)

    def summary_markdown(self) -> str:
        """给界面用的「数据接入质量」面板。"""
        lines: list[str] = []
        if self.source:
            lines.append(f"**来源**：`{self.source}`　{self.reader_note}")
        lines.append(
            f"**读入 {self.n_rows_read} 行 → 采用 {self.n_rows_used} 点**"
            + (f"（超出上限截去最早的 {self.n_rows_truncated} 行，保留最近的）"
               if self.n_rows_truncated else "")
        )
        if self.columns:
            pairs = "、".join(
                f"`{raw}` → {_INDICATOR_LABEL.get(c, '时间')}"
                if c != "timestamp" else f"`{raw}` → 时间"
                for c, raw in sorted(self.columns.items())
            )
            lines.append(f"**列映射**：{pairs}")
        if self.interval_minutes:
            lines.append(
                f"**采样间隔**：{self.interval_minutes} 分钟"
                f"（{'由时间戳推断' if self.interval_source == 'timestamp' else self.interval_source or '输入指定'}）"
            )
        if self.span_hours is not None:
            lines.append(f"**时间跨度**：{self.span_hours:.1f} 小时")
        if self.time_range:
            lines.append(f"**时间范围**：{self.time_range[0]} ~ {self.time_range[1]}")
        ts_note = f"时间戳解析成功 {self.n_timestamps_parsed} 行"
        if self.n_timestamps_failed:
            ts_note += f"，失败 {self.n_timestamps_failed} 行"
        if self.n_timestamp_filled:
            ts_note += f"，补插 {self.n_timestamp_filled} 行"
        if self.n_timestamps_parsed or self.n_timestamps_failed:
            lines.append(f"**时间戳**：{ts_note}")
        if self.medians:
            med = "、".join(
                f"{_INDICATOR_LABEL.get(k, k)} {v:.4g}"
                for k, v in sorted(self.medians.items())
            )
            lines.append(f"**中位数**：{med}")
        if self.n_rows_incomplete:
            lines.append(
                f"**丢弃整行**：{self.n_rows_incomplete} 行缺少必填指标，"
                f"无法参与趋势外推（明细见下方提醒）"
            )
        if self.n_cells_missing:
            gaps = "、".join(
                f"{_INDICATOR_LABEL.get(k, k)} 缺 {v} 格"
                for k, v in sorted(self.n_cells_missing.items())
            )
            lines.append(f"**缺失格**：{gaps}")

        if self.errors:
            lines.append("")
            lines.append("#### ⛔ 无法用于评估")
            lines.extend(f"- {e}" for e in self.errors)
        if self.warnings:
            lines.append("")
            lines.append("#### ⚠️ 需要注意")
            lines.extend(f"- {w}" for w in self.warnings)
        if self.ignored_columns:
            lines.append("")
            lines.append(
                f"**未使用的列**（{len(self.ignored_columns)}）："
                + "、".join(f"`{c}`" for c in self.ignored_columns)
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def _merge_datetime(date_cell: Any, time_cell: Any) -> str:
    """把拆开的日期列与时间列拼成一串再解析。"""
    date_text = "" if date_cell is None else str(date_cell).strip()
    time_text = "" if time_cell is None else str(time_cell).strip()
    if not date_text and not time_text:
        return ""
    return f"{date_text} {time_text}".strip()


def ingest_rows(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    source: str = "",
    reader_note: str = "",
    max_rows: int = 720,
    interval_minutes: int | None = None,
) -> TableIngest:
    """把「表头 + 数据行」摄入成 :class:`TableIngest`。

    与 :func:`ingest_table` 分开，是为了让单元测试可以直接喂内存里的表格，
    不必先落一个临时文件。

    Args:
        max_rows: 最多采用多少行。超出时**保留最近的**（水质工具关心的是尾部趋势），
            并记进 ``n_rows_truncated``。
        interval_minutes: 表格没有可用时间列时使用的采样间隔。
    """
    result = TableIngest(source=source, reader_note=reader_note)
    result.n_rows_read = len(rows)

    mapping = map_columns(headers)
    result.columns = dict(mapping.columns)
    result.matched_by = dict(mapping.matched_by)
    result.ignored_columns = list(mapping.ignored)

    # ---- 时间列：独立 timestamp > (日期 + 时间) > 单独一列 ----------------
    #: ``merge_parts`` 非 None 表示需要把两列拼起来再解析。
    merge_parts: tuple[str, str] | None = None
    ts_col = mapping.columns.get("timestamp")
    date_col = mapping.columns.get("_date")
    time_col = mapping.columns.get("_time")
    if ts_col is None:
        if date_col is not None and time_col is not None:
            merge_parts = (date_col, time_col)
            result.columns["timestamp"] = f"{date_col} + {time_col}"
        elif date_col is not None or time_col is not None:
            ts_col = date_col or time_col
            result.columns["timestamp"] = str(ts_col)
    # ``_date`` / ``_time`` 是内部伪字段，不进展示用的列映射。
    result.columns.pop("_date", None)
    result.columns.pop("_time", None)

    # ---- 缺列检查：先报错，别让工具层抛 pydantic 异常 -------------------
    missing = [f for f in REQUIRED_FIELDS if f not in mapping.columns]
    if missing:
        names = "、".join(f"{_INDICATOR_LABEL[f]}（{f}）" for f in missing)
        accepted = "、".join(
            f"{_INDICATOR_LABEL[f]} 可写 {'/'.join(COLUMN_ALIASES[f][:5])}"
            for f in missing
        )
        result.errors.append(
            f"表格缺少必须的列：{names}。"
            f"水质风险工具要求这四项齐全，缺一项都无法定级。可接受的列名示例：{accepted}。"
        )
    if not rows:
        result.errors.append("表格里没有数据行。")

    if result.errors:
        result.preview = [list(headers)] + [[("" if c is None else str(c)) for c in r]
                                            for r in list(rows)[:5]]
        return result

    by_index = {f: headers.index(raw) for f, raw in mapping.columns.items()
                if raw in headers}
    # 时间列的表头 -> 列下标，循环外算一次。``headers.index`` 是 O(列数)，
    # 放在 720 行的循环里就是 720 次线性扫描，纯粹白烧。
    ts_index = (
        headers.index(ts_col) if ts_col is not None and ts_col in headers else None
    )
    merge_index = (
        (headers.index(merge_parts[0]), headers.index(merge_parts[1]))
        if merge_parts is not None
        and merge_parts[0] in headers
        and merge_parts[1] in headers
        else None
    )

    def cell_at(row: Sequence[Any], idx: int | None) -> Any:
        """按列序号取单元格。行是显式入参，不再靠闭包捕获循环变量。"""
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    # ---- 逐行取值 -------------------------------------------------------
    parsed: list[tuple[datetime | None, dict[str, Any]]] = []
    missing_cells: dict[str, int] = dict.fromkeys(INDICATORS, 0)
    negatives: dict[str, int] = dict.fromkeys(INDICATORS, 0)
    sentinels: dict[str, int] = dict.fromkeys(INDICATORS, 0)
    non_finite: dict[str, int] = dict.fromkeys(INDICATORS, 0)
    for row in rows:
        stamp: datetime | None = None
        if merge_index is not None:
            stamp = parse_timestamp(
                _merge_datetime(cell_at(row, merge_index[0]), cell_at(row, merge_index[1]))
            )
        elif ts_index is not None:
            stamp = parse_timestamp(cell_at(row, ts_index))

        point: dict[str, Any] = {}
        for field_name in INDICATORS:
            raw_cell = cell_at(row, by_index.get(field_name))
            value = to_float(raw_cell)
            if value is None:
                missing_cells[field_name] += 1
                if _looks_like_sentinel(raw_cell):
                    sentinels[field_name] += 1
                elif _looks_non_finite(raw_cell):
                    non_finite[field_name] += 1
            elif value < 0 and field_name != "water_temp":
                # 理化指标为负 = 传感器/导出异常，按缺失处理，别让它顶到危险档。
                negatives[field_name] += 1
                missing_cells[field_name] += 1
                value = None
            elif field_name == "water_temp" and value < -5.0:
                missing_cells[field_name] += 1
                value = None
            point[field_name] = value
        parsed.append((stamp, point))

    result.n_cells_missing = {k: v for k, v in missing_cells.items() if v}
    for field_name, count in negatives.items():
        if count:
            result.warnings.append(
                f"{_INDICATOR_LABEL[field_name]} 有 {count} 个负值，"
                f"理化指标不可能为负，已按缺失处理（请检查设备与导出）。"
            )
    for field_name, count in sentinels.items():
        if count:
            result.warnings.append(
                f"{_INDICATOR_LABEL[field_name]} 有 {count} 个负值哨兵"
                f"（-127 / -999 这类设备用来表示「本次读数无效」的固定值），"
                f"已按缺失处理。"
            )
    for field_name, count in non_finite.items():
        if count:
            result.warnings.append(
                f"{_INDICATOR_LABEL[field_name]} 有 {count} 个非有限值（inf / nan），"
                f"已按缺失处理。导出工具算不出数时会写 inf，"
                f"这一格并不是空的，建议回原始数据核对来源。"
            )

    # ---- 时间轴：排序 + 补齐 + 推断间隔 ---------------------------------
    # ``n_timestamps_*`` 记的是**补插之前**的真实情况：用户要知道的是
    # 「我那张表有多少行时间没读出来」，而不是补完之后的形式上的 0。
    n_ok = sum(1 for s, _ in parsed if s is not None)
    result.n_timestamps_parsed = n_ok
    result.n_timestamps_failed = len(parsed) - n_ok

    timed = [(s, p) for s, p in parsed if s is not None]
    is_sorted = all(timed[i][0] <= timed[i + 1][0] for i in range(len(timed) - 1))

    if n_ok and n_ok == len(parsed):
        if not is_sorted:
            parsed.sort(key=lambda item: item[0])
            result.warnings.append(
                "原文件的时间顺序是乱的，已按时间戳升序重排后再做趋势外推。"
            )
        result.n_timestamp_filled = 0
    elif n_ok and n_ok != len(parsed):
        # 时间戳只解析出一部分。工具层的 ``_resolve_interval`` 要求「**所有**点
        # 都有时间戳」才肯用时间戳推断间隔，否则整体退回 ``interval_minutes``
        # 入参 —— 那会让趋势外推的时长含义整体错掉。所以这里补插一条单调时间轴。
        #
        # 补插用**按行序的线性内插**，而不是「从首点往后每步加一个间隔」：
        # 后者会让补出来的点抢占真实时间戳的位置，把后面所有实测点的时刻
        # 整体推后一格（数值与时刻错位）。内插只改时刻，而且以真实的邻居为锚。
        known_idx = [i for i, (s, _) in enumerate(parsed) if s is not None]
        # 每行的平均步长：用相邻两个已知点的间隔除以它们隔了几行。
        per_row: list[float] = []
        for a, b in itertools.pairwise(known_idx):
            gap = (parsed[b][0] - parsed[a][0]).total_seconds()
            if gap > 0 and b > a:
                per_row.append(gap / (b - a))
        step_seconds = (
            statistics.median(per_row) if per_row
            else float((interval_minutes or 60) * 60)
        )
        for i, (stamp, point) in enumerate(parsed):
            if stamp is not None:
                continue
            # ``known_idx`` 升序，二分找左右邻居。原来每缺一行就扫一遍
            # 整张表（O(n·k)），720 行全缺时间戳时是 50 万次比较。
            pos = bisect.bisect_left(known_idx, i)
            prev_i = known_idx[pos - 1] if pos > 0 else None
            next_i = known_idx[pos] if pos < len(known_idx) else None
            if prev_i is not None and next_i is not None:
                # 落在两个已知点之间：按行序号线性内插
                fraction = (i - prev_i) / (next_i - prev_i)
                base = parsed[prev_i][0]
                span = parsed[next_i][0] - base
                filled = base + span * fraction
            elif prev_i is not None:
                filled = parsed[prev_i][0] + timedelta(
                    seconds=step_seconds * (i - prev_i)
                )
            else:
                assert next_i is not None
                filled = parsed[next_i][0] - timedelta(
                    seconds=step_seconds * (next_i - i)
                )
            parsed[i] = (filled, point)
        result.n_timestamp_filled = result.n_timestamps_failed
        result.warnings.append(
            f"有 {result.n_timestamps_failed} 行的时间戳没解析出来（格式不统一），"
            f"已按相邻已知时间点线性内插补上。"
            f"补插只影响时间轴，实测数值一格没动，"
            f"但落在这些时刻上的读数只是位置估计。"
        )
    else:
        # 一个时间戳都没解析出来：全部置 None，由工具层退回 interval_minutes。
        parsed = [(None, p) for _, p in parsed]
        result.warnings.append(
            "没有可用的时间列（或全部无法解析），本次按行序处理。"
            + (f"采样间隔按你填的 {interval_minutes} 分钟。"
               if interval_minutes else
               "采样间隔将退回工具默认值 60 分钟，趋势外推的时长含义可能偏差，"
               "建议补一列时间或显式填写采样间隔。")
        )

    # ---- 截长：保留最近的 max_rows 行 -----------------------------------
    # 截断本身只记进 ``n_rows_truncated``，不再另加一条 warning ——
    # ``summary_markdown`` 已经把它写进「读入 N 行 → 采用 M 点」那一行了，
    # 同一件事说两遍会让真正要提醒的问题被淹没。
    if max_rows > 0 and len(parsed) > max_rows:
        result.n_rows_truncated = len(parsed) - max_rows
        parsed = parsed[-max_rows:]
    result.n_rows_used = len(parsed)

    # ---- 间隔与跨度 ------------------------------------------------------
    stamps = [s for s, _ in parsed if s is not None]
    if len(stamps) >= 2:
        deltas = [(stamps[i + 1] - stamps[i]).total_seconds() / 60.0
                  for i in range(len(stamps) - 1)]
        deltas = [d for d in deltas if d > 0]
        if deltas:
            med = statistics.median(deltas)
            result.interval_minutes = round(med)
            result.interval_source = "timestamp"
            if med < 1:
                result.warnings.append(
                    f"推断出的采样间隔只有 {med:.2f} 分钟，过密的数据会让外推斜率失真，"
                    f"建议按小时聚合后再上传。"
                )
            spread = max(deltas) - min(deltas)
            if spread > 0.5 * med:
                result.warnings.append(
                    f"采样间隔不均匀（{min(deltas):.0f}–{max(deltas):.0f} 分钟），"
                    f"工具会按中位数 {med:.0f} 分钟处理，趋势斜率可能有偏差。"
                )
            result.span_hours = round(
                (stamps[-1] - stamps[0]).total_seconds() / 3600.0, 2
            )
            result.time_range = (
                stamps[0].strftime("%Y-%m-%d %H:%M"),
                stamps[-1].strftime("%Y-%m-%d %H:%M"),
            )
    elif interval_minutes:
        result.interval_minutes = int(interval_minutes)
        result.interval_source = "parameter"

    # ---- 指标质量 --------------------------------------------------------
    for field_name in INDICATORS:
        values = [p[field_name] for _, p in parsed if p.get(field_name) is not None]
        if not values:
            continue
        result.medians[field_name] = float(statistics.median(values))
        for note in unit_warnings(field_name, values):
            result.warnings.append(note)

    # ---- 硝酸盐 ≠ 亚硝酸盐 ----------------------------------------------
    if "nitrate_n" in mapping.columns:
        raw_nitrate = mapping.columns["nitrate_n"]
        if "nitrite_n" not in mapping.columns:
            result.warnings.append(
                f"表里有 `{raw_nitrate}`（硝酸盐），**没有亚硝酸盐列**。"
                f"亚硝酸盐毒性远高于硝酸盐（鲤科 >0.3 mg/L 即危险），"
                f"两者不能互相替代，本次亚硝酸盐风险无法评估，"
                f"报告里会按「缺失指标」处理。"
            )
        else:
            result.warnings.append(
                f"同时存在 `{raw_nitrate}`（硝酸盐）与 "
                f"`{mapping.columns['nitrite_n']}`（亚硝酸盐），"
                f"已分别取用，未混为一谈。"
            )

    if "nitrite_n" not in mapping.columns and "nitrate_n" not in mapping.columns:
        result.warnings.append(
            "表格里既没有亚硝酸盐（Nitrite）也没有硝酸盐（Nitrate），"
            "氮素风险只覆盖了氨氮一项。"
        )

    # ---- 组装读数 --------------------------------------------------------
    # 为什么是「丢整行」而不是「把缺失的格子留成 None」：
    # ``WaterQualityPoint`` 里 water_temp / ph / dissolved_oxygen / ammonia_n
    # 都是**必填**字段，传 None 和传空都是 pydantic 校验错误。只要有一格缺失，
    # 这一个点就没法参与定级了——所以丢的是这一行，不是整次运行。
    # 反过来，**绝不能把缺失的键悄悄删掉就发出去**：那样工具会对整个请求
    # 报字段缺失，用户看到的是「工具调用失败」，而真正的原因只是某一格是空的。
    readings: list[dict[str, Any]] = []
    dropped_incomplete = 0
    dropped_fields: dict[str, int] = {}
    for stamp, point in parsed:
        gaps = [f for f in REQUIRED_FIELDS if point.get(f) is None]
        if gaps:
            dropped_incomplete += 1
            for f in gaps:
                dropped_fields[f] = dropped_fields.get(f, 0) + 1
            continue
        item: dict[str, Any] = {}
        if stamp is not None:
            item["timestamp"] = stamp.isoformat()
        for field_name in INDICATORS:
            value = point.get(field_name)
            if value is not None:
                item[field_name] = float(value)
        readings.append(item)
    result.readings = readings
    result.n_rows_incomplete = dropped_incomplete
    if dropped_incomplete:
        detail = "、".join(
            f"{_INDICATOR_LABEL[f]} {c} 格" for f, c in sorted(dropped_fields.items())
        )
        result.warnings.append(
            f"有 {dropped_incomplete} 行因为缺少必填指标被整行丢弃（{detail}）。"
            f"水质工具要求每个点都同时具备水温、pH、溶氧、氨氮四项，"
            f"缺一项这个点就无法参与趋势外推。已采用的 {len(readings)} 个点不含这些行。"
        )
        if len(readings) < MIN_POINTS:
            result.errors.append(
                f"丢弃不完整行之后只剩 {len(readings)} 个可用点，"
                f"水质趋势至少需要 {MIN_POINTS} 个点。"
                f"请检查这几列的缺失原因（设备掉线？导出时留空？），"
                f"或换一段数据更完整的时段。"
            )

    # ---- 预览表 ----------------------------------------------------------
    preview_head = ["时间"] + [_INDICATOR_LABEL[f] for f in INDICATORS]
    result.preview = [preview_head]
    shown = readings[:6]
    if len(readings) > 8:
        shown = readings[:6] + readings[-2:]
    for item in shown:
        row_out = [str(item.get("timestamp", "—"))]
        for field_name in INDICATORS:
            value = item.get(field_name)
            row_out.append("—" if value is None else f"{value:.3f}")
        result.preview.append(row_out)
    if len(readings) > 8:
        result.preview.insert(7, ["…", "…", "…", "…", "…", "…"])

    return result


def ingest_table(
    path: str | Path,
    *,
    max_rows: int = 720,
    interval_minutes: int | None = None,
) -> TableIngest:
    """读一个表格文件并摄入。这是界面层唯一需要调用的入口。"""
    p = Path(path)
    headers, rows, note = read_table_rows(p)
    return ingest_rows(
        headers,
        rows,
        source=p.name,
        reader_note=note,
        max_rows=max_rows,
        interval_minutes=interval_minutes,
    )
