"""中英提示词回译层 —— 让纯英文训练的领域模型也能产出中文结论。

为什么需要这一层
==================
本项目的建议角色想用 `KurmaAI/AQUA-7B`，但它的训练数据标签是 ``language: en``，
纯英文，直接要中文输出质量无保证。取舍是"英文提示词进、英文答案出，再回译成中文"：
保住它的领域知识优势（这是选它的唯一理由），同时满足竞赛的中文交付要求。

这里有一条不能让步的底线
==========================
**绝不输出半英半中的文本。** 一个养殖户看到
"立即 turn on the aerator，keep it running until tomorrow morning"，
不仅读不懂，还会怀疑整套系统的可靠性。

所以回译按"逐字段判定"进行，每个字段只有两种结局：

1. **译得出来** → 用译文（且数值必须原样保留，见下）；
2. **译不出来** → 该字段整条改用规则模板里的中文句子。

没有第三种"凑合译一半"。判定标准是硬的：**译文里不许残留任何英文单词**。

四种回译后端
==============
======================  ============  ==================================
后端                    是否联网      质量与代价
======================  ============  ==================================
``local``（首选）       否            本地离线的轻量 NMT（默认
                                      ``models/translate/opus-mt-en-zh``）。
                                      **这是 D12 收益唯一能真正兑现的路径**：
                                      不联网、不需要 key，也能把 AQUA-7B 的
                                      整句英文译成中文。代价是通用域模型不懂
                                      水产，要靠"分块 + 后订正"补齐
                                      （见下文"本地模型为什么还需要后订正"）。
``glossary``（兜底）    否            受控词表逐词/逐短语替换。领域词汇
                                      覆盖得住就很好，自由长句往往覆盖不住，
                                      覆盖不住就整条退回模板。零依赖、确定性。
``llm``                 是            调一个中文能力强的模型（DeepSeek/Qwen）
                                      整句回译，质量最好。需要 API key。
``bypass``              否            不做回译，直接把英文原文交出去。**只在
                                      调试提示词时用**，正常演示不要开。
======================  ============  ==================================

一条字段能被译出来，要过三关：**分块长度、数值多重集、英文残留**。三关都过才用译文，
否则整条退回中文模板——这一层的底线是"宁可退回模板，也不要半英半中"。

长句为什么必须分块
====================
本地模型一次吞下整段英文时，实测会**把尾部整段丢掉**：输入
``"Turn on the aerator tonight and measure dissolved oxygen and ammonia
nitrogen again tomorrow morning at 6."``，输出只有"今晚打开气动器
测量溶解氧气和氨氮"——``again`` / ``tomorrow morning`` / ``at 6`` 全没了，
**连数字 6 都丢了**。数值校验能发现"数字少了"，但不知道少的是哪一句，
只能整条放弃。所以先按句、再按小句、最后按词把输入切到
``DEFAULT_CHUNK_CHARS`` 以内，逐块翻译再拼回去。切分**不会切在数字与单位之间**
（``1.5 mg/L`` 是一个整体）。

本地模型为什么还需要后订正
============================
通用 NMT 的失败模式和词表后端**正好相反**：

======================  ====================  ==================
现象                    词表后端              本地 NMT
======================  ====================  ==================
词认得吗                认得                  译成同义的通用词
语序对吗                拼不出               对
例子                    "检测溶氧并氨氮再今晚"  "turbidity" → "多发性"
======================  ====================  ==================

两者互补，所以方案是"**让 NMT 负责语序、让词表负责术语**"：NMT 先出通顺中文，
再用一张中文→中文的后订正表（``GLOSSARY_POST_ZH``）把术语扳回来，
最后把单位规范化回 ``mg/L`` / ``℃`` 这类标识符。这张表只收"本领域里不可能有
第二种解释"的错法——``措施`` 这种既是错译又是正当中文的词坚决不收。

数值保真
==========
提示词体系里有一条铁律："数值一个字都不能改"。回译层必须替这条铁律兜底：
译文里的**数字多重集**必须与原文完全一致（``2.6`` 不能变成 ``2.6 左右`` 之外的
任何形式，也不能凭空多出一个 ``3``）。校验不过就当作"译不出来"，
退回模板 —— 宁可少一条模型的措辞，也不要一个被译者改过的溶氧值。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# --------------------------------------------------------------------------- #
# 一、语言与后端选择
# --------------------------------------------------------------------------- #

LANG_ZH = "zh"
LANG_EN = "en"
SUPPORTED_LANGS: tuple[str, ...] = (LANG_ZH, LANG_EN)

#: 显式指定提示词语言。``auto``（默认）时按后端自动判断。
PROMPT_LANG_ENV = "FISHERIES_PROMPT_LANG"

#: 回译后端。``auto``（默认）时：配了翻译模型就用 ``llm``，否则用 ``glossary``。
TRANSLATOR_MODE_ENV = "FISHERIES_TRANSLATOR_MODE"

#: 翻译模型的环境变量前缀（与 ``FISHERIES_LLM_*`` 同构，可复用同一套配置习惯）。
TRANSLATOR_ENV_PREFIX = "FISHERIES_TRANSLATOR_"

TRANSLATOR_MODES: tuple[str, ...] = ("auto", "local", "glossary", "llm", "bypass")

#: 单个块允许的最大字符数 / 词数。**这两个数是实测出来的，不是估的。**
#:
#: 拿 ``"Turn on the aerator tonight and measure dissolved oxygen and ammonia
#: nitrogen again tomorrow morning at 6."``（106 字符 / 17 词）当标尺，逐档收紧
#: 预算重跑，看"数字 6 还在不在"：
#:
#: ==========  ==========  ==============  ==========================
#: 字符预算     词预算      实际切出几块     结果
#: ==========  ==========  ==============  ==========================
#: 200         40          1               ✘ 丢了 ``at 6``
#: 140         28          1               ✘ 丢了 ``at 6``
#: 110         20          1               ✘ 丢了 ``at 6``
#: 90          16          2               ✔ 6 保住了
#: 70          12          2               ✔
#: 55          9           3               ✔
#: ==========  ==========  ==============  ==========================
#:
#: 结论：**词预算是那条线，字符预算不是**。17 词的句子只要不切就必丢尾部，
#: 所以词预算必须 ≤16。取 110/16 而不是更小的 90/16，是因为切得越碎、
#: 模型越容易把术语译错（预算 200 时 ``quicklime`` 译成"生石灰"，
#: 预算 70 时变成"快生石灰灰"）——**够用就好，不要过度切**。
#:
#: 这个预算同时充当**重试触发器**：句子在预算内就是一次调用（与改造前完全
#: 相同），超出预算才切开。所以短句的延迟与质量都没有任何变化。
DEFAULT_CHUNK_CHARS = 110
DEFAULT_CHUNK_WORDS = 16

#: 一个字段最多切几块。超过就整条放弃——宁可退回模板，也不要一次回译
#: 打出几十次模型调用（模型是逐块前向的，块数直接等于耗时）。
DEFAULT_MAX_CHUNKS = 24

#: provider 名里出现这些片段就认为是"英文优先"的本地领域模型。
_ENGLISH_FIRST_PROVIDER_HINTS: tuple[str, ...] = ("aqua", "mistral", "llama")


def resolve_prompt_lang(explicit: str | None = None,
                        provider: str = "",
                        env: dict[str, str] | None = None) -> str:
    """决定建议/调度角色用哪套语言的提示词。

    优先级：显式入参 > 环境变量 ``FISHERIES_PROMPT_LANG`` > 按后端自动判断。

    自动判断的理由：`aqua-7b-*` 这类后端是英文优先的，中文提示词会让它
    在"理解指令"这一步就开始丢分；而 DeepSeek / Qwen 这类中文后端反过来，
    中文提示词更准。所以默认跟着模型走，而不是全局写死一种语言。

    Args:
        explicit: 调用方显式指定的语言（``zh`` / ``en`` / ``auto``）。
        provider: LLM provider 名，如 ``aqua-7b-ollama``。
        env: 注入的环境字典（测试用），默认读 ``os.environ``。

    Returns:
        ``"zh"`` 或 ``"en"``。

    Examples:
        >>> resolve_prompt_lang(None, "aqua-7b-ollama")
        'en'
        >>> resolve_prompt_lang(None, "deepseek")
        'zh'
        >>> resolve_prompt_lang("zh", "aqua-7b-ollama")
        'zh'
    """
    e = dict(os.environ if env is None else env)
    raw = str(explicit or e.get(PROMPT_LANG_ENV, "") or "auto").strip().lower()
    if raw in SUPPORTED_LANGS:
        return raw
    lowered = (provider or "").lower()
    if any(hint in lowered for hint in _ENGLISH_FIRST_PROVIDER_HINTS):
        return LANG_EN
    return LANG_ZH


def resolve_translator_mode(explicit: str | None = None,
                            env: dict[str, str] | None = None) -> str:
    """决定回译后端：``auto`` / ``glossary`` / ``llm`` / ``bypass``。

    ``auto`` 的判据是"有没有配翻译模型"——配了就用 ``llm``（质量优先），
    没配就用 ``glossary``（离线优先）。**不会因为没配就报错**，
    这是本项目"零依赖离线可跑"的硬要求。
    """
    e = dict(os.environ if env is None else env)
    raw = str(explicit or e.get(TRANSLATOR_MODE_ENV, "") or "auto").strip().lower()
    return raw if raw in TRANSLATOR_MODES else "auto"


# --------------------------------------------------------------------------- #
# 二、受控词表
# --------------------------------------------------------------------------- #
#
# 词表只收"这个项目的提示词里真的会出现的词"：指标名、病害名、塘口动作、
# 时间词、单位、以及少量连接词。**不收通用英文**——收了也覆盖不住自由长句，
# 反而给人"能全文翻译"的错觉。覆盖不住就老老实实退回模板。

GLOSSARY_ZH: tuple[tuple[str, str], ...] = (
    # ---- 指标 ----
    ("dissolved oxygen", "溶氧"),
    ("ammonia nitrogen", "氨氮"),
    ("non-ionized ammonia", "分子氨"),
    ("nitrite nitrogen", "亚硝酸盐"),
    ("nitrite", "亚硝酸盐"),
    ("nitrate", "硝酸盐"),
    ("water temperature", "水温"),
    ("temperature", "水温"),
    ("turbidity", "浊度"),
    ("salinity", "盐度"),
    ("oxygen", "氧气"),
    ("ammonia", "氨氮"),
    # ---- 病害 ----
    ("bacterial gill disease", "烂鳃病"),
    ("gill rot", "烂鳃病"),
    ("hemorrhagic septicemia", "细菌性出血病"),
    ("bacterial red disease", "细菌性败血症"),
    ("saprolegniasis", "水霉病"),
    ("saprolegnia", "水霉病"),
    ("fungal infection", "真菌感染"),
    ("parasitic disease", "寄生虫病"),
    ("white tail disease", "白尾病"),
    ("disease", "病害"),
    ("lesion", "病灶"),
    ("symptom", "症状"),
    ("redness", "充血"),
    ("hemorrhage", "出血"),
    ("ulcer", "溃疡"),
    ("mucus", "黏液"),
    ("gill", "鳃"),
    ("detected", "检出"),
    ("detect", "检出"),
    ("no disease", "无病害"),
    ("no", "无"),
    ("none", "无"),
    ("none detected", "未检出"),
    ("suspected", "疑似"),
    ("confirmed", "确诊"),
    # ---- 生物与塘口 ----
    ("fish", "鱼"),
    ("pond", "池塘"),
    ("water body", "水体"),
    ("water quality", "水质"),
    ("stocking density", "养殖密度"),
    ("population", "存量"),
    ("mortality", "死亡"),
    ("dead fish", "死鱼"),
    ("algae", "藻类"),
    ("phytoplankton", "浮游植物"),
    ("bottom sediment", "底泥"),
    # ---- 动作 ----
    ("stop feeding", "停料"),
    ("suspend feeding", "停料"),
    ("stop feed", "停料"),
    ("medicated feed", "药饵"),
    ("reduce feeding", "减料"),
    ("feed", "饲料"),
    ("feeding", "投料"),
    ("turn on the aerator", "开增氧机"),
    ("turn on aerator", "开增氧机"),
    ("start the aerator", "开增氧机"),
    ("shut off the aerator", "关增氧机"),
    ("aerator", "增氧机"),
    ("aeration", "增氧"),
    ("water exchange", "换水"),
    ("change water", "换水"),
    ("add water", "加水"),
    ("drain", "排水"),
    ("lime", "生石灰"),
    ("quicklime", "生石灰"),
    ("zeolite", "沸石粉"),
    ("probiotics", "微生态制剂"),
    ("photosynthetic bacteria", "光合细菌"),
    ("disinfect", "消毒"),
    ("bleaching powder", "漂白粉"),
    ("siphon", "吸污"),
    ("netting", "拉网"),
    ("isolate", "隔离"),
    ("remove", "捞出"),
    ("measure", "检测"),
    ("test", "检测"),
    ("monitor", "监测"),
    ("observe", "观察"),
    ("record", "记录"),
    ("count", "清点"),
    ("hold", "暂养"),
    ("adjust", "调节"),
    ("increase", "提高"),
    ("decrease", "降低"),
    ("raise", "提高"),
    ("lower", "降低"),
    ("keep", "保持"),
    # ---- 时间 ----
    ("immediately", "立即"),
    ("right away", "立即"),
    ("right now", "马上"),
    ("as soon as possible", "尽快"),
    ("tonight", "今晚"),
    ("tomorrow morning", "明早"),
    ("tomorrow evening", "明晚"),
    ("tomorrow", "明天"),
    ("today", "今天"),
    ("this evening", "今晚"),
    ("this morning", "今早"),
    ("morning", "早上"),
    ("evening", "傍晚"),
    ("afternoon", "下午"),
    ("night", "夜里"),
    ("before dawn", "天亮前"),
    ("after dawn", "天亮后"),
    ("dawn", "天亮"),
    ("midday", "中午"),
    ("now", "现在"),
    ("until", "直到"),
    ("for 1-2 days", "1–2 天"),
    ("for 2-3 days", "2–3 天"),
    ("for three days", "3 天"),
    ("for two days", "2 天"),
    ("day", "天"),
    ("days", "天"),
    ("hour", "小时"),
    ("hours", "小时"),
    ("minute", "分钟"),
    ("minutes", "分钟"),
    ("weekly", "每周"),
    ("daily", "每天"),
    ("first", "先"),
    ("then", "然后"),
    # ---- 程度与状态 ----
    ("critical", "高危"),
    ("warning", "预警"),
    ("normal", "正常"),
    ("dangerous", "危险"),
    ("safe", "安全"),
    ("healthy", "健康"),
    ("abnormal", "异常"),
    ("too high", "偏高"),
    ("too low", "偏低"),
    ("high", "高"),
    ("low", "低"),
    ("slightly", "略微"),
    ("seriously", "严重"),
    ("severe", "严重"),
    ("mild", "轻微"),
    ("stable", "稳定"),
    ("rising", "上升"),
    ("falling", "下降"),
    ("drop", "下降"),
    ("below", "低于"),
    ("above", "高于"),
    ("reached", "达到"),
    ("reaches", "达到"),
    ("exceed", "超过"),
    ("limit", "限值"),
    ("threshold", "阈值"),
    ("redline", "红线"),
    # ---- 单位 ----
    ("mg/l", "mg/L"),
    ("mg per liter", "mg/L"),
    ("ppm", "ppm"),
    ("ntu", "NTU"),
    ("celsius", "℃"),
    ("centigrade", "℃"),
    ("degrees c", "℃"),
    # ---- 结构词（译为中文连接方式，或直接省略） ----
    ("and then", "然后"),
    ("all day", "全天"),
    ("at least", "至少"),
    ("up to", "最多"),
    ("no more than", "不超过"),
    ("do not", "不要"),
    ("don't", "不要"),
    ("must", "必须"),
    ("should", "应"),
    ("need to", "需要"),
    ("please", "请"),
    ("if", "如果"),
    ("when", "当"),
    ("while", "同时"),
    ("because", "因为"),
    ("but", "但"),
    ("and", "并"),
    ("or", "或"),
    ("with", "并"),
    ("without", "不"),
    ("for", ""),
    ("the", ""),
    ("a", ""),
    ("an", ""),
    ("to", ""),
    ("of", ""),
    ("in", ""),
    ("on", ""),
    ("at", ""),
    ("by", ""),
    ("is", ""),
    ("are", ""),
    ("was", ""),
    ("were", ""),
    ("be", ""),
    ("been", ""),
    ("it", ""),
    ("its", ""),
    ("this", ""),
    ("that", ""),
    ("these", ""),
    ("those", ""),
    ("there", ""),
)

#: 单位片段：出现在译文里允许原样保留，不算"残留英文"。
_UNIT_ALLOWLIST: frozenset[str] = frozenset({
    "mg", "l", "ml", "g", "kg", "ntu", "ppm", "ph", "c", "h", "min",
})

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
#: 句末中文标点，用于把逐词替换的结果拼得顺一点。
_CJK_PUNCT = "。，、；：！？"
_CJK_RANGE = "\u4e00-\u9fff"
_ASCII_TO_CJK_PUNCT = {
    ".": "。", ",": "，", ";": "；", ":": "：", "!": "！", "?": "？",
    "(": "（", ")": "）",
}


def _polish(text: str, sentence: bool = True) -> str:
    """把逐词替换的结果收拾成能读的中文。

    逐词替换天然会产生两类毛病，都会让养殖户觉得"这东西不是给人看的"：

    1. **中文之间夹空格** —— ``"开始 下降 低于"``；
    2. **中英标点混用** —— ``"溶氧偏低."``。

    这两类都能机械地修好，成本极低，所以在这里统一收拾干净。
    数字与单位之间的空格**要保留**（``"溶氧 2.6 mg/L"`` 比 ``"溶氧2.6mg/L"`` 好读），
    所以只收中文与中文之间的空格。

    ``sentence=False`` 用于**名词短语**（如建议里的 ``target``）。那种字段
    不该有句号——``"氨氮。"`` 作为"针对哪个指标"读起来是个病句，
    而 ``"氨氮"`` 才对。句末标点只在成句的字段上加。
    """
    if not text:
        return ""
    out = text
    out = re.sub(rf"(?<=[{_CJK_RANGE}])[ \t]+(?=[{_CJK_RANGE}])", "", out)
    for ascii_ch, cjk_ch in _ASCII_TO_CJK_PUNCT.items():
        esc = re.escape(ascii_ch)
        out = re.sub(rf"(?<=[{_CJK_RANGE}]){esc}", cjk_ch, out)
        out = re.sub(rf"{esc}(?=[{_CJK_RANGE}])", cjk_ch, out)
    out = re.sub(r"[ \t]+", " ", out).strip()
    out = re.sub(rf"\s+([{_CJK_PUNCT}])", r"\1", out)
    if not sentence:
        # 名词短语：把尾部的句号/句点去掉，其余标点（逗号、顿号）保留。
        return out.rstrip("。.！!？? ").strip()
    if out.endswith("."):
        out = out[:-1] + "。"
    if out and out[-1] not in _CJK_PUNCT:
        out += "。"
    return out


def _norm_key(key: str) -> str:
    return re.sub(r"\s+", " ", key.strip().lower())


def _compile_glossary(
    glossary: tuple[tuple[str, str], ...],
) -> tuple[tuple[re.Pattern[str], str], ...]:
    """把词表编译成"最长优先"的正则列表。

    最长优先很关键：``dissolved oxygen`` 必须先于 ``oxygen`` 匹配，
    否则"溶氧"会被拆成"溶+氧气"。词表按 key 长度降序排，逐个替换。
    同一个 key 出现多次时**后者覆盖前者**（字典语义），
    这样 ``tonight`` 这类重复词以最后一条为准，不会出现随机行为。
    """
    merged: dict[str, str] = {}
    for key, val in glossary:
        merged[_norm_key(key)] = val
    items = sorted(merged.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    compiled: list[tuple[re.Pattern[str], str]] = []
    for key, val in items:
        pattern = re.escape(key).replace(r"\ ", r"\s+")
        if re.fullmatch(r"[a-z0-9/'\-]+", key):
            pattern = rf"(?<![A-Za-z]){pattern}(?![A-Za-z])"
        compiled.append((re.compile(pattern, re.IGNORECASE), val))
    return tuple(compiled)


_GLOSSARY_COMPILED = _compile_glossary(GLOSSARY_ZH)


# --------------------------------------------------------------------------- #
# 三、翻译结果与校验
# --------------------------------------------------------------------------- #

@dataclass
class TranslationOutcome:
    """一个文本字段的回译结果。"""

    text: str
    ok: bool
    coverage: float = 0.0
    unknown: tuple[str, ...] = ()
    backend: str = "glossary"
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.ok and bool(self.text.strip())


def _numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def numbers_preserved(source: str, target: str) -> bool:
    """译文里的数字多重集必须与原文一致。

    用多重集（排序后的列表）而不是集合：原文出现两次 ``2.6``，译文也必须两次。
    这条校验抓的是最危险的一类翻译错误——**数字被漏掉或被改写**。
    养殖场景里"溶氧掉到 1.5"和"溶氧掉到 15"是两个世界。

    Examples:
        >>> numbers_preserved("DO drops to 1.5 mg/L", "溶氧掉到 1.5 mg/L")
        True
        >>> numbers_preserved("DO drops to 1.5 mg/L", "溶氧掉到 15 mg/L")
        False
    """
    return sorted(_numbers(source)) == sorted(_numbers(target))


def _leftover_words(text: str) -> tuple[str, ...]:
    """找出译文里残留的英文单词（排除单位白名单）。"""
    leftovers: list[str] = []
    for raw in _WORD_RE.findall(text):
        token = raw.strip("'-").lower()
        if not token or token in _UNIT_ALLOWLIST:
            continue
        leftovers.append(token)
    # 去重但保持出现顺序，便于报错信息稳定
    seen: dict[str, None] = {}
    for token in leftovers:
        seen.setdefault(token, None)
    return tuple(seen)


# --------------------------------------------------------------------------- #
# 四、后端一：受控词表（离线、确定性）
# --------------------------------------------------------------------------- #

class GlossaryTranslator:
    """受控词表回译。零依赖、完全确定性。

    **这是一个降级路径，不是质量路径。** 它只保证"输出是全中文、数值没被动过"，
    不保证文从字顺 —— 逐词替换做不出地道的汉语句子结构。因此它设了两道闸：

    - ``max_words``：源句英文词数超过这个值就直接放弃。**上限默认只有 6 个词**，
      这个数字是实测出来的：超过 6 词后，即使所有词都被词表覆盖，**语序也会错**。
      例如 ``"Measure dissolved oxygen and ammonia nitrogen again tonight"``
      每个词都认得出，拼出来却是"检测溶氧并氨氮再今晚"——中文该说"今晚再测一次溶氧和氨氮"。
      语序问题不是扩充词表能解决的，所以只能靠限制句子长度来回避。
      长句硬译出来的是"看着像中文、但读不出意思"的文本，比不回译更糟。
    - 残留英文单词：只要还剩一个未覆盖的英文词就放弃。

    任一道闸没过 → 该字段退回规则模板的中文句子。要真正的地道译文，
    请配 ``FISHERIES_TRANSLATOR_*`` 走 :class:`LLMTranslator`。
    """

    backend = "glossary"

    def __init__(self,
                 glossary: tuple[tuple[str, str], ...] = GLOSSARY_ZH,
                 max_words: int = 6) -> None:
        self._compiled = (
            _GLOSSARY_COMPILED if glossary is GLOSSARY_ZH
            else _compile_glossary(glossary)
        )
        self.max_words = max_words

    def translate(self, text: str, *, sentence: bool = True) -> TranslationOutcome:
        source = text or ""
        if not source.strip():
            return TranslationOutcome(text="", ok=True, coverage=1.0,
                                      backend=self.backend)

        n_words = len(_WORD_RE.findall(source))
        if n_words > self.max_words:
            return TranslationOutcome(
                text="", ok=False, coverage=0.0, backend=self.backend,
                note=f"句子含 {n_words} 个英文词，超出词表回译的可靠上限 "
                     f"{self.max_words}，逐词硬译会得到读不通的中文",
            )

        out = source
        for pattern, target in self._compiled:
            out = pattern.sub(lambda m, t=target: t, out)

        out = _polish(out, sentence=sentence)
        leftover = _leftover_words(out)
        total = max(len(re.sub(r"\s+", "", source)), 1)
        coverage = max(0.0, 1.0 - len("".join(leftover)) / total)

        if leftover:
            return TranslationOutcome(
                text="", ok=False, coverage=coverage, unknown=leftover,
                backend=self.backend,
                note="词表未覆盖：" + "、".join(leftover[:6])
                     + ("…" if len(leftover) > 6 else ""),
            )
        if not numbers_preserved(source, out):
            return TranslationOutcome(
                text="", ok=False, coverage=coverage, backend=self.backend,
                note=f"数值未原样保留（原文 {_numbers(source)} → 译文 {_numbers(out)}）",
            )
        return TranslationOutcome(text=out, ok=True, coverage=1.0,
                                  backend=self.backend)


# --------------------------------------------------------------------------- #
# 五、后端二：翻译模型（联网，质量最好）
# --------------------------------------------------------------------------- #

TRANSLATOR_SYSTEM_PROMPT = """\
You are a professional translator for Chinese aquaculture (freshwater pond farming).

Translate the user's text from English into Simplified Chinese for a fish farmer
with limited education.

Hard rules:
1. Keep EVERY number, unit and threshold EXACTLY as written. Never round, never
   drop, never add a number. "1.5 mg/L" must stay "1.5 mg/L".
2. Keep these tokens exactly as-is (they are identifiers, not prose):
   the words 正常 / 预警 / 高危 if present, and any pH / mg/L / NTU / ppm value.
3. Output the Chinese translation ONLY. No explanation, no quotes, no markdown.
4. Keep it short and operational: a farmer must be able to act on it directly.
5. If a term has no common Chinese equivalent, keep the English term in
   parentheses rather than inventing one.
"""


class TranslatorClient(Protocol):
    """翻译模型客户端协议，与 :class:`fisheries.agent.llm.ChatClient` 同构。"""

    def chat(self,
             messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None) -> Any:
        ...


#: 回译时告诉实现"这段是句子还是名词短语"。名词短语不该加句号。
SENTENCE_HINT: dict[bool, str] = {
    True: "text",
    False: "noun phrase",
}


class LLMTranslator:
    """用一个中文能力强的模型整句回译。

    与本地后端一样走"分块 → 逐块译 → 拼回 → 校验"的骨架：远端大模型的
    长句丢尾部比本地模型轻，但**不是没有**，而且块越长单次调用的失败面越大。
    分块还有第二个好处——块是独立的，某一块被拒不会连坐其它块的重试。
    """

    backend = "llm"

    def __init__(self, client: TranslatorClient | None = None, *,
                 chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 chunk_words: int = DEFAULT_CHUNK_WORDS,
                 max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        self._client = client
        self.chunk_chars = chunk_chars
        self.chunk_words = chunk_words
        self.max_chunks = max_chunks

    def _one_call(self, chunk: str, *, sentence: bool = True) -> TranslationOutcome:
        try:
            resp = self._client.chat(                    # type: ignore[union-attr]
                [
                    {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
                    {"role": "user",
                     "content": chunk + "\n\n(Expected output form: "
                              + SENTENCE_HINT[sentence]
                              + ". Do not add a full stop to a noun phrase.)"},
                ],
                tools=None,
            )
        except Exception as exc:                       # noqa: BLE001 - 见下
            # 翻译失败绝不能中断整条 Agent 链路：调用方会把该字段退回中文模板。
            # 这里兜住所有异常（含网络、鉴权、协议），只上报原因。
            return TranslationOutcome(
                text="", ok=False, backend=self.backend,
                note=f"翻译模型调用失败：{type(exc).__name__}: {exc}",
            )

        out = str(getattr(resp, "content", "") or "").strip()
        if not out:
            return TranslationOutcome(text="", ok=False, backend=self.backend,
                                      note="翻译模型返回空内容")
        if not numbers_preserved(chunk, out):
            return TranslationOutcome(
                text="", ok=False, backend=self.backend,
                note=f"译文数值与原文不符（{_numbers(chunk)} → {_numbers(out)}）",
            )
        return TranslationOutcome(text=out, ok=True, coverage=1.0,
                                  backend=self.backend)

    def translate(self, text: str, *, sentence: bool = True) -> TranslationOutcome:
        source = text or ""
        if not source.strip():
            return TranslationOutcome(text="", ok=True, coverage=1.0,
                                      backend=self.backend)
        if self._client is None:
            return TranslationOutcome(
                text="", ok=False, backend=self.backend,
                note="未配置翻译模型",
            )
        return _translate_chunked(
            source, sentence=sentence, translate_one=self._one_call,
            backend=self.backend, max_chars=self.chunk_chars,
            max_words=self.chunk_words, max_chunks=self.max_chunks,
        )


class BypassTranslator:
    """不做回译，直接交出英文原文。

    **只在调提示词时用。** 它存在的意义是让"提示词是否写得让模型听得懂"
    和"回译是否可靠"这两件事可以分开排查——否则回译失败会把提示词的问题掩盖掉。
    """

    backend = "bypass"

    def translate(self, text: str, *, sentence: bool = True) -> TranslationOutcome:
        return TranslationOutcome(text=text or "", ok=True, coverage=1.0,
                                  backend=self.backend,
                                  note="bypass：未回译，保留英文原文")


# --------------------------------------------------------------------------- #
# 五之二、分块：长句为什么必须切开
# --------------------------------------------------------------------------- #
#
# 回译的输入不是"一个词"，而是**整段英文建议**。让模型一次吞下整段时，
# 实测会出现三类错乱：
#
# 1. **尾部被整段丢掉。** ``"Turn on the aerator tonight and measure
#    dissolved oxygen and ammonia nitrogen again tomorrow morning at 6."``
#    实测只译出「今晚打开气动器 测量溶解氧气和氨氮」——``again``、
#    ``tomorrow morning``、``at 6`` 全没了，**连数字 6 都丢了**。
#    数值校验会发现"数字少了"，但不知道少的是哪一句，只能整条放弃；
# 2. **语气串味。** 一段里同时有"当前观察"和"建议动作"时，模型会把两者揉成一句；
# 3. **指代漂移。** 代词离先行词太远，译文会把"它"指到别的东西上。
#
# 所以策略是**先切、再逐块译、最后拼**：每块的输入长度有硬上限，模型每次
# 只处理一个完整小句，上面三类错乱都没有发生的空间。切分守两条硬约束：
# **不切在数字与单位之间**（``1.5 mg/L`` 是一个整体）、**不切碎名词短语**。

_SENT_BOUNDARY = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s)")
_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[,，、:：])|(?<=\s)(?=(?:and|or|then|but|while|because|so|after"
    r"|before|until|when|if)\s)",
    re.IGNORECASE,
)
#: 纯数字 token（含 ``1.5`` / ``0.45`` / ``2-3`` / ``>5``）。小数点在内部，不影响。
_NUMBER_TOKEN = re.compile(r"^[<>≤≥~≈约]?\d[\d.,:/%-]*$")
#: 单位 token。数字后面的这些词必须和数字留在同一个块里。
_UNIT_TOKEN = re.compile(
    r"^(?:mg/l|mg|g/l|µg/l|ug/l|ntu|ppm|ppt|℃|°c|%|h|hr|hrs|hours?|min|mins"
    r"|minutes?|days?|weeks?|c)$",
    re.IGNORECASE,
)


def _glue_number_units(tokens: list[str]) -> list[list[str]]:
    """把 ``1.5`` 与紧随其后的 ``mg/L`` 粘成一组，硬切分时不许切在中间。"""
    groups: list[list[str]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if (_NUMBER_TOKEN.match(token) and i + 1 < len(tokens)
                and _UNIT_TOKEN.match(tokens[i + 1])):
            groups.append([token, tokens[i + 1]])
            i += 2
            continue
        groups.append([token])
        i += 1
    return groups


def _pack(groups: list[list[str]], max_chars: int, max_words: int) -> list[str]:
    """把词组按预算装箱，尽量少切。"""
    out: list[str] = []
    buf: list[str] = []
    for group in groups:
        candidate = buf + group
        if buf and (len(" ".join(candidate)) > max_chars
                    or len(candidate) > max_words):
            out.append(" ".join(buf))
            buf = list(group)
        else:
            buf = candidate
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_text(text: str,
               *,
               max_chars: int = DEFAULT_CHUNK_CHARS,
               max_words: int = DEFAULT_CHUNK_WORDS) -> list[str]:
    """把一段英文切成"每块都短到不会让模型丢尾部"的小块。

    切分顺序是**从贵到便宜**：先按句子切（最自然），句子仍然超预算就按逗号、
    分号这类小句边界切，还超预算才按词硬切——而硬切会成组地保留
    ``数字 + 单位``，不会把 ``1.5 mg/L`` 拆到两个块里。

    切完还会**把相邻的碎片合并回去**，避免一句话被切成五段读起来断气。

    Examples:
        >>> chunk_text("Turn on the aerator tonight.", max_chars=200)
        ['Turn on the aerator tonight.']
        >>> len(chunk_text("a b c d e f g h", max_chars=5, max_words=3))
        3
    """
    source = (text or "").strip()
    if not source:
        return []

    def within(piece: str) -> bool:
        return len(piece) <= max_chars and len(piece.split()) <= max_words

    if within(source):
        return [source]

    pieces: list[str] = []
    for sentence in _SENT_BOUNDARY.split(source):
        sentence = sentence.strip()
        if not sentence:
            continue
        if within(sentence):
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_BOUNDARY.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if within(clause):
                pieces.append(clause)
            else:
                pieces.extend(_pack(_glue_number_units(clause.split()),
                                    max_chars, max_words))

    merged: list[str] = []
    for piece in pieces:
        if merged:
            joined = f"{merged[-1]} {piece}"
            if within(joined):
                merged[-1] = joined
                continue
        merged.append(piece)
    return merged


def _join_chunks(parts: list[str]) -> str:
    """把逐块译文拼回一段中文。

    只收拾"中文之间夹空格"这一类——模型自己产出的标点、数字与单位
    **一律不动**，那是它比词表回译强的地方。
    """
    text = "".join(p.strip() for p in parts if p and p.strip())
    text = re.sub(rf"(?<=[{_CJK_RANGE}])[ \t]+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def _translate_chunked(source: str,
                       *,
                       sentence: bool,
                       translate_one: Any,
                       backend: str,
                       max_chars: int = DEFAULT_CHUNK_CHARS,
                       max_words: int = DEFAULT_CHUNK_WORDS,
                       max_chunks: int = DEFAULT_MAX_CHUNKS) -> TranslationOutcome:
    """分块翻译的公共骨架，``LLMTranslator`` 与 ``LocalTranslator`` 共用。

    两条判定：

    - **任一块译不出来 → 整条字段放弃。** 不做"译一半留一半"——半英半中
      是这份代码的底线问题（见模块 docstring）。
    - **拼起来之后再做一次数值校验。** 逐块校验只能保证每块的数字没丢，
      拼接时仍可能多出或漏掉一个（例如相邻块边界上的数字被吃进标点里）。

    只有最后一块带 ``sentence`` 语义：中间块加句号会把一句话切成好几句。
    """
    chunks = chunk_text(source, max_chars=max_chars, max_words=max_words)
    if not chunks:
        return TranslationOutcome(text="", ok=True, coverage=1.0, backend=backend)
    if len(chunks) > max_chunks:
        return TranslationOutcome(
            text="", ok=False, backend=backend,
            note=f"切出 {len(chunks)} 个块，超出单次回译上限 {max_chunks}，整条放弃",
        )

    outcomes: list[TranslationOutcome] = []
    for index, chunk in enumerate(chunks, 1):
        out = translate_one(chunk, sentence=sentence and index == len(chunks))
        if not out.ok:
            return TranslationOutcome(
                text="", ok=False, backend=backend,
                unknown=out.unknown,
                note=(f"第 {index}/{len(chunks)} 块回译失败"
                      f"（{out.note or '未给原因'}）"),
            )
        outcomes.append(out)

    joined = _join_chunks([o.text for o in outcomes])
    if not joined:
        return TranslationOutcome(text="", ok=False, backend=backend,
                                  note="分块译文拼起来是空的")
    if not numbers_preserved(source, joined):
        return TranslationOutcome(
            text="", ok=False, backend=backend,
            note=(f"拼起来后数值与原文不符"
                  f"（{_numbers(source)} → {_numbers(joined)}）"),
        )
    leftover = _leftover_words(joined)
    if leftover:
        return TranslationOutcome(
            text="", ok=False, backend=backend, unknown=leftover,
            note="译文仍残留英文：" + "、".join(leftover[:6]),
        )
    coverage = sum(o.coverage for o in outcomes) / len(outcomes)
    return TranslationOutcome(text=joined, ok=True, coverage=coverage,
                              backend=backend)


# --------------------------------------------------------------------------- #
# 五之三、译文后订正：通用 NMT 不懂水产
# --------------------------------------------------------------------------- #
#
# 本地翻译模型是**通用域**的。实测把它接上以后，它译得很通顺，但把每个
# 领域词都译错了：
#
# ======================  ====================  ==================
# 英文                    模型给出的            本工程要的
# ======================  ====================  ==================
# dissolved oxygen        溶解氧 / 破解的氧气     溶氧
# nitrite nitrogen        亚硝酸氮               亚硝酸盐
# turbidity               多发性                 浊度
# bacterial gill disease  细菌刺痛病             烂鳃病
# saprolegniasis          沙普罗尼丝虫病          水霉病
# aerator                 呼吸器 / 气动器         增氧机
# aeration                发热                   增氧
# stocking density        丝袜密度               放养密度
# probiotics              先生生物               益生菌
# celsius                 西里乌斯 / 中度          ℃
# ======================  ====================  ==================
#
# 这些错法**不是模型差，是它没有这个领域的词表**。词表回译当初的失败
# 正好相反：词认得、语序拼不出来。所以两者是互补的——**让 NMT 负责语序、
# 让词表负责术语**：NMT 先出通顺的中文，再用一张"中文→中文"的后订正表
# 把术语扳回来。
#
# 这张表只收**在本领域里不可能有第二种解释**的错法。像 ``措施``（measure 被
# 译成"措施"）就不收——``采取措施`` 是正当中文，全局替换会改坏它。

#: 中文译文后订正表：``(模型给的错法, 本工程要的说法)``。
#:
#: 表里的每一条都有实跑记录，不是猜的（见 ``docs/M10_回译层说明.md``）。
#: **收录标准只有一条：在本领域里不可能有第二种解释。**
#: 反例（**故意不收**）：
#:
#: - ``措施``（``measure`` 会被译成"措施"，但"采取措施"是正当中文）；
#: - ``发病率``（``turbidity`` 偶尔被译成"发病率"，但"发病率"本身是
#:   正当的病害统计词，全局替换会把病害句子改坏）；
#: - ``中度``（``centigrade`` 会被译成"中度"，但"中度污染"是正当中文）。
GLOSSARY_POST_ZH: tuple[tuple[str, str], ...] = (
    # ---- 指标 ----
    # 模型把 dissolved oxygen 译成"溶解氧 / 溶解的氧气 / 溶氧气"三种，
    # 本工程的统一说法是"溶氧"。
    ("溶解的氧气", "溶氧"),
    ("溶解氧气", "溶氧"),
    ("溶氧气", "溶氧"),
    ("溶解氧", "溶氧"),
    ("亚硝酸氮", "亚硝酸盐"),
    # 上面这条的替换结果以"亚硝酸盐"开头，而"亚硝酸"本身就是一条规则
    # （见下），交替正则按最长优先匹配，"亚硝酸盐"没有自己的条目时
    # 会被"亚硝酸"吃掉前三个字再补一个"盐"字，得到"亚硝酸盐盐"。
    # 这条恒等规则的作用是**占位**：让"亚硝酸盐"作为一个整体被消费掉。
    # 不变量见 tests/unit/test_translate.py
    # ::TestPostEdit::test_every_replacement_is_itself_stable。
    ("亚硝酸盐", "亚硝酸盐"),
    ("亚硝酸", "亚硝酸盐"),
    ("非离子化氨", "分子氨"),
    # turbidity 的错法：多发性 / 发炎率 / 易发性。
    # "发病率"故意不收（见上）。
    ("多发性", "浊度"),
    ("发炎率", "浊度"),
    ("易发性", "浊度"),
    ("氨氮氮", "氨氮"),
    # ---- 病害 ----
    # bacterial gill disease / gill rot 的错法。
    ("细菌刺痛病", "烂鳃病"),
    ("细菌刺菌病", "烂鳃病"),
    ("细菌刺状疾病", "烂鳃病"),
    ("细菌刺状病", "烂鳃病"),
    ("刺状腐烂", "烂鳃病"),
    ("细菌红病", "细菌性败血症"),
    ("出血性败血症", "细菌性出血病"),
    ("沙普罗尼丝虫病", "水霉病"),
    ("萨普罗格尼亚", "水霉病"),
    # ---- 塘口动作与投入品 ----
    # aerator 的错法：呼吸器 / 气动器 / 加速器 / 电动器。
    ("呼吸器", "增氧机"),
    ("气动器", "增氧机"),
    ("加速器", "增氧机"),
    ("电动器", "增氧机"),
    # aeration 的错法：发热 / 增加气流 / 增加呼吸 / 增加空气循环。
    ("增加空气循环", "增氧"),
    ("增加气流", "增氧"),
    ("增加呼吸", "增氧"),
    ("发热", "增氧"),
    ("丝袜密度", "放养密度"),
    ("先生生物", "益生菌"),
    ("佐利特", "沸石"),
    # quicklime 的错法：速石 / 速溶液 / 速力石 / 快生石灰灰 / 快生石灰。
    ("快生石灰灰", "生石灰"),
    ("快生石灰", "生石灰"),
    ("速溶液", "生石灰"),
    ("速力石", "生石灰"),
    ("速石", "生石灰"),
    ("漂白粉粉", "漂白粉"),
    ("网状图", "拉网"),
    ("海底沉积物", "底泥"),
    ("药物喂养", "药饵"),
    ("水交换", "换水"),
    ("人口", "存塘量"),
    # ---- 时间（tonight 被译成"表演"，或残留重复）----
    ("今夜今晚", "今晚"),
    ("表演", "今晚"),
)

#: 单位规范化：NMT 习惯把单位翻成中文甚至翻成人名，而本工程的铁律是
#: **单位当标识符原样保留**。
#:
#: ``NTU`` 的错法尤其离谱——实测被译成"纽芬兰先令"（Newfoundland shilling）。
#: 这类错法是开放的（换个句子可能又变一个词），所以这张表和后订正表一样，
#: **只能覆盖实测到的**；发现新的就补一行，这一条写进了使用说明。
UNIT_NORMALIZE: tuple[tuple[str, str], ...] = (
    ("纽芬兰先令", "NTU"),
    ("毫克/升", "mg/L"),
    ("毫克∕升", "mg/L"),
    ("毫克每升", "mg/L"),
    ("mg/升", "mg/L"),
    ("毫克/公升", "mg/L"),
    ("微克/升", "μg/L"),
    ("微克每升", "μg/L"),
    ("摄氏度", "℃"),
    ("西里乌斯", "℃"),
    ("°C", "℃"),
    ("百万分之", "ppm"),
)


def _compile_alternation(mapping: tuple[tuple[str, str], ...]) -> re.Pattern[str]:
    """把替换表编成**一次扫描**的交替正则，保证替换不会级联。

    为什么不能像英文词表那样逐条 ``sub``：中文后订正表里存在包含关系
    （``亚硝酸`` ⊂ ``亚硝酸盐``）。逐条替换时先把 ``亚硝酸氮`` 改成
    ``亚硝酸盐``，下一条 ``亚硝酸`` 又把刚改好的结果切掉一半。
    一次扫描 + 最长优先就没有这个问题。
    """
    keys = sorted({k for k, _ in mapping}, key=lambda k: (-len(k), k))
    return re.compile("|".join(re.escape(k) for k in keys))


_POST_ZH_LOOKUP = dict(GLOSSARY_POST_ZH)
_POST_ZH_RE = _compile_alternation(GLOSSARY_POST_ZH)
_UNIT_LOOKUP = dict(UNIT_NORMALIZE)
_UNIT_RE = _compile_alternation(UNIT_NORMALIZE)


def post_edit_terms(text: str) -> str:
    """把 NMT 译文里的领域术语扳回本工程的说法。"""
    if not text:
        return ""
    return _POST_ZH_RE.sub(lambda m: _POST_ZH_LOOKUP[m.group(0)], text)


def normalize_units(text: str) -> str:
    """把 NMT 译出来的中文单位扳回 ``mg/L`` / ``℃`` 这类标识符。

    数字与单位之间补一个空格，让 ``1.5毫克/升`` 读成 ``1.5 mg/L``，
    与报告里其他位置的写法一致。
    """
    if not text:
        return ""
    out = _UNIT_RE.sub(lambda m: _UNIT_LOOKUP[m.group(0)], text)
    out = re.sub(r"(?<=\d)(?=[A-Za-zμ])", " ", out)
    out = re.sub(r"(?<=\d)(?=℃)", " ", out)
    return out


def post_edit(text: str) -> str:
    """后订正流水线：先规范单位，再扳术语。

    顺序不能反：单位里也含中文字（``毫克/升``），先扳术语会把
    ``毫克/升`` 这种整体切碎（表里没有它，但术语表以后可能有）。
    """
    return post_edit_terms(normalize_units(text or ""))


# --------------------------------------------------------------------------- #
# 五之四、后端三：本地离线翻译模型
# --------------------------------------------------------------------------- #

#: 本地翻译模型的**名字**（相对 ``models/translate/``）或绝对路径。
LOCAL_MODEL_ENV = TRANSLATOR_ENV_PREFIX + "LOCAL_MODEL"

#: 本地翻译模型的**根目录**，默认 ``<仓库根>/models/translate``。
LOCAL_DIR_ENV = TRANSLATOR_ENV_PREFIX + "LOCAL_DIR"

#: 默认模型：专用英中 NMT，约 77M 参数，CPU 上单句约 0.5 秒。
DEFAULT_LOCAL_MODEL = "opus-mt-en-zh"

#: 这些文件名命中一个就算"有权重"。
_LOCAL_WEIGHT_NAMES: tuple[str, ...] = (
    "model.safetensors", "pytorch_model.bin", "model.onnx",
    "onnx/model.onnx", "onnx/model_quantized.onnx",
)

#: 已加载的模型缓存：``路径 -> (tokenizer, model, kind)``。
#: 一次回译要过十几个字段，不缓存就会把 300 MB 权重加载十几遍。
_LOCAL_CACHE: dict[str, tuple[Any, Any, str]] = {}


def clear_local_cache() -> None:
    """丢掉已加载的本地模型（测试用；也给长时间运行的服务一个释放内存的入口）。"""
    _LOCAL_CACHE.clear()


def local_model_root(env: dict[str, str] | None = None) -> Path:
    """本地翻译模型的根目录：``FISHERIES_TRANSLATOR_LOCAL_DIR`` 或 ``models/translate``。"""
    e = dict(os.environ if env is None else env)
    raw = str(e.get(LOCAL_DIR_ENV, "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[3] / "models" / "translate"


def local_model_path(model: str | None = None,
                     env: dict[str, str] | None = None) -> Path:
    """解析出具体模型目录。``model`` 为空时看环境变量，再退回默认模型名。"""
    e = dict(os.environ if env is None else env)
    name = str(model or e.get(LOCAL_MODEL_ENV, "") or DEFAULT_LOCAL_MODEL).strip()
    candidate = Path(name).expanduser()
    return candidate if candidate.is_absolute() else local_model_root(env) / candidate


def local_model_available(model: str | None = None,
                          env: dict[str, str] | None = None) -> tuple[bool, str]:
    """本地模型能不能用；不能用时给出**人能照做**的原因。

    检查顺序刻意从"最容易缺的东西"开始：目录 → 权重 → 依赖。
    返回的 ``str`` 会进 ``degraded_reasons``，所以必须是可执行的提示，
    不能是 ``FileNotFoundError`` 那种栈信息。
    """
    path = local_model_path(model, env)
    if not path.is_dir():
        return False, f"没有本地翻译模型目录 {path}（先跑 scripts/fetch_translate_model.py）"
    if not (path / "config.json").exists():
        return False, f"{path} 缺少 config.json"
    if not any((path / n).exists() for n in _LOCAL_WEIGHT_NAMES):
        return False, f"{path} 里没有权重文件（model.safetensors / pytorch_model.bin）"
    if importlib.util.find_spec("transformers") is None:
        return False, "没装 transformers（pip install -e .[translate]）"
    return True, ""


def _loadable_path(path: Path) -> str:
    """给 sentencepiece 一个它**打得开**的路径字符串。

    sentencepiece 的 C++ 加载器在 Windows 上走窄字符路径，路径里只要出现
    非 ASCII 字符就报 ``NOT_FOUND``——本工程恰好就在 ``D:\\设置\\项目\\...``
    下，实测绝对路径必然失败、相对路径（纯 ASCII）正常。所以非 ASCII 时
    先退到相对路径；退不了（跨盘符、或 cwd 本身也是中文）就原样返回，
    那种情况下绝对路径在纯 ASCII 的工程目录里本来就能工作。
    """
    text = str(path)
    if text.isascii():
        return text
    try:
        relative = os.path.relpath(path)
    except ValueError:                              # 跨盘符
        return text
    return relative if relative.isascii() else text


class LocalTranslator:
    """用 ``models/translate/`` 里的**本地离线模型**做整句回译。

    它是 D12 收益真正兑现的那条路径：不联网、不需要 key，也能把 AQUA-7B
    的整句英文译成中文（词表后端遇到自由长句只能整条放弃）。

    管线是四步，缺一不可::

        分块  →  NMT 逐块翻译  →  单位规范化  →  领域术语后订正  →  数值校验

    - **分块**：见 ``chunk_text``，解决长句丢尾部；
    - **NMT**：Marian 系 seq2seq 或 Qwen 系 causal LM，按 ``config`` 自动选；
    - **后订正**：见 ``post_edit``。通用 NMT 不懂水产，实测会把
      ``dissolved oxygen`` 译成「破解的氧气」、``turbidity`` 译成「多发性」，
      不做后订正的话本地模型反而比模板更难读；
    - **数值校验**：与其余后端一致，数字多重集必须完全不变。

    依赖（``transformers`` / ``torch`` / ``sentencepiece``）**全部惰性导入**：
    没装它们时 ``import fisheries.agent.translate`` 照常成功，只是
    :meth:`available` 返回 ``False``、构造方退回词表后端。
    """

    backend = "local"

    def __init__(self,
                 model: str | None = None,
                 *,
                 env: dict[str, str] | None = None,
                 device: str | None = None,
                 max_new_tokens: int = 200,
                 num_beams: int = 4,
                 chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 chunk_words: int = DEFAULT_CHUNK_WORDS,
                 max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        self.model_path = local_model_path(model, env)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.chunk_chars = chunk_chars
        self.chunk_words = chunk_words
        self.max_chunks = max_chunks
        self._tokenizer: Any = None
        self._model: Any = None
        self._kind: str = ""
        self._load_error: str = ""

    # -- 可用性 ---------------------------------------------------------- #

    def available(self) -> tuple[bool, str]:
        """能不能真的拿来翻译。返回 ``(可用?, 不可用的原因)``。"""
        return local_model_available(str(self.model_path))

    # -- 加载 ------------------------------------------------------------ #

    def _load(self) -> None:
        """惰性加载权重。只在第一次 ``translate`` 时发生。"""
        if self._model is not None:
            return
        cache_key = str(self.model_path)
        cached = _LOCAL_CACHE.get(cache_key)
        if cached is not None:
            self._tokenizer, self._model, self._kind = cached
            return

        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)

        import torch                                   # noqa: PLC0415 - 惰性
        from transformers import (                     # noqa: PLC0415 - 惰性
            AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
        )
        # ``from_pretrained`` 默认给权重加载画一条 tqdm 进度条（258 个分片）。它是
        # 写给 stderr 的，在演示和自检里都被 PowerShell 当成"命令出错"高亮出来。
        # 加载是幂等的、耗时固定，这条进度条没有信息量，关掉。
        try:
            from transformers.utils import logging as _hf_logging  # noqa: PLC0415
            _hf_logging.disable_progress_bar()
        except Exception:  # pragma: no cover - 版本差异，关不掉也不影响功能
            pass

        key = _loadable_path(self.model_path)
        config = AutoConfig.from_pretrained(key)
        tokenizer = AutoTokenizer.from_pretrained(key)
        if getattr(config, "is_encoder_decoder", False):
            net = AutoModelForSeq2SeqLM.from_pretrained(key)
            kind = "seq2seq"
        else:
            net = AutoModelForCausalLM.from_pretrained(key)
            kind = "causal"
        net.eval()
        if self.device:
            net.to(self.device)
        # 模型的 ``generation_config.json`` 自带 ``max_length=512``，而我们要的是
        # ``max_new_tokens``。两者同时存在时 transformers 每次 generate 都会
        # 往 stderr 打一行 "Both max_new_tokens and max_length seem to have been
        # set"——一次回译十几个字段就是十几行噪声，演示时很扎眼。清掉它。
        if getattr(net, "generation_config", None) is not None:
            net.generation_config.max_length = None
        self._tokenizer, self._model, self._kind = tokenizer, net, kind
        _LOCAL_CACHE[cache_key] = (tokenizer, net, kind)
        self._torch = torch

    # -- 单块翻译 -------------------------------------------------------- #

    def _generate(self, chunk: str, *, sentence: bool) -> str:
        import torch                                   # noqa: PLC0415 - 惰性
        tokenizer, net = self._tokenizer, self._model
        with torch.no_grad():
            if self._kind == "seq2seq":
                encoded = tokenizer(chunk, return_tensors="pt",
                                    truncation=True, max_length=512)
                ids = net.generate(**encoded,
                                   max_new_tokens=self.max_new_tokens,
                                   num_beams=self.num_beams)
                return tokenizer.decode(ids[0], skip_special_tokens=True)
            messages = [
                {"role": "system", "content": TRANSLATOR_SYSTEM_PROMPT},
                {"role": "user",
                 "content": chunk + "\n\n(Expected output form: "
                            + SENTENCE_HINT[sentence]
                            + ". Do not add a full stop to a noun phrase.)"},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1024)
            ids = net.generate(**encoded, max_new_tokens=self.max_new_tokens,
                               do_sample=False)
            new_tokens = ids[0][encoded["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _one_call(self, chunk: str, *, sentence: bool = True) -> TranslationOutcome:
        try:
            self._load()
        except Exception as exc:                       # noqa: BLE001 - 见下
            # 加载失败绝不能中断 Agent 链路：调用方会把该字段退回中文模板。
            self._load_error = f"{type(exc).__name__}: {exc}"
            return TranslationOutcome(text="", ok=False, backend=self.backend,
                                      note=f"本地翻译模型加载失败：{self._load_error}")
        try:
            raw = self._generate(chunk, sentence=sentence)
        except Exception as exc:                       # noqa: BLE001 - 同上
            return TranslationOutcome(
                text="", ok=False, backend=self.backend,
                note=f"本地翻译模型推理失败：{type(exc).__name__}: {exc}",
            )

        text = post_edit(str(raw or "").strip())
        if not text:
            return TranslationOutcome(text="", ok=False, backend=self.backend,
                                      note="本地翻译模型返回空内容")
        if not numbers_preserved(chunk, text):
            return TranslationOutcome(
                text="", ok=False, backend=self.backend,
                note=f"译文数值与原文不符（{_numbers(chunk)} → {_numbers(text)}）",
            )
        return TranslationOutcome(text=text, ok=True, coverage=1.0,
                                  backend=self.backend)

    # -- 对外入口 -------------------------------------------------------- #

    def translate(self, text: str, *, sentence: bool = True) -> TranslationOutcome:
        source = text or ""
        if not source.strip():
            return TranslationOutcome(text="", ok=True, coverage=1.0,
                                      backend=self.backend)
        return _translate_chunked(
            source, sentence=sentence, translate_one=self._one_call,
            backend=self.backend, max_chars=self.chunk_chars,
            max_words=self.chunk_words, max_chunks=self.max_chunks,
        )


def build_translator(mode: str | None = None,
                     *,
                     client: TranslatorClient | None = None,
                     env: dict[str, str] | None = None) -> GlossaryTranslator | LLMTranslator | BypassTranslator:
    """按模式构造回译后端。

    分派规则：

    - ``local``：本地离线模型。**模型缺失时退回词表**，不抛异常；
    - ``glossary`` / ``llm`` / ``bypass``：显式指定，照办（``llm`` 没 client
      时会在翻译时上报"未配置翻译模型"，不在这里拦）；
    - ``auto``：**本地模型优先**，其次远端 client，最后词表。

    ``auto`` 的顺序是这一层收益的关键。以前 auto 只看有没有 client，
    离线机器上必然落到词表后端，而词表后端对超过 ``max_words`` 的自由长句
    只能整条拒绝（``build_translator`` 的调用方随后退回规则模板），
    于是**AQUA-7B 的全部输出被丢掉、D12 的收益等于 0**。把本地模型放到
    第一位，离线也能把整句英文译成中文。

    任何失败都**不抛异常**：评委机器上没网、没 key、没下模型，链路都要照常走完。
    """
    resolved = resolve_translator_mode(mode, env)
    if resolved == "bypass":
        return BypassTranslator()
    if resolved == "glossary":
        return GlossaryTranslator()
    if resolved == "llm":
        return LLMTranslator(client)
    if resolved == "local":
        local = LocalTranslator(env=env)
        if local.available()[0]:
            return local
        return GlossaryTranslator()
    # auto：本地离线模型优先 → 远端中文模型 → 词表兜底。
    local = LocalTranslator(env=env)
    if local.available()[0]:
        return local
    return LLMTranslator(client) if client is not None else GlossaryTranslator()


# --------------------------------------------------------------------------- #
# 六、逐字段回译
# --------------------------------------------------------------------------- #

#: 报告里需要回译的自由文本字段。
TEXT_FIELDS: tuple[str, ...] = ("headline", "observation_window", "confidence_note")

#: 动作条目里需要回译的字段。``priority`` 是数字，不动。
ACTION_TEXT_FIELDS: tuple[str, ...] = ("target", "action", "timing")

#: 动作清单字段名 → 规则模板里对应的字段名。
ACTION_LIST_FIELDS: tuple[str, ...] = ("water_actions", "disease_actions")


@dataclass
class BackTranslation:
    """整份报告的回译结果。"""

    parsed: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    n_translated: int = 0
    n_replaced: int = 0
    backend: str = "glossary"
    ok: bool = True

    @property
    def replaced(self) -> bool:
        return self.n_replaced > 0


def _translate_field(value: Any, translator: Any,
                     *, sentence: bool = True) -> TranslationOutcome | None:
    """回译一个字符串字段；非字符串或空白返回 ``None``（表示"无需处理"）。

    ``sentence=False`` 用于名词短语字段（建议条目的 ``target``）：那种字段
    加句号会读成病句（``"氨氮。"``），而 ``translate`` 的实现都会被这个
    标志告知该不该收尾。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return translator.translate(value, sentence=sentence)


def _fallback_text(fallback: dict[str, Any], key: str, index: int | None = None) -> str:
    """从规则模板里取兜底中文。取不到就返回空串（调用方会整条丢弃）。"""
    if index is None:
        return str(fallback.get(key, "") or "").strip()
    items = fallback.get(key) or []
    if isinstance(items, list) and 0 <= index < len(items):
        item = items[index]
        if isinstance(item, dict):
            return str(item.get("target", "") or "").strip()
    return ""


def back_translate_report(
    parsed: dict[str, Any],
    *,
    translator: Any,
    fallback: dict[str, Any] | None = None,
    lang: str = LANG_EN,
) -> BackTranslation:
    """把英文建议报告逐字段回译成中文。

    逐字段的策略，每条都有明确理由：

    - 单个文本字段（``headline`` / ``observation_window`` / ``confidence_note``）：
      译得出来就用译文，译不出来就用规则模板的同名字段。
      **不保留英文原文**——留着就是半英半中。
    - 动作条目：逐条判定。某条的 ``target`` / ``action`` / ``timing`` 任一处译不出来，
      这一条就用模板里同序号的对应条目补位；模板也没有就丢弃这一条并记轨迹。
      逐条而不是整表替换，是为了不因为一条译不出来就丢掉其余译得好的建议。
    - ``risk_level`` / ``alerts``：**不回译**。它们是机器比对的标识符与
      融合层的确定性结果，回译只会引入风险。英文提示词里已要求原样照抄中文枚举值。

    Args:
        parsed: 大模型返回的 JSON（英文）。
        translator: 具备 ``translate(text) -> TranslationOutcome`` 的对象。
        fallback: 规则模板报告（``AdvisorReport.model_dump(mode="json")``）。
        lang: 原文语言。``zh`` 时直接原样返回。

    Returns:
        :class:`BackTranslation`，``parsed`` 是回译后的新字典（不改原对象）。
    """
    if lang != LANG_EN:
        return BackTranslation(parsed=dict(parsed), backend=getattr(translator, "backend", "none"))

    fb = fallback or {}
    result = dict(parsed)
    out = BackTranslation(parsed=result,
                          backend=getattr(translator, "backend", "unknown"))

    # ---- 单字段 ----
    for key in TEXT_FIELDS:
        outcome = _translate_field(parsed.get(key), translator)
        if outcome is None:
            continue
        if outcome.usable:
            result[key] = outcome.text
            out.n_translated += 1
            continue
        replacement = _fallback_text(fb, key)
        if replacement:
            result[key] = replacement
            out.n_replaced += 1
            out.notes.append(
                f"字段 {key} 回译失败（{outcome.note or '原因不明'}），"
                f"已改用规则模板的中文表述"
            )
        else:
            result.pop(key, None)
            out.n_replaced += 1
            out.notes.append(
                f"字段 {key} 回译失败（{outcome.note or '原因不明'}），"
                f"模板无对应内容，已丢弃该字段"
            )

    # ---- 动作清单 ----
    for list_key in ACTION_LIST_FIELDS:
        raw_items = parsed.get(list_key)
        if not isinstance(raw_items, list) or not raw_items:
            continue
        fb_items = fb.get(list_key) or []
        kept: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            failed: str | None = None
            for fkey in ACTION_TEXT_FIELDS:
                outcome = _translate_field(
                item.get(fkey), translator, sentence=(fkey != "target"),
            )
                if outcome is None:
                    continue
                if outcome.usable:
                    new_item[fkey] = outcome.text
                    out.n_translated += 1
                else:
                    failed = f"{fkey}（{outcome.note or '原因不明'}）"
                    break
            if failed is None:
                kept.append(new_item)
                continue

            substitute = None
            if isinstance(fb_items, list) and idx < len(fb_items):
                candidate = fb_items[idx]
                if isinstance(candidate, dict):
                    substitute = dict(candidate)
            if substitute is not None:
                kept.append(substitute)
                out.notes.append(
                    f"{list_key} 第 {idx + 1} 条回译失败，已改用规则模板的同序条目"
                )
            else:
                out.notes.append(
                    f"{list_key} 第 {idx + 1} 条回译失败，模板无对应条目，已丢弃"
                )
            out.n_replaced += 1
        result[list_key] = kept

    out.ok = out.n_replaced == 0
    return out


def describe_translator(translator: Any) -> str:
    """人类可读的回译后端描述，用于启动日志与 demo 输出。

    本地后端额外报出**用的是哪个模型**——"回译后端：本地离线模型"不足以
    判断离线收益能不能兑现，得知道它到底加载了 `opus-mt-en-zh` 还是别的。
    """
    backend = getattr(translator, "backend", "unknown")
    human = {
        "glossary": "受控词表回译（离线、确定性）",
        "llm": "翻译模型回译",
        "local": "本地离线翻译模型",
        "bypass": "不回译（调试用，保留英文）",
    }.get(backend, backend)
    if backend == "local":
        path = getattr(translator, "model_path", None)
        if path is not None:
            return f"回译后端：{human}（{Path(path).name}）"
    return f"回译后端：{human}"


# --------------------------------------------------------------------------- #
# 七、翻译模型配置（复用 FISHERIES_TRANSLATOR_* 环境变量）
# --------------------------------------------------------------------------- #

def load_translator_client(env: dict[str, str] | None = None,
                           *, allow_scheduler_fallback: bool = True) -> Any | None:
    """按 ``FISHERIES_TRANSLATOR_*`` 构造翻译模型客户端；没配返回 ``None``。

    复用 :class:`fisheries.agent.llm.OpenAICompatibleClient`，
    所以翻译模型也可以是任何 OpenAI 兼容后端（DeepSeek / Qwen / Ollama）。

    配置示例::

        FISHERIES_TRANSLATOR_PROVIDER=deepseek
        FISHERIES_TRANSLATOR_API_KEY=sk-xxx

    ``allow_scheduler_fallback``（默认开）决定了 **没单配翻译模型时** 的行为：
    退回复用**调度角色**的模型配置。理由很实在——D12 这条路线的前提是
    "AQUA-7B 用英文思考、另一个中文模型负责回译"，而**推荐部署里调度角色本来
    就是一个中文强模型**（DeepSeek / Qwen）。不借用它的话，用户必须把同一个
    key 配两遍，忘了配就会静默退化成词表回译、白丢 AQUA-7B 的整句输出。

    代价是每次运行多一次小请求；不想要就设 ``FISHERIES_TRANSLATOR_MODE=glossary``
    显式关掉。调度角色本身没配 key（离线 Mock）时这里仍然返回 ``None``，
    所以没网的评委机器上不会因此多出一次失败的调用。
    """
    from fisheries.agent.llm import (          # 局部导入，避免循环依赖
        PROVIDER_PRESETS, LLMConfig, OpenAICompatibleClient, load_llm_config,
    )

    e = dict(os.environ if env is None else env)
    provider = e.get(TRANSLATOR_ENV_PREFIX + "PROVIDER", "").strip()
    base_url = e.get(TRANSLATOR_ENV_PREFIX + "BASE_URL", "").strip()
    if not provider and not base_url:
        if not allow_scheduler_fallback:
            return None
        sched: LLMConfig = load_llm_config("scheduler", env=e)
        if sched.is_mock:
            return None
        # ``is_mock`` 只看 provider 与 base_url：provider=deepseek 但没填 key 时
        # 它并**不**算 mock，可每次请求都会 401。这种"注定失败的调用"要在
        # 构造阶段就挡掉，否则每次运行都白等一个网络超时。
        # 本地后端（ollama / vLLM）不需要 key，所以按 base_url 放行。
        local = any(h in (sched.base_url or "")
                    for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))
        if not sched.api_key and not local:
            return None
        try:
            return OpenAICompatibleClient(sched)
        except Exception:                          # noqa: BLE001
            return None

    # 借用 load_llm_config 的解析逻辑：把 TRANSLATOR_* 映射成 LLM_* 再看。
    mapped: dict[str, str] = {}
    for key, val in e.items():
        if key.startswith(TRANSLATOR_ENV_PREFIX):
            mapped["FISHERIES_LLM_" + key[len(TRANSLATOR_ENV_PREFIX):]] = val
    cfg: LLMConfig = load_llm_config("advisor", env=mapped)
    if cfg.is_mock:
        # provider 配了但没解析出 base_url：用预置补齐，别静默退化成 mock。
        preset = PROVIDER_PRESETS.get(cfg.provider, {})
        cfg = cfg.model_copy(update={
            "base_url": cfg.base_url or preset.get("base_url", ""),
            "model": cfg.model or preset.get("model", ""),
            "api_key": cfg.api_key or e.get(preset.get("api_key_env", ""), "")
                       or preset.get("api_key", ""),
        })
    if cfg.is_mock:
        return None
    try:
        return OpenAICompatibleClient(cfg)
    except Exception:                              # noqa: BLE001 - base_url 空等
        return None


def dumps(obj: Any) -> str:
    """调试用：稳定的 JSON 文本。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


__all__ = [
    "ACTION_LIST_FIELDS",
    "ACTION_TEXT_FIELDS",
    "BackTranslation",
    "BypassTranslator",
    "DEFAULT_CHUNK_CHARS",
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_MAX_CHUNKS",
    "GLOSSARY_POST_ZH",
    "GLOSSARY_ZH",
    "GlossaryTranslator",
    "LANG_EN",
    "LANG_ZH",
    "LLMTranslator",
    "LOCAL_DIR_ENV",
    "LOCAL_MODEL_ENV",
    "LocalTranslator",
    "PROMPT_LANG_ENV",
    "SUPPORTED_LANGS",
    "TEXT_FIELDS",
    "TRANSLATOR_MODES",
    "TRANSLATOR_MODE_ENV",
    "TRANSLATOR_SYSTEM_PROMPT",
    "TranslationOutcome",
    "UNIT_NORMALIZE",
    "back_translate_report",
    "build_translator",
    "chunk_text",
    "clear_local_cache",
    "describe_translator",
    "load_translator_client",
    "local_model_available",
    "local_model_path",
    "local_model_root",
    "normalize_units",
    "numbers_preserved",
    "post_edit",
    "post_edit_terms",
    "resolve_prompt_lang",
    "resolve_translator_mode",
]
