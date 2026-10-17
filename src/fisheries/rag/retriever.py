"""检索：稠密向量 + BM25 稀疏的混合检索，带元数据过滤与 RRF 融合。

## 为什么要做混合检索（而不是只用向量）

向量检索擅长语义相似，但**在专有名词上很脆弱**：

    查询"亚硝酸盐超标怎么办"时，向量模型可能把
    "氨氮超标的处理"排在更前面——因为两者语义结构几乎一样，
    而对 BGE-small 这个 512 维小模型来说，
    "亚硝酸盐" 与 "氨氮" 的向量距离非常近。

BM25 恰恰相反：它只看词有没有出现，所以"亚硝酸盐"这个词
只有真正包含它的文档才会得高分。**两者互补**。

本项目的语料里有大量这类**必须精确命中的术语**：
溶氧 / 氨氮 / 亚硝酸盐 / 分子氨 / 烂鳃 / 水霉 / 泛塘 / 生石灰 / 硝化细菌。
这些词错一个，建议就完全变了。所以混合检索在本项目里不是"锦上添花"，
而是**正确性要求**。

## 融合方式：RRF（Reciprocal Rank Fusion）

    score(d) = Σ_over_lists  1 / (k + rank_list(d))      k = 60

选 RRF 而不是"加权归一化分数求和"的原因：
**BM25 分与余弦相似度的量纲完全不同、且随查询变化**，
归一化到同一个尺度要靠调参，而且换个查询就失效。
RRF 只用**排名**，不需要两者可比，是业界默认的稳健做法。
"""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .embedder import (
    Embeddings,
    build_embeddings,
    describe_embeddings,
    project_root,
)
from .schemas import RetrievalQuery, RetrievalResult, RetrievedChunk
from .store import (
    DEFAULT_COLLECTION,
    GENERIC_DISEASE,
    count_chunks,
    load_store,
    read_manifest,
    split_list,
    store_exists,
)

#: RRF 的平滑常数。60 是原论文推荐值，作用是压低"排名第 1"的相对优势，
#: 让"两个检索器都排第 3"的文档有机会胜过"只有一个检索器排第 1"的文档。
RRF_K = 60

#: 每个检索器先各取多少候选进入融合。取 4×k 是因为 RRF 的收益
#: 主要来自"被另一个检索器捞回来"的文档——候选池太小就捞不回来。
FETCH_MULTIPLIER = 4
MIN_FETCH_K = 20

#: 软加权的幅度。刻意做得很小（相对 RRF 的 1.0 量级），
#: 因为软加权只应"在分数接近时打破平局"，不应推翻检索器的判断。
RISK_LEVEL_BONUS = 0.04
INDICATOR_BONUS = 0.03
MAX_SOFT_BONUS = 0.10

#: 领域词表。告诉 jieba 这些是**一个词**，不要切开。
#: 例如"亚硝酸盐"如果被切成"亚硝/酸盐"，BM25 就退化成子串匹配噪音；
#: "增氧机"如果切成"增氧/机"，"增氧剂"也会被高分命中。
DOMAIN_WORDS: tuple[str, ...] = (
    "溶解氧",
    "溶氧",
    "增氧机",
    "增氧剂",
    "微孔增氧",
    "叶轮式增氧机",
    "水车式增氧机",
    "亚硝酸盐",
    "硝酸盐",
    "氨氮",
    "总氨氮",
    "分子氨",
    "非离子氨",
    "离子氨",
    "硫化氢",
    "泛塘",
    "浮头",
    "开机时机",
    "褐血病",
    "高铁血红蛋白",
    "烂鳃",
    "细菌性烂鳃病",
    "水霉",
    "水霉病",
    "肤霉病",
    "出血病",
    "细菌性出血病",
    "败血症",
    "赤皮病",
    "红体病",
    "寄生虫",
    "白尾病",
    "孔雀石绿",
    "硝化细菌",
    "生石灰",
    "沸石粉",
    "硫代硫酸钠",
    "过碳酸钠",
    "小苏打",
    "碳酸氢钠",
    "休药期",
    "投喂率",
    "投喂量",
    "四定",
    "残饵",
    "底改",
    "藻相",
    "倒藻",
    "应激",
    "阈值",
    "风险等级",
    "预警",
    "高危",
)

#: 检索停用词。只去掉对区分度毫无贡献的词，**不追求激进过滤**——
#: 过度去停用词会删掉"低温""超标"这类对水产场景有意义的词。
STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "或", "及", "等", "有", "会",
        "要", "就", "都", "而", "也", "还", "被", "把", "对", "从", "到",
        "请", "问", "怎么", "如何", "什么", "怎样", "为", "以", "并", "但",
        "这", "那", "其", "之", "上", "下", "中", "个", "该", "可以", "需要",
        "给出", "告诉", "一下", "吗", "呢", "啊", "吧", "以及", "进行",
        " ", "", "\n", "\t", "，", "。", "、", "；", "：", "！", "？", "（", "）",
        "(", ")", "[", "]", "《", "》", "“", "”", "'", '"', "-", "—", "|",
    }
)


# ------------------------------------------------------------------------- 分词


@functools.lru_cache(maxsize=1)
def _jieba() -> Any:
    """懒加载 jieba 并注册领域词。

    不用模块级 import：jieba 首次加载要构建词典（约 0.5–1 秒），
    而很多调用路径（如只做分块统计）根本不需要分词。
    """
    import jieba  # type: ignore

    jieba.setLogLevel(60)  # 关掉 "Building prefix dict..." 之类的日志
    for word in DOMAIN_WORDS:
        jieba.add_word(word, freq=10000)
    return jieba


def tokenize(text: str) -> list[str]:
    """中英混排分词 + 停用词过滤。

    英文与数字统一小写，避免 ``pH`` / ``PH`` / ``ph`` 被当成三个词。
    """
    if not text.strip():
        return []
    raw = _jieba().lcut(text)
    out: list[str] = []
    for tok in raw:
        t = tok.strip().lower()
        if not t or t in STOPWORDS or len(t) > 24:
            continue
        out.append(t)
    return out


# ------------------------------------------------------------------------ BM25


@dataclass(slots=True)
class SparseHit:
    index: int
    score: float


class SparseIndex:
    """基于 rank_bm25 的稀疏索引，支持**与稠密路径一致的元数据过滤**。

    过滤在打分前用谓词完成（本项目语料只有百量级块，代价可忽略），
    这样就不需要维护两套过滤逻辑的语义一致性——
    ``predicate`` 是唯一的过滤真相，稠密路径的 Chroma ``where``
    由同一份:func:`build_filter` 生成，并有单测保证两者等价。
    """

    def __init__(self, texts: Sequence[str], metadatas: Sequence[dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self.texts = list(texts)
        self.metadatas = list(metadatas)
        self.tokenized = [tokenize(t) for t in self.texts]
        # BM25Okapi 在空语料上会抛异常；用空列表时退化成一个恒返回 0 的假索引。
        self._bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def __len__(self) -> int:
        return len(self.texts)

    def search(
        self,
        query: str,
        k: int,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[SparseHit]:
        """返回按 BM25 分降序的前 ``k`` 条（已应用谓词过滤）。"""
        if self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        hits: list[SparseHit] = []
        for i, s in enumerate(scores):
            if s <= 0:
                # 分数 <= 0 说明没有任何查询词命中（BM25 的负分来自 IDF）。
                # 这类文档留在候选里只会稀释 RRF，直接丢弃。
                continue
            if predicate is not None and not predicate(self.metadatas[i]):
                continue
            hits.append(SparseHit(index=i, score=float(s)))

        hits.sort(key=lambda h: (-h.score, h.index))
        return hits[:k]


# ------------------------------------------------------------------------ 过滤


def build_predicate(q: RetrievalQuery) -> Callable[[dict[str, Any]], bool]:
    """构造元数据谓词。**这是过滤语义的唯一真相。**"""

    def predicate(meta: dict[str, Any]) -> bool:
        if q.category is not None and meta.get("category") != q.category:
            return False
        if q.disease is not None:
            allowed = {q.disease}
            if q.allow_disease_null:
                allowed.add(GENERIC_DISEASE)
            if meta.get("disease") not in allowed:
                return False
        # 元章节（"与本项目 Agent 的联动"）默认不参与检索，
        # 它们措辞里全是领域关键词，会挤掉真正的处置方案。
        return q.include_meta or meta.get("section_role") != "meta"

    return predicate


def build_filter(q: RetrievalQuery) -> dict[str, Any] | None:
    """把 :class:`RetrievalQuery` 翻译成 Chroma 的 ``where`` 子句。

    注意 ``risk_level`` / ``indicators`` **不在这里出现**——它们是软加权，
    理由见 :class:`RetrievalQuery` 的文档。
    """
    clauses: list[dict[str, Any]] = []
    if q.category is not None:
        clauses.append({"category": q.category})
    if q.disease is not None:
        allowed = [q.disease]
        if q.allow_disease_null:
            allowed.append(GENERIC_DISEASE)
        clauses.append({"disease": {"$in": allowed}})
    if not q.include_meta:
        clauses.append({"section_role": "domain"})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def filter_description(q: RetrievalQuery) -> dict[str, Any]:
    """给人看的过滤条件描述，写进 :class:`RetrievalResult`。"""
    desc: dict[str, Any] = {}
    if q.category is not None:
        desc["category"] = q.category
    if q.disease is not None:
        allowed = [q.disease] + ([GENERIC_DISEASE] if q.allow_disease_null else [])
        desc["disease"] = allowed
    if not q.include_meta:
        desc["section_role"] = "domain（已排除元章节）"
    if q.risk_level:
        desc["risk_level(软加权)"] = q.risk_level
    if q.indicators:
        desc["indicators(软加权)"] = list(q.indicators)
    return desc


# ------------------------------------------------------------------------ 融合


@dataclass(slots=True)
class FusedHit:
    """融合后的候选。"""

    index: int
    rrf: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    soft_bonus: float = 0.0

    @property
    def sources(self) -> list[str]:
        out = []
        if self.dense_rank is not None:
            out.append("dense")
        if self.sparse_rank is not None:
            out.append("sparse")
        return out


def rrf_fuse(
    dense: Sequence[tuple[int, float]],
    sparse: Sequence[tuple[int, float]],
) -> list[FusedHit]:
    """RRF 融合两路召回结果。

    入参是 ``[(全局索引, 分数), ...]``，**顺序即排名**。
    """
    acc: dict[int, FusedHit] = {}

    for rank, (idx, score) in enumerate(dense, start=1):
        hit = acc.setdefault(idx, FusedHit(index=idx))
        hit.rrf += 1.0 / (RRF_K + rank)
        hit.dense_score = score
        hit.dense_rank = rank

    for rank, (idx, score) in enumerate(sparse, start=1):
        hit = acc.setdefault(idx, FusedHit(index=idx))
        hit.rrf += 1.0 / (RRF_K + rank)
        hit.sparse_score = score
        hit.sparse_rank = rank

    out = list(acc.values())
    # 排序键加上 index 保证**完全确定性**：分数相同时按索引升序，
    # 不依赖 dict 插入顺序（那是实现细节，会随检索器行为变化）。
    out.sort(key=lambda h: (-h.rrf, -len(h.sources), h.index))
    return out


def _soft_bonus(meta: dict[str, Any], q: RetrievalQuery) -> float:
    """风险等级与指标的软加权。"""
    bonus = 0.0
    if q.risk_level:
        levels = split_list(meta.get("risk_levels"))
        if q.risk_level in levels:
            bonus += RISK_LEVEL_BONUS
    if q.indicators:
        inds = set(split_list(meta.get("indicators")))
        overlap = inds & set(q.indicators)
        if overlap:
            bonus += min(INDICATOR_BONUS * len(overlap), MAX_SOFT_BONUS)
    return min(bonus, MAX_SOFT_BONUS)


def _normalize_rrf(rrf: float) -> float:
    """把 RRF 分归一化到 0–1。

    分母取"两个检索器都排第 1"的理论满分 ``2/(RRF_K+1)``，
    因此 1.0 表示"两路都排第一"。

    **注意这套分数很扁**：前几名的归一化值都在 0.95 以上。
    它只适合排序，不适合当"相关度百分比"展示——
    要展示相关度请用 ``dense_score``（余弦相似度）。
    """
    best = 2.0 / (RRF_K + 1)
    return min(1.0, rrf / best)


def _diversify(
    candidates: Sequence[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """标题去重：同一标题下只保留得分最高的一块（**并非无条件**）。

    语料里较长的章节（如"2. 投喂率：随水温变化"）会被按长度切成 2 块。
    于是检索 top-2 全落在同一节的上下半截上，
    8 个注入槽位里就有 2 个在讲同一件事，白占上下文，
    而"投料管理"这个用途分组却被整体挤掉。

    实现是**先按标题去重、不够再回填**：
    如果去重后凑不满 ``limit`` 条（例如某个类别只有一节），
    就把被去掉的重复块补回来——多一段同主题内容，
    好过让整个用途分组空掉。
    """
    kept: list[RetrievedChunk] = []
    seen: set[str] = set()
    overflow: list[RetrievedChunk] = []

    for c in candidates:
        key = c.heading_text or c.chunk_id
        if key in seen:
            overflow.append(c)
            continue
        seen.add(key)
        kept.append(c)
        if len(kept) >= limit:
            return kept

    for c in overflow:
        if len(kept) >= limit:
            break
        kept.append(c)
    return kept


# ------------------------------------------------------------------ 知识库门面


@dataclass(slots=True)
class AgentKnowledgeState:
    """从 Agent 的融合结果里抽出的、与检索相关的字段。

    刻意做成一个**显式的小结构**而不是直接传 ``dict``：
    检索计划的输入直接决定召回什么，把它做成有类型的字段，
    单测就能钉住"高危时必须检索应急预案"这类规则。
    """

    risk_level: str = "正常"
    limiting_factor: str = ""
    limiting_factor_label: str = ""
    redline_hits: list[str] = field(default_factory=list)
    severe_indicators: list[str] = field(default_factory=list)
    disease_type: str = ""
    is_healthy: bool = True
    is_low_confidence: bool = False
    do_eta_hours: float | None = None
    water_temp: float | None = None
    species: str = "鲤科"


def state_from_agent(water: dict[str, Any] | None, vision: dict[str, Any] | None, fusion: dict[str, Any] | None) -> AgentKnowledgeState:
    """从三个工具/融合层的返回值里抽取检索状态。全部按"缺失即默认"处理。"""
    st = AgentKnowledgeState()
    if fusion:
        st.risk_level = str(fusion.get("risk_level") or st.risk_level)
    if water and water.get("status") == "ok":
        st.limiting_factor = str(water.get("limiting_factor") or "")
        st.limiting_factor_label = str(water.get("limiting_factor_label") or "")
        st.redline_hits = [str(x) for x in (water.get("redline_hits") or [])]
        sev = water.get("severity_map") or {}
        st.severe_indicators = sorted(
            k for k, v in sev.items() if isinstance(v, (int, float)) and float(v) >= 0.5
        )
        eta = water.get("projected_do_alert_eta_hours")
        st.do_eta_hours = float(eta) if isinstance(eta, (int, float)) else None
        trends = water.get("trends") or {}
        temp = (trends.get("water_temp") or {}).get("current")
        st.water_temp = float(temp) if isinstance(temp, (int, float)) else None
    if vision and vision.get("status") == "ok":
        st.disease_type = str(vision.get("disease_type") or "")
        st.is_healthy = bool(vision.get("is_healthy", True))
        st.is_low_confidence = bool(vision.get("is_low_confidence", False))
    return st


#: 红线/严重指标 → 应急预案里的**章节词**。
#: 为什么需要它：预案文档把缺氧单独写成「1. 泛塘（严重缺氧）」，
#: 标题里根本没有"溶氧"二字，只按指标名检索会命中氨氮/亚硝酸盐两节，
#: 于是最该看的"泛塘立即动作"反而排不进来（实测 top-2 就是氨氮与亚硝酸盐）。
#: 这里把指标翻译成文档实际使用的措辞，让 BM25 能精确落到对应小节。
_EMERGENCY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dissolved_oxygen": ("泛塘", "缺氧", "浮头", "开机时机"),
    "ammonia_n": ("氨氮", "中毒"),
    "nitrite_n": ("亚硝酸盐", "褐血病"),
    "ph": ("pH", "酸碱度"),
    "water_temp": ("水温",),
}

#: 红线编码前缀 → 指标名。`DO_REDLINE` / `DO_REDLINE_PROJECTED` / `DO_CRASH_TREND`
#: 都属于溶氧，所以按前缀归并，而不是逐个编码写死（新增红线时不用改这里）。
_REDLINE_PREFIX_TO_INDICATOR: tuple[tuple[str, str], ...] = (
    ("DO_", "dissolved_oxygen"),
    ("NH3", "ammonia_n"),
    ("NO2", "nitrite_n"),
    ("PH_", "ph"),
)


def _emergency_keywords(st: AgentKnowledgeState) -> list[str]:
    """按命中红线与严重指标，挑出应急预案里真正对应的章节词。

    只收集本次**真实命中**的指标，不做"把所有可能都塞进去"的兜底——
    那样会把四个章节一起拉上来，等于没有定向检索。
    """
    words: list[str] = []
    for _indicator, group in _emergency_groups(st):
        for w in group:
            if w not in words:
                words.append(w)
    return words


def _emergency_groups(st: AgentKnowledgeState) -> list[tuple[str, list[str]]]:
    """把"命中红线 / 严重指标"归并成**按指标分组**的章节词。

    为什么必须分组而不是拼成一条查询：实测把溶氧与氨氮的章节词
    混在同一条 query 里时，BM25 会把「急性氨氮中毒 > 2.2 立即动作」
    顶到第一，而「1. 泛塘（严重缺氧）> 立即动作」掉出前二——
    因为泛塘小节正文更长，被长度归一化压了分。
    而这次的主控风险因子恰恰是溶氧。分组后每个指标各查一次，
    每个指标都能拿到自己的头名，主控因子排在最前。
    """
    groups: list[tuple[str, list[str]]] = []

    def add(indicator: str) -> None:
        words = _EMERGENCY_KEYWORDS.get(indicator)
        if words and all(indicator != name for name, _ in groups):
            groups.append((indicator, list(words)))

    for code in st.redline_hits:
        upper = str(code).upper()
        for prefix, indicator in _REDLINE_PREFIX_TO_INDICATOR:
            if upper.startswith(prefix):
                add(indicator)
                break
    # 主控风险因子优先于其它严重指标：它才是本次真正要应对的东西。
    if st.limiting_factor:
        add(st.limiting_factor)
    for indicator in st.severe_indicators:
        add(indicator)

    # 兜底：等级是高危但一个具体指标都没带（比如融合分≥70 但没命中红线，
    # 或调用方只传了 risk_level）。此时仍需召回预案——
    # 只是没办法定向到某一节，就退回不带指标词的通用应急查询。
    if not groups:
        groups.append(("", []))

    return groups[:3]   # 最多三条：再多会把预算吃光，反而挤掉水质/病害/投料


def plan_queries(st: AgentKnowledgeState) -> list[RetrievalQuery]:
    """把 Agent 状态翻译成一组检索请求。

    这是本项目 RAG 与"直接拿用户问题去搜"最关键的区别：
    **检索不是由用户措辞驱动的，而是由 Agent 已经确定的事实驱动的。**
    工具已经算出"溶氧是主控风险因子、命中了 DO_REDLINE"，
    那就没必要让向量检索去猜该找什么——直接按这个事实定向检索。

    固定产出 1–4 条请求，顺序即优先级：

        1. 应急处置（命中红线，或等级为高危）
        2. 水质调控（总有：针对主控风险因子）
        3. 病害处置（图像检出具体病害且非健康）
        4. 投料管理（未检测到病害，或只有低置信度提示）
    """
    plan: list[RetrievalQuery] = []

    # 1) 应急处置 —— 只在真正紧急时检索。
    #    理由：预案文档里全是"立即""紧急"的祈使句，
    #    在正常状态下混进 prompt 会让模型把日常巡检也写成紧急告警。
    #    每个命中的指标各查一次（见 _emergency_groups 的说明），
    #    主控风险因子排在最前，保证它的"立即动作"一定被召回。
    if st.redline_hits or st.risk_level == "高危":
        for indicator, words in _emergency_groups(st):
            bits = ["应急", "处置", *words, "立即措施", "增氧", "停料", "换水"]
            plan.append(
                RetrievalQuery(
                    query=" ".join(bits),
                    k=3,
                    purpose="应急处置",
                    category="应急",
                    risk_level=st.risk_level,
                    indicators=[indicator, *st.severe_indicators[:3]],
                )
            )

    # 2) 水质调控 —— 永远检索，因为任何等级下都需要"保持水质"的依据。
    #    query 里带上主控风险因子，让 BM25 精确命中对应小节标题。
    wq_bits = ["水质调控"]
    if st.limiting_factor_label:
        wq_bits.append(st.limiting_factor_label)
    if st.do_eta_hours is not None:
        wq_bits.append("增氧机 开机时机")
    plan.append(
        RetrievalQuery(
            query=" ".join(wq_bits) + " 处置 措施",
            k=3,
            purpose="水质调控",
            category="水质",
            risk_level=st.risk_level,
            indicators=st.severe_indicators[:4],
        )
    )

    # 3) 病害处置 —— 只在检出具体病害时检索，且**硬过滤病害标签**。
    #    健康或"其他"（底层为寄生虫/病毒）时不检索病害处置方案：
    #    我们没有那两类的处置文档，检索只会返回不相关的病种文档。
    if st.disease_type and not st.is_healthy and st.disease_type in {"烂鳃", "出血", "水霉"}:
        plan.append(
            RetrievalQuery(
                # 检索词刻意**以处置动作为主，不放"症状""鉴别"**：
                # 加了这两个词之后，BM25 会把"2.1 外观观察"顶到第一，
                # 而 Agent 这一刻需要的是"该做什么"，不是"怎么认病"
                # （认病已经由图像模型做完了）。
                query=f"{st.disease_type} 处置方案 立即措施 第一步 用药 步骤 停药期",
                k=3,
                purpose="病害处置",
                category="病害",
                disease=st.disease_type,  # type: ignore[arg-type]
                indicators=st.severe_indicators[:4],
                # 只召回该病害专属文档：跨病种串用药方案是本项目最不能接受的失败。
                allow_disease_null=False,
            )
        )

    # 4) 投料管理 —— 关键：**有病害时也要检索投料**。
    #    因为病害处置的第一步几乎都是"停料"，而"停几天、怎么恢复投喂"
    #    只有投料规范里写了。漏掉这块会让建议缺少落地细节。
    feed_bits = ["投喂"]
    if st.water_temp is not None:
        feed_bits.append(f"水温 {st.water_temp:.0f} 度 投喂率")
    if st.disease_type and not st.is_healthy:
        feed_bits.append("发病期 停料 减料")
    plan.append(
        RetrievalQuery(
            query=" ".join(feed_bits) + " 投喂量 调整",
            k=3,
            purpose="投料管理",
            category="投料",
            indicators=st.severe_indicators[:4],
        )
    )

    return plan


@dataclass(slots=True)
class KnowledgeContext:
    """一次 Agent 调用的完整检索产物。"""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    by_purpose: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    results: list[RetrievalResult] = field(default_factory=list)
    backend: str = ""
    degraded: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def all_unverified(self) -> bool:
        return bool(self.chunks) and all(not c.verified for c in self.chunks)

    def render(self, *, max_chars_per_chunk: int = 900) -> str:
        """渲染成注入 prompt 的知识块。

        按用途分组，每组带用途标题——让大模型知道
        "这段是讲怎么救急的，那段是讲怎么投料的"，
        而不是收到一坨无结构的文本。
        """
        if not self.chunks:
            return "（本次未检索到相关知识条目）"

        lines: list[str] = []
        for purpose, items in self.by_purpose.items():
            if not items:
                continue
            lines.append(f"### 【{purpose}】")
            for i, c in enumerate(items, start=1):
                body = c.text.strip()
                if len(body) > max_chars_per_chunk:
                    body = body[:max_chars_per_chunk].rstrip() + " …（已截断）"
                flag = "已核对原文" if c.verified else "⚠ 未与标准原文核对"
                # 展示**余弦相似度**而不是 RRF 分：RRF 前几名都挤在 0.95 以上，
                # 显示出来全是"相关度 1.00"，看不出差异，还会让人误以为是相似度。
                if c.dense_score is not None:
                    rel = f"向量相似度 {c.dense_score:.2f}"
                else:
                    rel = f"仅词面命中（BM25 第 {c.sparse_rank} 名）"
                head = f"{i}. [{c.category}] {c.title} > {c.heading_text}（{flag}；{rel}）"
                lines.append(head)
                lines.append(body)
                lines.append("")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "degraded": self.degraded,
            "n_chunks": self.n_chunks,
            "warnings": list(self.warnings),
            "purposes": {k: [c.chunk_id for c in v] for k, v in self.by_purpose.items()},
            "chunks": [c.model_dump() for c in self.chunks],
        }


class KnowledgeBase:
    """RAG 门面：加载 → 检索 → 渲染。

    典型用法：:

        kb = KnowledgeBase.open()                  # 加载已构建的向量库
        ctx = kb.retrieve_for_agent(water=w, vision=v, fusion=f)
        prompt_block = ctx.render()

    也支持直接构建（比赛演示、CI）::

        kb, stats = KnowledgeBase.build()
    """

    def __init__(
        self,
        store: object,
        embeddings: Embeddings,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self.ids = list(ids)
        self.texts = list(texts)
        self.metadatas = list(metadatas)
        self.sparse = SparseIndex(self.texts, self.metadatas)
        # ``count_chunks`` 读不到时返回 ``None``（见 rag/store.py），别让它
        # 渲染成 "chroma(None 块)"。
        n_chroma = count_chunks(store)
        self.backend = (
            f"chroma({n_chroma if n_chroma is not None else '?'} 块) + "
            f"bm25({len(self.sparse)} 块) / "
            f"{describe_embeddings(embeddings)}"
        )
        self._index_of: dict[str, int] = {cid: i for i, cid in enumerate(self.ids)}

    # ------------------------------------------------------------------ 构造

    @classmethod
    def _from_store(cls, store: object, embeddings: Embeddings) -> KnowledgeBase:
        """把 Chroma 里的全部块读进内存，供 BM25 与元数据谓词使用。"""
        data = store.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
        ids = [str(i) for i in data.get("ids", [])]
        texts = [str(t) for t in data.get("documents", [])]
        metadatas = [dict(m or {}) for m in data.get("metadatas", [])]

        if not ids:
            raise RuntimeError(
                "向量库是空的（0 个块）。请运行 scripts/build_knowledge_base.py 构建，"
                "或检查 collection 名是否正确。"
            )
        # Chroma 的 get 不保证顺序，按 chunk_id 排序让 BM25 的索引位置确定。
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        return cls(
            store,
            embeddings,
            ids=[ids[i] for i in order],
            texts=[texts[i] for i in order],
            metadatas=[metadatas[i] for i in order],
        )

    @classmethod
    def open(
        cls,
        *,
        persist_dir: str | pathlib.Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embeddings: Embeddings | None = None,
        embedder_backend: str | None = None,
        strict: bool = True,
    ) -> KnowledgeBase:
        """加载已构建的向量库。未构建时抛 :class:`StoreMismatchError`。"""
        emb = embeddings or build_embeddings(embedder_backend)
        store = load_store(
            emb, persist_dir=persist_dir, collection_name=collection_name, strict=strict
        )
        return cls._from_store(store, emb)

    @classmethod
    def build(
        cls,
        *,
        corpus_dir: str | pathlib.Path | None = None,
        persist_dir: str | pathlib.Path | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        embeddings: Embeddings | None = None,
        embedder_backend: str | None = None,
        max_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> tuple[KnowledgeBase, Any]:
        """从语料目录构建向量库并返回 ``(KnowledgeBase, BuildStats)``。"""
        from .loader import load_corpus
        from .splitter import (
            DEFAULT_MAX_CHARS,
            DEFAULT_OVERLAP_CHARS,
            build_embedding_text,
            chunk_corpus,
        )
        from .store import build_store

        emb = embeddings or build_embeddings(embedder_backend)
        corpus = pathlib.Path(corpus_dir) if corpus_dir else project_root() / "knowledge" / "corpus"

        docs = load_corpus(corpus)
        pairs = chunk_corpus(
            docs,
            max_chars=max_chars or DEFAULT_MAX_CHARS,
            overlap_chars=overlap_chars or DEFAULT_OVERLAP_CHARS,
        )
        embed_texts = [build_embedding_text(meta, text) for meta, text in pairs]

        store, stats = build_store(
            pairs,
            emb,
            persist_dir=persist_dir,
            collection_name=collection_name,
            reset=True,
            embedding_texts=embed_texts,
        )
        return cls._from_store(store, emb), stats

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query: str | RetrievalQuery,
        *,
        k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """执行一次混合检索。"""
        q = RetrievalQuery(query=query, **kwargs) if isinstance(query, str) else query
        if k is not None:
            q = q.model_copy(update={"k": k})

        predicate = build_predicate(q)
        candidates = [i for i, m in enumerate(self.metadatas) if predicate(m)]
        if not candidates:
            return RetrievalResult(
                query=q.query,
                purpose=q.purpose,
                filters=filter_description(q),
                chunks=[],
                n_candidates=0,
                backend=self.backend,
                warnings=["过滤条件过严：没有任何块通过元数据过滤。"],
            )

        fetch_k = max(MIN_FETCH_K, q.k * FETCH_MULTIPLIER)
        fetch_k = min(fetch_k, len(candidates))

        dense = self._dense_search(q, fetch_k)
        sparse = self._sparse_search(q, fetch_k, predicate)

        fused = rrf_fuse(dense, sparse)
        for hit in fused:
            hit.soft_bonus = _soft_bonus(self.metadatas[hit.index], q)
        fused.sort(key=lambda h: (-(h.rrf + h.soft_bonus), h.index))

        # 从更大的候选池里做标题去重后再取 k 条。
        # 池子取 3k：去重要"有得选"才行，只从 top-k 里去重等于没去重。
        pool = fused[: max(q.k * 3, q.k)]
        chunks = _diversify([self._to_chunk(h) for h in pool], q.k)

        warnings: list[str] = []
        if len(fused) < q.k:
            warnings.append(
                f"仅召回 {len(fused)} 条（请求 {q.k} 条），语料可能未覆盖该主题。"
            )
        if not dense and not sparse:
            warnings.append("稠密与稀疏检索均无命中。")

        return RetrievalResult(
            query=q.query,
            purpose=q.purpose,
            filters=filter_description(q),
            chunks=chunks,
            n_candidates=len(candidates),
            backend=self.backend,
            warnings=warnings,
        )

    def _dense_search(
        self, q: RetrievalQuery, fetch_k: int
    ) -> list[tuple[int, float]]:
        """稠密检索。返回 ``[(全局索引, 余弦相似度), ...]``，顺序即排名。

        用 ``similarity_search_with_score``（返回**距离**）而不是
        ``..._with_relevance_scores``（返回"相关度"）：
        后者依赖 LangChain 猜测 relevance 转换函数，不同版本行为不同，
        而且对 cosine 空间会给出误导性的 0–1 分数。
        这里自己把距离转成相似度：``sim = 1 - distance``（cosine 空间下成立）。
        """
        where = build_filter(q)
        try:
            pairs = self._store.similarity_search_with_score(  # type: ignore[attr-defined]
                q.query, k=fetch_k, filter=where
            )
        except Exception:  # noqa: BLE001 - filter 参数在各版本签名不一致时退化为不过滤
            # 但必须让调用方知道：静默返回未过滤结果会导致"跨病种串答案"。，但必须让调用方知道：
            # 静默返回未过滤结果会导致"跨病种串答案"。
            pairs = self._store.similarity_search_with_score(q.query, k=fetch_k)  # type: ignore[attr-defined]
            self._last_dense_filter_failed = True

        out: list[tuple[int, float]] = []
        for doc, distance in pairs:
            idx = self._index_of.get(str(doc.metadata.get("chunk_id", "")))
            if idx is None:
                continue
            out.append((idx, 1.0 - float(distance)))
        return out

    def _sparse_search(
        self,
        q: RetrievalQuery,
        fetch_k: int,
        predicate: Callable[[dict[str, Any]], bool],
    ) -> list[tuple[int, float]]:
        """稀疏检索（BM25）。"""
        hits = self.sparse.search(q.query, fetch_k, predicate)
        return [(h.index, h.score) for h in hits]

    def _to_chunk(self, hit: FusedHit) -> RetrievedChunk:
        meta = self.metadatas[hit.index]
        return RetrievedChunk(
            chunk_id=str(meta.get("chunk_id", self.ids[hit.index])),
            text=self.texts[hit.index],
            heading_text=str(meta.get("heading_text", "")),
            title=str(meta.get("title", "")),
            category=str(meta.get("category", "水质")),  # type: ignore[arg-type]
            disease=None
            if str(meta.get("disease", GENERIC_DISEASE)) == GENERIC_DISEASE
            else str(meta.get("disease")),  # type: ignore[arg-type]
            verified=bool(meta.get("verified", False)),
            source=str(meta.get("source", "")),
            standard_codes=split_list(meta.get("standard_codes")),
            verified_on=str(meta.get("verified_on", "")),
            citation_keys=split_list(meta.get("citation_keys")),
            score=round(min(1.0, _normalize_rrf(hit.rrf)), 4),
            dense_score=None if hit.dense_score is None else round(hit.dense_score, 4),
            sparse_score=None if hit.sparse_score is None else round(hit.sparse_score, 4),
            soft_boost=round(hit.soft_bonus, 4),
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
            matched_by=hit.sources,
        )

    # ------------------------------------------------------- 面向 Agent 的入口

    def retrieve_for_state(
        self, st: AgentKnowledgeState, *, max_chunks: int = 8
    ) -> KnowledgeContext:
        """按检索计划执行多条检索，合并去重后返回 :class:`KnowledgeContext`。

        去重规则：同一块被多条查询命中时**保留最高分**，
        并归属到**优先级最高的那个用途**（计划顺序即优先级）。
        这样"应急处置"里的块不会因为也被"水质调控"命中就被降级展示。
        """
        ctx = KnowledgeContext(backend=self.backend)

        seen: dict[str, RetrievedChunk] = {}
        order: list[str] = []

        for q in plan_queries(st):
            res = self.search(q)
            ctx.results.append(res)
            ctx.warnings.extend(res.warnings)
            bucket = ctx.by_purpose.setdefault(q.purpose, [])
            for chunk in res.chunks:
                existing = seen.get(chunk.chunk_id)
                if existing is None:
                    seen[chunk.chunk_id] = chunk
                    order.append(chunk.chunk_id)
                    bucket.append(chunk)
                elif chunk.score > existing.score:
                    # 同一块被更高优先级的查询以更高分召回：更新分数，
                    # 但**不改变它归属的用途分组**（先到的高优先级用途优先展示）。
                    seen[chunk.chunk_id] = chunk
                    for items in ctx.by_purpose.values():
                        for j, it in enumerate(items):
                            if it.chunk_id == chunk.chunk_id:
                                items[j] = chunk

        # 轮转分配预算，而不是简单地从队尾截断。
        #
        # 直接 `[seen[cid] for cid in order][:max_chunks]` 会有个隐蔽后果：
        # 计划是 应急3 + 水质3 + 病害3 + 投料2 = 11 条，max_chunks=8，
        # 于是排最后的「投料管理」被**整块丢掉**。
        # 而病害处置的第一步几乎都是"停料"——最需要投料规范的时候它不见了。
        #
        # 轮转（每个用途轮流拿 1 条）保证低优先级用途也至少分到 2 条。
        buckets = [(p, list(items)) for p, items in ctx.by_purpose.items()]
        picked: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        depth = 0
        while len(picked) < max_chunks:
            progressed = False
            for _purpose, items in buckets:
                if depth >= len(items):
                    continue
                progressed = True
                if len(picked) >= max_chunks:
                    break
                c = items[depth]
                if c.chunk_id not in seen_ids:
                    seen_ids.add(c.chunk_id)
                    picked.append(c)
            if not progressed:
                break
            depth += 1

        ctx.chunks = picked
        kept = {c.chunk_id for c in picked}
        for purpose, items in ctx.by_purpose.items():
            ctx.by_purpose[purpose] = [c for c in items if c.chunk_id in kept][:4]

        if ctx.all_unverified and ctx.chunks:
            ctx.warnings.append(
                "检索到的知识条目全部标注为『未与标准原文核对』，"
                "引用时必须提示需人工复核。"
            )
        return ctx

    def retrieve_for_agent(
        self,
        *,
        water: dict[str, Any] | None = None,
        vision: dict[str, Any] | None = None,
        fusion: dict[str, Any] | None = None,
        max_chunks: int = 8,
    ) -> KnowledgeContext:
        """从 Agent 的工具返回值直接检索。"""
        st = state_from_agent(water, vision, fusion)
        return self.retrieve_for_state(st, max_chunks=max_chunks)

    # ------------------------------------------------------------------ 其他

    def describe(self) -> dict[str, Any]:
        """知识库概览，供 README/报告/健康检查使用。"""
        manifest = read_manifest()
        cats: dict[str, int] = {}
        diseases: dict[str, int] = {}
        for m in self.metadatas:
            cats[str(m.get("category"))] = cats.get(str(m.get("category")), 0) + 1
            d = str(m.get("disease"))
            diseases[d] = diseases.get(d, 0) + 1
        return {
            "backend": self.backend,
            "n_chunks": len(self.ids),
            "by_category": dict(sorted(cats.items())),
            "by_disease": dict(sorted(diseases.items())),
            "manifest": manifest,
        }

    def __len__(self) -> int:
        return len(self.ids)


def open_knowledge_base(
    *,
    auto_build: bool = True,
    embedder_backend: str | None = None,
    persist_dir: str | pathlib.Path | None = None,
) -> KnowledgeBase:
    """给上层调用用的便捷入口：没有库就构建，有库就加载。

    ``auto_build`` 默认 True 是为了**演示体验**：
    比赛现场一条命令就能跑通。但生产环境应设 False——
    自动构建会在服务启动时触发 90 MB 模型的加载与全量重算。
    """
    from .store import StoreMismatchError

    if store_exists(persist_dir):
        return KnowledgeBase.open(
            persist_dir=persist_dir, embedder_backend=embedder_backend, strict=True
        )
    if not auto_build:
        raise StoreMismatchError(
            "向量库不存在，且 auto_build=False。请先运行 scripts/build_knowledge_base.py。"
        )
    kb, _ = KnowledgeBase.build(persist_dir=persist_dir, embedder_backend=embedder_backend)
    return kb
