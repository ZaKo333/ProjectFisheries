"""RAG 层的数据契约。

本模块只定义**结构**，不含任何 IO 与算法。所有跨模块传递的数据都先在这里
定型，好处是：加载、分块、向量化、检索四个环节可以各自单测，
且检索结果天然是 JSON 可序列化的——它要直接进大模型的 prompt。

设计取舍说明：
    * 知识块的元数据（``ChunkMeta``）刻意做得比一般 RAG 项目"厚"。
      一般 RAG 只存 ``source`` 和 ``page``，但水产场景下**检索到错误的病种
      比检索不到更危险**——用户问烂鳃，检索回水霉的用药方案，
      模型会把它当成本次病害的处置建议。因此必须把
      ``disease`` / ``category`` / ``indicators`` 做成可过滤的结构化字段，
      而不只是塞在正文里等向量去"理解"。
    * ``verified`` 字段记录该文档是否已与标准原文核对。
      检索结果会把它带进 prompt，让模型知道哪些数值可以直说、
      哪些必须加"待与标准原文核对"。
"""

from __future__ import annotations

from datetime import date as _date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 语料类别。与 front-matter 的 ``category`` 字段取值一致。
Category = Literal["水质", "病害", "投料", "应急", "标准"]

#: 本项目支持的病害标签（与图像识别工具的输出口径一致）。
DiseaseLabel = Literal["烂鳃", "出血", "水霉", "其他"]


class DocMeta(BaseModel):
    """单个语料文档的 front-matter。

    对应 Markdown 文件头部的 YAML 块。``verified=False`` 表示该文档的
    数值尚未与标准原文逐条核对——本项目**所有种子语料都是 False**，
    这一点会在检索结果里如实传递给大模型。
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="文档唯一标识，用于溯源与去重")
    title: str = Field(description="文档标题，会作为检索结果的一级面包屑")
    category: Category = Field(description="语料类别，供条件过滤")
    disease: DiseaseLabel | None = Field(
        default=None, description="该文档针对的病害；通用文档为 None"
    )
    indicators: list[str] = Field(
        default_factory=list, description="涉及的指标键，如 dissolved_oxygen"
    )
    risk_levels: list[str] = Field(
        default_factory=list, description="适用的风险等级：正常 / 预警 / 高危"
    )
    source: str = Field(default="自编", description="语料来源说明")
    verified: bool = Field(
        default=False, description="是否已与标准原文核对；False 时提示模型加免责说明"
    )
    standard_codes: list[str] = Field(
        default_factory=list,
        description=(
            "本文件引用的标准编号，如 'GB 11607-1989'。"
            "做成结构化字段而不是只写在正文里，是为了能按标准号过滤与审计"
        ),
    )
    access_source: str = Field(
        default="", description="核验时实际访问的来源（数据库名 + URL）"
    )
    verified_by: str = Field(
        default="", description="核验人；verified=True 时应非空，否则核验无从追溯"
    )
    verified_on: str = Field(default="", description="核验日期，格式 YYYY-MM-DD")
    source_path: str = Field(default="", description="源文件路径（相对项目根）")

    @field_validator("verified_on", mode="before")
    @classmethod
    def _coerce_verified_on(cls, value: Any) -> Any:
        """把 YAML 日期统一成 ISO 字符串。

        YAML 会把**不加引号**的 ``2026-10-15`` 解析成 ``datetime.date``，
        而字段是 ``str``。与其要求每个语料作者记得加引号（迟早会漏），
        不如在入口处统一——漏引号的代价不该是一条校验错误。
        """
        if isinstance(value, _date):
            return value.isoformat()
        return value


class ChunkMeta(BaseModel):
    """单个知识块（chunk）的元数据。

    ``heading_path`` 是检索质量的关键：chunk 被切出来后往往只有一两段正文，
    脱离标题就看不出在讲什么。检索时会把 `标题链 + 正文` 一起向量化，
    但**返回给大模型时仍用正文**，避免面包屑重复占满上下文。
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="稳定 ID：同源文件同标题路径恒等，便于增量更新")
    doc_id: str
    title: str
    category: Category
    disease: DiseaseLabel | None = None
    indicators: list[str] = Field(default_factory=list)
    risk_levels: list[str] = Field(default_factory=list)
    verified: bool = False
    source: str = ""
    standard_codes: list[str] = Field(default_factory=list)
    access_source: str = ""
    verified_by: str = ""
    verified_on: str = ""
    source_path: str = ""
    heading_path: list[str] = Field(
        default_factory=list, description="标题链，如 ['水霉病', '3. 处置方案', '3.1 第一步']"
    )
    heading_text: str = Field(default="", description="标题链的可读形式，用 ' > ' 连接")
    chunk_index: int = Field(default=0, description="块在该文档内的序号，从 0 开始")
    section_role: Literal["domain", "meta"] = Field(
        default="domain",
        description=(
            "该块在文档里扮演的角色。'domain' = 养殖专业知识（可检索）；"
            "'meta' = 讲这份文档如何与本项目代码对接的说明性章节（默认不可检索）。"
        ),
    )
    has_table: bool = Field(default=False, description="是否含 Markdown 表格")
    citation_keys: list[str] = Field(
        default_factory=list, description="正文中出现的引用锚点，如 'GB 11607-1989'"
    )
    char_count: int = 0


class RetrievedChunk(BaseModel):
    """一条检索结果。**这是唯一会进入大模型 prompt 的检索数据结构。**"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str = Field(description="知识块正文（不含面包屑）")
    heading_text: str
    title: str
    category: Category
    disease: DiseaseLabel | None = None
    verified: bool = False
    source: str = ""
    standard_codes: list[str] = Field(default_factory=list)
    verified_on: str = ""
    citation_keys: list[str] = Field(default_factory=list)
    score: float = Field(
        default=0.0,
        description=(
            "融合排序分（RRF 归一化到 0–1）。**只用于排序与相对比较，"
            "不是余弦相似度**——RRF 的数值区间天然很扁（前几名都在 0.9 以上），"
            "不要把它当成『相关度百分比』展示给用户。"
            "真正可解释的相关度看 dense_score。"
        ),
    )
    dense_score: float | None = Field(
        default=None, description="向量余弦相似度（0–1，可解释）；None 表示未被向量检索召回"
    )
    sparse_score: float | None = Field(default=None, description="BM25 分，None 表示未被词面检索召回")
    soft_boost: float = Field(
        default=0.0, description="风险等级/指标软加权带来的加分（只影响排序，不计入 score）"
    )
    dense_rank: int | None = None
    sparse_rank: int | None = None
    matched_by: list[str] = Field(
        default_factory=list, description="召回来源：'dense' / 'sparse'，便于调优排查"
    )

    def provenance_mark(self) -> str:
        """引用标记。**接口引用列表与 prompt 共用这一处口径。**

        已核验的条目必须**正面给出依据**——只标「没核对」而不标「核对过什么」，
        调用方仍然无法判断这条数值能不能直接引用。
        """
        if not self.verified:
            return "（未核对标准原文）"
        codes = "、".join(self.standard_codes)
        if codes and self.verified_on:
            return f"（依据 {codes}；核验日期 {self.verified_on}）"
        return "（已核对标准原文）"

    def reference_line(self) -> str:
        """``references`` 列表里的一行，供接口层与报告层共用。"""
        return f"《{self.title}》> {self.heading_text}{self.provenance_mark()}"

    def to_prompt_block(self) -> str:
        """渲染成注入 prompt 的文本块。

        已核对的标准数值与未核对的种子语料要**显式区分**——
        不区分的话，模型会把二手转述当成国标原文直接引用，
        这是本项目最不想发生的事。
        """
        flag = "已核对原文" if self.verified else "⚠ 未与标准原文核对"
        head = f"[{self.category}] {self.title} > {self.heading_text}（{flag}）"
        # 标准编号与核验日期直接摆到模型眼前。用户要的是"这条建议依据哪份标准"，
        # 而这些信息若只留在正文里，模型很容易在总结时丢掉。
        provenance: list[str] = []
        if self.standard_codes:
            provenance.append("依据 " + "、".join(self.standard_codes))
        if self.verified and self.verified_on:
            provenance.append(f"核验日期 {self.verified_on}")
        if provenance:
            head = head + "\n（" + "；".join(provenance) + "）"
        body = self.text.strip()
        return f"{head}\n{body}"


class RetrievalQuery(BaseModel):
    """一次检索请求。

    ## 硬过滤 vs 软加权

    只有 ``category`` 与 ``disease`` 是**硬过滤**：

        * ``disease`` 必须硬过滤——用户问烂鳃，却检索回水霉的用药方案，
          大模型会把它当成本次病害的处置建议。**召回错病种比召回不到更危险。**
        * ``category`` 是单值字段，语义明确（水质/病害/投料/应急/标准），
          适合做硬边界。

    ``risk_level`` 与 ``indicators`` 是**软加权，不做过滤**：
    它们在元数据里是**列表**（一篇文档可同时适用"预警"与"高危"），
    而 Chroma 的 metadata 不支持列表与 ``$contains``。
    更关键的是：做硬过滤会**误杀**——
    查"高危"时排除掉只标了"预警"的文档，
    而那份文档里可能正好有高危升级后的处置办法。
    因此它们进入排序阶段作为小幅加分项，而不是过滤条件。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="自然语言检索词")
    k: int = Field(default=4, ge=1, le=20)
    purpose: str = Field(
        default="",
        description="本次检索的用途标签，如 '应急处置'/'病害处置'，用于组织注入 prompt 的文本",
    )
    category: Category | None = Field(default=None, description="硬过滤：语料类别")
    disease: DiseaseLabel | None = Field(default=None, description="硬过滤：病害标签")
    risk_level: str | None = Field(default=None, description="软加权：期望的风险等级")
    indicators: list[str] = Field(
        default_factory=list, description="软加权：期望覆盖的指标键"
    )
    allow_disease_null: bool = Field(
        default=True,
        description=(
            "True 时，除该病害专属文档外还允许召回 disease=None 的通用文档"
            "（如水质调控手册）。False 时只召回该病害专属文档。"
        ),
    )
    include_meta: bool = Field(
        default=False,
        description=(
            "是否把『与本项目 Agent 的联动』这类**元章节**也纳入检索。"
            "默认 False：它们是写给评委看的对接说明，不是养殖知识；"
            "纳入后会挤掉真正的处置方案（实测『水霉病怎么处理』的前两名"
            "曾是该章节）。调试语料覆盖度时可打开。"
        ),
    )


class RetrievalResult(BaseModel):
    """一次检索的完整结果，含查询与调优信息。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    purpose: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    n_candidates: int = Field(default=0, description="过滤后参与排序的候选块总数")
    backend: str = Field(default="", description="使用的检索后端描述，便于排查")
    warnings: list[str] = Field(default_factory=list)
