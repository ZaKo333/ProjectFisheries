"""RAG 层（M5）单元测试。

覆盖重点：
    1. **分块不破坏表格**——阈值表被切散后，"氨氮 ≤0.2 适宜"会丢掉分级列
    2. **chunk_id 与正文解耦**——改措辞不该让向量库全量重建
    3. **元章节默认不参与检索**——否则"与本项目 Agent 的联动"会挤掉真处置方案
    4. **检索计划由 Agent 事实驱动**——高危才查应急预案、病种必须硬过滤
    5. **一致性守卫**——向量模型/维度/块数对不上时必须报错，不能悄悄返回旧数据
    6. **离线可跑**——用 HashEmbeddings 让全部用例不依赖 90 MB 向量模型

注意：本文件**大部分用例走 HashEmbeddings**（确定性伪向量）。
它不验证语义质量，只验证**管道正确性**；语义质量由
`scripts/demo_rag.py` 用真实模型人工校验。
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from fisheries.rag.embedder import (
    HashEmbeddings,
    build_embeddings,
    describe_embeddings,
)
from fisheries.rag.loader import (
    CorpusError,
    discover_corpus,
    load_corpus,
    load_document,
    stable_hash,
)
from fisheries.rag.retriever import (
    AgentKnowledgeState,
    KnowledgeBase,
    build_filter,
    build_predicate,
    plan_queries,
    rrf_fuse,
    state_from_agent,
    tokenize,
)
from fisheries.rag.schemas import DocMeta, RetrievalQuery, RetrievedChunk
from fisheries.rag.splitter import (
    MIN_CHUNK_CHARS,
    build_chunk_meta,
    build_embedding_text,
    chunk_corpus,
    chunk_document,
    is_meta_section,
)
from fisheries.rag.store import (
    GENERIC_DISEASE,
    StoreMismatchError,
    chunk_to_metadata,
    corpus_fingerprint,
    load_store,
    read_manifest,
    split_list,
)

# --------------------------------------------------------------------------- #
# 测试语料：刻意包含表格、元章节、不同层级标题、引用锚点
# --------------------------------------------------------------------------- #

DOC_A = """\
---
doc_id: test_threshold
title: 测试用阈值文档
category: 水质
disease: null
indicators: [dissolved_oxygen, ammonia_n]
risk_levels: [正常, 预警]
source: 单元测试构造
verified: false
---

# 测试用阈值文档

## 1. 溶氧分级

溶氧是池塘养殖最关键的指标。低于危险线必须立即开增氧机。

| 指标 | 适宜 | 警戒 | 危险 |
| --- | --- | --- | --- |
| 溶氧 | ≥5.0 | 3.0–5.0 | <3.0 |
| 氨氮 | ≤0.2 | 0.2–0.5 | >0.5 |

表中的数值来自 [GB 11607-89 §4]，具体口径待核对。

## 2. 操作要点

发现溶氧偏低时，应在半小时内开启增氧机，并停止投喂。
夜间至凌晨是泛塘高发时段，需要加密巡塘。

## 3. 与本项目 Agent 的联动

本节说明文档字段如何与 Agent 的代码字段一一对应，属于工程说明而非养殖知识。
agent 字段 water_temp 对应本文档的"水温"小节，dissolved_oxygen 对应"溶氧"小节。
"""

DOC_B = """\
---
doc_id: test_gill
title: 测试用烂鳃处置
category: 病害
disease: 烂鳃
indicators: [dissolved_oxygen]
risk_levels: [高危]
source: 单元测试构造
verified: true
---

# 测试用烂鳃处置

## 1. 立即措施

第一步是停料，并加大换水量。第二步是全池泼洒消毒剂，具体用量需遵医嘱。

## 2. 用药合规

用药必须遵守休药期规定。禁止使用孔雀石绿等禁用药物。
"""

DOC_SAPRO = """\
---
doc_id: test_sapro
title: 测试用水霉处置
category: 病害
disease: 水霉
indicators: [water_temp]
risk_levels: [预警]
source: 单元测试构造
verified: false
---

# 测试用水霉处置

## 1. 立即措施

水霉病的处置重点是消除应激源，而不是用药。先提高水温，再考虑药物治疗。
"""

DOC_EMERGENCY = """\
---
doc_id: test_emergency
title: 测试用应急处置预案
category: 应急
indicators: [dissolved_oxygen]
risk_levels: [高危]
source: 单元测试构造
verified: false
---

# 测试用应急处置预案

## 1. 泛塘的识别

鱼群浮头且人走近也不下沉，说明已经进入泛塘阶段，必须立即抢救。

## 2. 分钟级响应

立即开启全部增氧机，同时加注新水，并停止一切投喂。必要时抛撒增氧剂。
"""


DOC_FEED = """\
---
doc_id: test_feeding
title: 测试用投料管理规范
category: 投料
indicators: []
risk_levels: [正常, 预警]
source: 单元测试构造
verified: false
---

# 测试用投料管理规范

## 1. 四定原则

投喂要做到定时、定量、定质、定位，避免残饵败坏水质。

## 2. 投喂量必须下调的情况

溶氧偏低时应减料三到五成；氨氮超过 0.5 mg/L 时直接停料。
发病期应停料并观察，恢复投喂要从正常量的三成起逐步加回。
"""


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """五篇最小语料：水质（含表格与元章节）+ 烂鳃 + 水霉 + 应急 + 投料。"""
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "01_阈值.md").write_text(DOC_A, encoding="utf-8")
    (d / "02_烂鳃.md").write_text(DOC_B, encoding="utf-8")
    (d / "03_水霉.md").write_text(DOC_SAPRO, encoding="utf-8")
    (d / "04_应急.md").write_text(DOC_EMERGENCY, encoding="utf-8")
    (d / "05_投料.md").write_text(DOC_FEED, encoding="utf-8")
    return d


@pytest.fixture()
def kb(corpus_dir: Path, tmp_path: Path) -> KnowledgeBase:
    """用 HashEmbeddings 构建一个临时知识库（不加载真实模型）。"""
    emb = HashEmbeddings(dim=64)
    base, _stats = KnowledgeBase.build(
        corpus_dir=corpus_dir,
        persist_dir=tmp_path / "chroma",
        embeddings=emb,
        max_chars=200,
        overlap_chars=40,
    )
    return base


# --------------------------------------------------------------------------- #
# 1. loader —— front-matter 与语料发现
# --------------------------------------------------------------------------- #


class TestLoader:
    def test_front_matter_parsed(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "01_阈值.md")
        assert doc.meta.doc_id == "test_threshold"
        assert doc.meta.category == "水质"
        assert doc.meta.verified is False
        assert doc.meta.indicators == ["dissolved_oxygen", "ammonia_n"]
        # 正文里不应残留 front-matter 的分隔符
        assert not doc.body.startswith("---")
        assert "# 测试用阈值文档" in doc.body

    def test_disease_null_becomes_none(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "01_阈值.md")
        assert doc.meta.disease is None

    def test_disease_label_parsed(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "02_烂鳃.md")
        assert doc.meta.disease == "烂鳃"
        assert doc.meta.verified is True

    def test_discover_is_sorted_and_top_level_only(self, corpus_dir: Path) -> None:
        found = discover_corpus(corpus_dir)
        names = [p.name for p in found]
        assert names == sorted(names), "语料必须按文件名排序，才能保证 chunk_index 确定"
        assert names == ["01_阈值.md", "02_烂鳃.md", "03_水霉.md", "04_应急.md", "05_投料.md"]

        # 子目录里的文件不参与（否则备份/草稿会被意外入库）
        nested = corpus_dir / "draft"
        nested.mkdir()
        (nested / "99_x.md").write_text(DOC_A, encoding="utf-8")
        assert [p.name for p in discover_corpus(corpus_dir)] == names

    def test_duplicate_doc_id_rejected(self, corpus_dir: Path) -> None:
        (corpus_dir / "04_dup.md").write_text(DOC_A, encoding="utf-8")
        with pytest.raises(CorpusError, match="doc_id"):
            load_corpus(corpus_dir)

    def test_missing_front_matter_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.md"
        p.write_text("# 没有 front-matter\n\n正文。\n", encoding="utf-8")
        with pytest.raises(CorpusError):
            load_document(p)

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.md"
        p.write_text(
            "---\ndoc_id: x\ntitle: t\ncategory: 不存在的类别\n---\n\n# t\n正文\n",
            encoding="utf-8",
        )
        with pytest.raises(CorpusError):
            load_document(p)

    def test_stable_hash_is_process_independent(self) -> None:
        """内置 hash() 对 str 加随机盐，进程重启就变——不能让 chunk_id 用它。"""
        a = stable_hash("doc", "标题", 1)
        assert a == stable_hash("doc", "标题", 1)
        assert a != stable_hash("doc", "标题", 2)
        assert len(a) == 12


# --------------------------------------------------------------------------- #
# 2. splitter —— 标题感知分块
# --------------------------------------------------------------------------- #


class TestSplitter:
    def test_h1_not_in_heading_path(self, corpus_dir: Path) -> None:
        """H1 是文档标题，重复出现在每个块的面包屑里是噪声。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        chunks = chunk_document(doc)
        assert chunks
        for c in chunks:
            assert "测试用阈值文档" not in c.heading_path

    def test_heading_path_rebuilt_correctly(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "01_阈值.md")
        paths = [" > ".join(c.heading_path) for c in chunk_document(doc)]
        assert any("溶氧分级" in p for p in paths)
        assert any("操作要点" in p for p in paths)

    def test_table_is_kept_intact(self, corpus_dir: Path) -> None:
        """表格被切散 → 阈值数字与"适宜/警戒/危险"表头分离 → 检索到的规则不完整。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        chunks = chunk_document(doc)
        table_chunks = [c for c in chunks if c.has_table]
        assert table_chunks, "阈值表没有被识别为表格块"

        joined = "\n".join(c.text for c in table_chunks)
        # 表头、两条数据行、以及分级列都必须落在同一个块里
        assert "| 指标 | 适宜 | 警戒 | 危险 |" in joined
        assert "| 溶氧 | ≥5.0 | 3.0–5.0 | <3.0 |" in joined
        assert "| 氨氮 | ≤0.2 | 0.2–0.5 | >0.5 |" in joined

        # 更严格：整张表必须在**同一个**块内，不能跨块
        whole = next(c for c in table_chunks if "氨氮" in c.text)
        assert "| 指标 | 适宜 | 警戒 | 危险 |" in whole.text
        assert "| 溶氧 |" in whole.text

    def test_table_never_silently_truncated(self, tmp_path: Path) -> None:
        """超长表格独占一块并告警，绝不截断——截断的阈值表比没有更危险。"""
        rows = "\n".join(f"| 指标{i} | 1 | 2 | 3 |" for i in range(80))
        doc_text = textwrap.dedent(f"""\
            ---
            doc_id: big_table
            title: 大表格
            category: 水质
            ---

            # 大表格

            ## 1. 超长表格

            | 指标 | 适宜 | 警戒 | 危险 |
            | --- | --- | --- | --- |
            {rows}
            """)
        p = tmp_path / "big.md"
        p.write_text(doc_text, encoding="utf-8")
        chunks = chunk_document(load_document(p), max_chars=100)
        table_chunks = [c for c in chunks if c.has_table]
        assert len(table_chunks) == 1
        assert table_chunks[0].text.count("| 指标") == 81  # 表头 + 80 行
        assert any("表格" in w for w in table_chunks[0].warnings)

    # ---- 列表原子性：应急流程被切散比不召回更危险 ----

    def test_numbered_steps_stay_in_one_chunk(self, tmp_path: Path) -> None:
        """标题写着"按顺序，不要跳步"，切散后模型只会看到从第 N 步开始的残缺流程。

        这里故意让步骤之间**空一行**——语料里就是这么写的。
        若把空行当分块边界，每条步骤会各成一块，等于没做原子化。
        """
        steps = "\n\n".join(
            f"{i}. 第 {i} 步：开启增氧设备并保持运行，观察鱼群反应。" for i in range(1, 13)
        )
        doc_text = textwrap.dedent(f"""\
            ---
            doc_id: steps
            title: 步骤文档
            category: 应急
            ---

            # 步骤文档

            ## 1. 立即动作（按顺序，不要跳步）

            {steps}
            """)
        p = tmp_path / "steps.md"
        p.write_text(doc_text, encoding="utf-8")
        chunks = chunk_document(load_document(p), max_chars=200)

        lists = [c for c in chunks if "第 1 步" in c.text or "第 12 步" in c.text]
        assert len(lists) == 1, [c.text[:40] for c in chunks]
        joined = lists[0].text
        # 第一步与最后一步必须同块，中间不能断
        assert "第 1 步" in joined and "第 12 步" in joined
        assert all(f"第 {i} 步" in joined for i in range(1, 13))

    def test_oversized_list_kept_whole_with_warning(self, tmp_path: Path) -> None:
        """超长列表独占一块并告警，绝不截断——缺步骤比不召回更糟。"""
        steps = "\n".join(f"{i}. 第 {i} 步做一件需要占位说明的具体事情。" for i in range(1, 21))
        doc_text = textwrap.dedent(f"""\
            ---
            doc_id: longlist
            title: 长列表
            category: 应急
            ---

            # 长列表

            ## 1. 长流程

            {steps}
            """)
        p = tmp_path / "long.md"
        p.write_text(doc_text, encoding="utf-8")
        # cohesion_chars=0 关掉「整节装得下就不切」的规则，专门验证打包路径。
        chunks = chunk_document(load_document(p), max_chars=100, cohesion_chars=0)
        target = [c for c in chunks if "第 1 步" in c.text]
        assert len(target) == 1
        assert target[0].text.count("步做一件") == 20
        assert any("列表" in w for w in target[0].warnings)

    def test_list_continuation_line_stays_with_its_item(self, tmp_path: Path) -> None:
        """缩进的续行属于上一条步骤，不能被切成独立段落。"""
        doc_text = textwrap.dedent("""\
            ---
            doc_id: cont
            title: 续行
            category: 应急
            ---

            # 续行

            ## 1. 流程

            1. 开启增氧机——
               全部开机，不要留备用；
            2. 停止投喂。
            """)
        p = tmp_path / "cont.md"
        p.write_text(doc_text, encoding="utf-8")
        chunks = chunk_document(load_document(p), max_chars=500)
        joined = "\n".join(c.text for c in chunks)
        assert "开启增氧机——\n   全部开机" in joined

    def test_top_level_paragraph_ends_list_run(self, tmp_path: Path) -> None:
        """列表后面的顶格段落必须是新块，不能被吸进列表里。"""
        doc_text = textwrap.dedent("""\
            ---
            doc_id: tail
            title: 收尾
            category: 应急
            ---

            # 收尾

            ## 1. 流程

            1. 第一步：开启全部增氧机并保持连续运行，直到日出后两小时再评估是否停机。
            2. 第二步：立即停止一切投喂，并捞出死鱼做无害化处理，避免继续耗氧。

            这是一段顶格的说明文字，讲的是与本流程无关的记录要求，应当单独成块，而不要被吸进上面的列表里，否则列表的原子性就被破坏了。
            """)
        p = tmp_path / "tail.md"
        p.write_text(doc_text, encoding="utf-8")
        # 同样关掉 cohesion，否则整节会被合并成一块，看不出块边界。
        chunks = chunk_document(load_document(p), max_chars=40, cohesion_chars=0)
        texts = [c.text for c in chunks]
        assert any(t.startswith("1. 第一步") and "无关" not in t for t in texts), texts
        assert any("顶格" in t and "第一步" not in t for t in texts), texts

    def test_small_section_kept_whole_despite_max_chars(self, tmp_path: Path) -> None:
        """整节只要不超过 cohesion 上限就一块装完——标题是比字符数更硬的边界。

        实测教训：549 字的「1.2 立即动作（按顺序，不要跳步）」被 max_chars=500
        切成 407+142 两块，检索端按标题去重只留一块，
        于是大模型拿到的是**从第 5 步开始的残缺应急流程**。
        """
        steps = "\n".join(f"{i}. 第 {i} 步做一件需要占位说明的具体事情。" for i in range(1, 19))
        doc_text = textwrap.dedent(f"""\
            ---
            doc_id: cohesion
            title: 粘合
            category: 应急
            ---

            # 粘合

            ## 1. 完整流程

            **第一阶段**

            {steps}
            """)
        p = tmp_path / "cohesion.md"
        p.write_text(doc_text, encoding="utf-8")
        chunks = chunk_document(load_document(p), max_chars=100, cohesion_chars=800)
        assert len(chunks) == 1
        assert "第 1 步" in chunks[0].text and "第 18 步" in chunks[0].text

    def test_short_chunks_merged_only_within_same_heading(self, corpus_dir: Path) -> None:
        """跨小节合并会把 A 小节的内容挂到 B 小节名下，检索时张冠李戴。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        for c in chunk_document(doc, max_chars=200):
            assert len(c.heading_path) <= 2

    def test_chunk_id_independent_of_body_text(self, corpus_dir: Path) -> None:
        """改措辞不该让整个向量库重建——chunk_id 只由"位置"决定。"""
        src = corpus_dir / "01_阈值.md"
        doc = load_document(src)
        texts_before = [c.text for c in chunk_document(doc)]
        ids_before = [
            build_chunk_meta(doc, c, i).chunk_id for i, c in enumerate(chunk_document(doc))
        ]

        # 就地改一句措辞（不新增/删除标题、不改变段落数）
        src.write_text(
            DOC_A.replace("低于危险线必须立即开增氧机", "低于危险线要马上开增氧机"),
            encoding="utf-8",
        )
        doc2 = load_document(src)
        ids_after = [
            build_chunk_meta(doc2, c, i).chunk_id for i, c in enumerate(chunk_document(doc2))
        ]
        assert ids_before == ids_after
        assert texts_before != [c.text for c in chunk_document(doc2)]

    def test_chunk_id_changes_when_section_added(self, corpus_dir: Path) -> None:
        """插入小节会改变后续块的序号，chunk_id 必须跟着变（否则会指向错内容）。"""
        src = corpus_dir / "01_阈值.md"
        doc = load_document(src)
        before = [build_chunk_meta(doc, c, i).chunk_id
                  for i, c in enumerate(chunk_document(doc))]

        src.write_text(DOC_A.replace("## 2. 操作要点", "## 1.5 新增小节\n\n新增的一句话说明。\n\n## 2. 操作要点"),
                       encoding="utf-8")
        doc2 = load_document(src)
        after = [build_chunk_meta(doc2, c, i).chunk_id
                 for i, c in enumerate(chunk_document(doc2))]
        assert before != after

    def test_embedding_text_has_breadcrumb(self, corpus_dir: Path) -> None:
        """向量化文本带标题链，短块（"见上表"）才有机会被召回。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        chunks = chunk_document(doc)
        meta = build_chunk_meta(doc, chunks[-1], 0)
        text = build_embedding_text(meta, chunks[-1].text)
        assert meta.heading_text in text
        assert text.endswith(chunks[-1].text.strip())

    def test_meta_section_detected(self) -> None:
        assert is_meta_section(["3. 与本项目 Agent 的联动"])
        assert is_meta_section(["3. 与本项目 Agent 的联动", "3.1 字段对照"])
        assert not is_meta_section(["1. 溶氧分级"])

    def test_meta_section_ignores_incidental_mention(self) -> None:
        """领域小节只是**顺带**提到本项目，不能被当成元章节排掉。

        真实事故：`## 4. 投喂量必须下调的四种情况（本项目 Agent 的决策依据）`
        整节连同 4.1–4.4 四个子节被判成元章节、从检索里消失，
        而"溶氧偏低时减料""氨氮超标停料"正是 Agent 最需要的依据。
        """
        assert not is_meta_section(["4. 投喂量必须下调的四种情况（本项目 Agent 的决策依据）"])
        assert not is_meta_section(
            ["4. 投喂量必须下调的四种情况（本项目 Agent 的决策依据）", "4.1 溶氧偏低时减料"]
        )
        # 同理：真正讲标准归属的章节也是领域内容，不是元信息。
        assert not is_meta_section(["1. 本项目涉及的标准清单"])
        assert not is_meta_section(["3. 本项目阈值与标准的关系（重要说明）"])

    def test_meta_section_catches_project_advantage_talk(self) -> None:
        """`2.4 本项目的预测性优势` 通篇在讲本项目，旧的子串表反而漏掉了它。"""
        assert is_meta_section(["2. 溶解氧偏低（最紧急）", "2.4 本项目的预测性优势"])

    def test_meta_section_strips_heading_numbers(self) -> None:
        for title in ("7. 与本项目代码的对应关系", "7、与本项目代码的对应关系", "7 与本项目代码的对应关系"):
            assert is_meta_section([title]), title

    def test_meta_section_tagged_in_metadata(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "01_阈值.md")
        metas = [build_chunk_meta(doc, c, i) for i, c in enumerate(chunk_document(doc))]
        roles = {m.heading_text: m.section_role for m in metas}
        assert roles.get("1. 溶氧分级") == "domain"
        assert roles.get("3. 与本项目 Agent 的联动") == "meta"
        assert any(m.section_role == "meta" for m in metas)

    def test_citation_keys_extracted(self, corpus_dir: Path) -> None:
        doc = load_document(corpus_dir / "01_阈值.md")
        metas = [build_chunk_meta(doc, c, i) for i, c in enumerate(chunk_document(doc))]
        all_cites = {k for m in metas for k in m.citation_keys}
        assert any("GB 11607" in k for k in all_cites)

    def test_no_dangling_short_fragment(self, corpus_dir: Path) -> None:
        """短块必须**单独占一个小节**。

        `MIN_CHUNK_CHARS` 是"合并目标"而不是"硬下限"：
        一个小节本身就只写了两句话（如"2. 操作要点"），它没有兄弟块可合并——
        强行合并会把它挂到相邻小节名下，检索时张冠李戴。
        但**同一个标题链下有多个块时出现短块**就是真的碎片，
        说明切分把一段话拆散了。
        """
        doc = load_document(corpus_dir / "01_阈值.md")
        chunks = chunk_document(doc)
        by_heading: dict[tuple[str, ...], list] = {}
        for c in chunks:
            by_heading.setdefault(tuple(c.heading_path), []).append(c)

        for heading, group in by_heading.items():
            if len(group) == 1:
                continue
            for c in group:
                assert len(c.text.strip()) >= MIN_CHUNK_CHARS or c.has_table, (
                    f"标题 {heading} 下有 {len(group)} 个块，"
                    f"其中一块只有 {len(c.text.strip())} 字：{c.text[:40]!r}"
                )

    def test_chunk_corpus_counts_docs(self, corpus_dir: Path) -> None:
        docs = load_corpus(corpus_dir)
        pairs = chunk_corpus(docs)
        assert len({m.doc_id for m, _ in pairs}) == len(docs)
        assert all(m.chunk_id and t for m, t in pairs)


# --------------------------------------------------------------------------- #
# 3. embedder —— 离线与后端选择
# --------------------------------------------------------------------------- #


class TestEmbedder:
    def test_hash_embeddings_deterministic(self) -> None:
        emb = HashEmbeddings(dim=32)
        a = emb.embed_query("溶氧偏低怎么办")
        b = emb.embed_query("溶氧偏低怎么办")
        assert a == b
        assert len(a) == 32

    def test_hash_embeddings_differ_by_text(self) -> None:
        emb = HashEmbeddings(dim=32)
        assert emb.embed_query("溶氧") != emb.embed_query("氨氮")

    def test_hash_embeddings_normalised(self) -> None:
        """L2 归一化后，余弦与内积/欧氏距离单调一致 —— 换距离度量不改变排序。"""
        v = HashEmbeddings(dim=32).embed_query("溶氧 3.0 危险线")
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-6

    def test_hash_embeddings_batch_matches_single(self) -> None:
        emb = HashEmbeddings(dim=32)
        texts = ["溶氧", "氨氮", "亚硝酸盐"]
        assert emb.embed_documents(texts) == [emb.embed_query(t) for t in texts]

    def test_unknown_backend_raises_not_silently_degrades(self) -> None:
        """拼错后端名时静默降级成 hash 会让"语义检索"悄悄失效。"""
        with pytest.raises(ValueError, match="未知的 embedding 后端"):
            build_embeddings("fastembedd")

    def test_build_embeddings_hash(self) -> None:
        emb = build_embeddings("hash")
        assert "hash" in describe_embeddings(emb)


# --------------------------------------------------------------------------- #
# 4. store —— 元数据编码与一致性守卫
# --------------------------------------------------------------------------- #


class TestStore:
    def test_lists_joined_with_separator(self, corpus_dir: Path) -> None:
        """Chroma 的 metadata 不接受列表，必须编码成字符串。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        meta = build_chunk_meta(doc, chunk_document(doc)[0], 0)
        md = chunk_to_metadata(meta)
        assert isinstance(md["heading_path"], str)
        assert isinstance(md["indicators"], str)
        for value in md.values():
            assert not isinstance(value, (list, dict))

    def test_split_list_roundtrip(self) -> None:
        assert split_list("a|b|c") == ["a", "b", "c"]
        assert split_list("") == []
        assert split_list(None) == []
        assert split_list(["a", "b"]) == ["a", "b"]

    def test_disease_none_encoded_as_generic(self, corpus_dir: Path) -> None:
        """disease=None 存成字面量 "通用"：Chroma 对 $eq: None 的语义跨版本不一致。"""
        doc = load_document(corpus_dir / "01_阈值.md")
        meta = build_chunk_meta(doc, chunk_document(doc)[0], 0)
        assert chunk_to_metadata(meta)["disease"] == GENERIC_DISEASE

    def test_fingerprint_changes_with_chunks(self, corpus_dir: Path) -> None:
        pairs = chunk_corpus(load_corpus(corpus_dir))
        fp1 = corpus_fingerprint(pairs)
        fp2 = corpus_fingerprint(pairs + [pairs[0]])
        assert fp1 != fp2

    def test_fingerprint_ignores_light_edits(self, corpus_dir: Path) -> None:
        """改标点不该触发全库重建。"""
        pairs = chunk_corpus(load_corpus(corpus_dir))
        edited = [(m, t.replace("，", ",")) for m, t in pairs]
        assert corpus_fingerprint(pairs) == corpus_fingerprint(edited)

    def test_manifest_written_and_readable(self, kb: KnowledgeBase, tmp_path: Path) -> None:
        man = read_manifest(tmp_path / "chroma")
        assert man["collection"]
        assert man["n_chunks"] == len(kb)
        assert man["dim"] == 64
        assert "fingerprint" in man
        assert "built_at" in man

    def test_load_store_strict_rejects_wrong_dim(self, corpus_dir: Path, tmp_path: Path) -> None:
        """维度不匹配说明换了向量模型——必须报错，不能返回一堆垃圾检索结果。"""
        KnowledgeBase.build(
            corpus_dir=corpus_dir,
            persist_dir=tmp_path / "chroma",
            embeddings=HashEmbeddings(dim=64),
        )
        with pytest.raises(StoreMismatchError):
            load_store(HashEmbeddings(dim=128), persist_dir=tmp_path / "chroma")


# --------------------------------------------------------------------------- #
# 5. retriever —— 过滤、融合、检索计划
# --------------------------------------------------------------------------- #


class TestTokenizer:
    def test_domain_words_kept_whole(self) -> None:
        """jieba 默认会把"亚硝酸盐"切成"亚硝/酸盐"，导致查询匹配不上标准原文写法。"""
        for word in ("亚硝酸盐", "溶解氧", "增氧机", "水霉病", "氨氮", "烂鳃"):
            assert word in tokenize(f"{word}超标了怎么办"), word

    def test_stopwords_removed(self) -> None:
        assert "的" not in tokenize("溶氧的数值")
        assert "怎么" not in tokenize("怎么处理")

    def test_lowercased(self) -> None:
        assert "ph" in tokenize("PH 偏高")


class TestFilter:
    def test_category_hard_filter(self) -> None:
        pred = build_predicate(RetrievalQuery(query="x", category="病害"))
        assert pred({"category": "病害", "disease": "通用", "section_role": "domain"})
        assert not pred({"category": "水质", "disease": "通用", "section_role": "domain"})

    def test_disease_hard_filter_blocks_other_disease(self) -> None:
        """跨病种串用药方案是最不能接受的失败模式。"""
        pred = build_predicate(RetrievalQuery(query="x", disease="烂鳃"))
        assert pred({"category": "病害", "disease": "烂鳃", "section_role": "domain"})
        assert not pred({"category": "病害", "disease": "水霉", "section_role": "domain"})

    def test_generic_allowed_when_allow_disease_null(self) -> None:
        pred = build_predicate(RetrievalQuery(query="x", disease="烂鳃", allow_disease_null=True))
        assert pred({"category": "病害", "disease": GENERIC_DISEASE, "section_role": "domain"})

    def test_generic_blocked_when_strict(self) -> None:
        pred = build_predicate(RetrievalQuery(query="x", disease="烂鳃", allow_disease_null=False))
        assert not pred({"category": "病害", "disease": GENERIC_DISEASE, "section_role": "domain"})

    def test_meta_sections_excluded_by_default(self) -> None:
        """实测：不排除时"水霉病怎么处理"的前两名是"与本项目 Agent 的联动"。"""
        pred = build_predicate(RetrievalQuery(query="x"))
        assert not pred({"section_role": "meta"})
        assert pred({"section_role": "domain"})

    def test_meta_sections_included_when_asked(self) -> None:
        pred = build_predicate(RetrievalQuery(query="x", include_meta=True))
        assert pred({"section_role": "meta"})

    def test_chroma_filter_translation(self) -> None:
        f = build_filter(RetrievalQuery(query="x", category="病害", disease="烂鳃",
                                        allow_disease_null=False))
        assert isinstance(f, dict)
        assert "$and" in f

    def test_risk_level_not_in_chroma_filter(self) -> None:
        """risk_level 是软加权，做成硬过滤会误杀"只标了预警但含高危措施"的文档。"""
        f = build_filter(RetrievalQuery(query="x", risk_level="高危")) or {}
        assert "risk_level" not in json.dumps(f)


class TestRRF:
    def test_fuse_is_deterministic(self) -> None:
        dense = [(3, 0.9), (1, 0.8), (2, 0.7)]
        sparse = [(1, 12.0), (3, 9.0), (0, 5.0)]
        a = [(h.index, round(h.rrf, 9)) for h in rrf_fuse(dense, sparse)]
        b = [(h.index, round(h.rrf, 9)) for h in rrf_fuse(dense, sparse)]
        assert a == b

    def test_agreement_beats_single_route(self) -> None:
        """被两路都召回的结果应排在只有一路召回的前面。"""
        dense = [(0, 0.9), (1, 0.8)]
        sparse = [(1, 9.0), (2, 5.0)]
        fused = rrf_fuse(dense, sparse)
        assert fused[0].index == 1

    def test_sources_recorded(self) -> None:
        fused = rrf_fuse([(0, 1.0)], [(0, 1.0)])
        assert set(fused[0].sources) == {"dense", "sparse"}


class TestPlanQueries:
    def test_normal_state_has_no_emergency_query(self) -> None:
        """预案里全是"立即/紧急"，正常状态下混进去会让模型把巡检写成告警。"""
        plan = plan_queries(AgentKnowledgeState(risk_level="正常"))
        purposes = [q.purpose for q in plan]
        assert "应急处置" not in purposes
        assert "水质调控" in purposes
        assert "投料管理" in purposes

    def test_high_risk_adds_emergency_query(self) -> None:
        plan = plan_queries(AgentKnowledgeState(risk_level="高危"))
        assert "应急处置" in [q.purpose for q in plan]

    def test_redline_alone_adds_emergency_query(self) -> None:
        """等级判定可能因加权而偏低，红线命中是更硬的信号。"""
        plan = plan_queries(
            AgentKnowledgeState(risk_level="预警", redline_hits=["DO_REDLINE"])
        )
        assert "应急处置" in [q.purpose for q in plan]

    def test_healthy_fish_has_no_disease_query(self) -> None:
        plan = plan_queries(
            AgentKnowledgeState(risk_level="正常", disease_type="健康", is_healthy=True)
        )
        assert "病害处置" not in [q.purpose for q in plan]

    def test_disease_query_is_hard_filtered(self) -> None:
        plan = plan_queries(
            AgentKnowledgeState(risk_level="高危", disease_type="烂鳃", is_healthy=False)
        )
        disease_q = next(q for q in plan if q.purpose == "病害处置")
        assert disease_q.disease == "烂鳃"
        assert disease_q.allow_disease_null is False
        assert disease_q.category == "病害"

    def test_other_disease_family_not_queried(self) -> None:
        """寄生虫/病毒性白尾没有对应处置文档，检索只会返回不相关病种。"""
        plan = plan_queries(
            AgentKnowledgeState(risk_level="预警", disease_type="其他", is_healthy=False)
        )
        assert "病害处置" not in [q.purpose for q in plan]

    def test_feeding_still_queried_when_diseased(self) -> None:
        """病害处置第一步几乎都是"停料"，停几天/怎么恢复只有投料规范写了。"""
        plan = plan_queries(
            AgentKnowledgeState(risk_level="高危", disease_type="烂鳃", is_healthy=False)
        )
        feed = next(q for q in plan if q.purpose == "投料管理")
        assert "停料" in feed.query

    def test_query_order_is_priority_order(self) -> None:
        plan = plan_queries(
            AgentKnowledgeState(risk_level="高危", disease_type="烂鳃", is_healthy=False)
        )
        assert [q.purpose for q in plan] == [
            "应急处置", "水质调控", "病害处置", "投料管理"
        ]

    # ---- 应急处置的定向化：这是"按事实检索"最容易被做糊的一处 ----

    def test_do_redline_targets_pantu_section(self) -> None:
        """预案把缺氧写成「1. 泛塘（严重缺氧）」，标题里没有"溶氧"二字。

        只按指标名检索会落到氨氮/亚硝酸盐两节（实测发生过），
        所以必须把指标翻译成文档真正使用的措辞。
        """
        plan = plan_queries(
            AgentKnowledgeState(
                risk_level="高危",
                limiting_factor="dissolved_oxygen",
                redline_hits=["DO_REDLINE_PROJECTED"],
            )
        )
        emergency = [q for q in plan if q.purpose == "应急处置"]
        assert len(emergency) == 1
        assert "泛塘" in emergency[0].query
        assert "缺氧" in emergency[0].query

    def test_do_and_ammonia_yield_separate_emergency_queries(self) -> None:
        """混在一条 query 里时，泛塘小节会被长度归一化压下去。

        分组后每个指标各查一次，主控因子（溶氧）必须排在最前。
        """
        plan = plan_queries(
            AgentKnowledgeState(
                risk_level="高危",
                limiting_factor="dissolved_oxygen",
                redline_hits=["DO_REDLINE_PROJECTED"],
                severe_indicators=["ammonia_n", "dissolved_oxygen"],
            )
        )
        emergency = [q for q in plan if q.purpose == "应急处置"]
        assert len(emergency) == 2
        assert "泛塘" in emergency[0].query        # 主控因子优先
        assert "氨氮" in emergency[1].query
        assert "泛塘" not in emergency[1].query    # 不把两组词混在一起

    def test_emergency_words_absent_from_other_purposes(self) -> None:
        """应急章节词只出现在应急查询里，别把水质查询也带偏。"""
        plan = plan_queries(
            AgentKnowledgeState(
                risk_level="高危",
                limiting_factor="dissolved_oxygen",
                redline_hits=["DO_REDLINE_PROJECTED"],
            )
        )
        water_q = next(q for q in plan if q.purpose == "水质调控")
        assert "泛塘" not in water_q.query

    def test_emergency_queries_capped_at_three(self) -> None:
        """再多会把预算吃光，反而挤掉水质/病害/投料。"""
        plan = plan_queries(
            AgentKnowledgeState(
                risk_level="高危",
                limiting_factor="dissolved_oxygen",
                redline_hits=["DO_REDLINE", "NH3_REDLINE", "NO2_REDLINE", "PH_REDLINE"],
                severe_indicators=["ammonia_n", "nitrite_n", "ph", "water_temp"],
            )
        )
        assert len([q for q in plan if q.purpose == "应急处置"]) == 3

    def test_high_risk_without_indicator_still_gets_generic_emergency(self) -> None:
        """融合分≥70 也能判高危，此时没有任何具体指标可定向。

        不能因此就干脆不召回预案——只是退回不带指标词的通用应急查询。
        """
        plan = plan_queries(AgentKnowledgeState(risk_level="高危"))
        emergency = [q for q in plan if q.purpose == "应急处置"]
        assert len(emergency) == 1
        assert "应急" in emergency[0].query
        assert "泛塘" not in emergency[0].query

    def test_state_from_agent_extracts_facts(self) -> None:
        st = state_from_agent(
            water={
                "status": "ok",
                "limiting_factor": "dissolved_oxygen",
                "limiting_factor_label": "溶氧",
                "redline_hits": ["DO_REDLINE_PROJECTED"],
                "severity_map": {"dissolved_oxygen": 1.0, "ph": 0.0, "ammonia_n": 0.5},
                "projected_do_alert_eta_hours": 6.9,
                "trends": {"water_temp": {"current": 22.4}},
            },
            vision={"status": "ok", "disease_type": "水霉", "is_healthy": False},
            fusion={"risk_level": "高危"},
        )
        assert st.risk_level == "高危"
        assert st.limiting_factor_label == "溶氧"
        assert st.severe_indicators == ["ammonia_n", "dissolved_oxygen"]
        assert st.water_temp == 22.4
        assert st.disease_type == "水霉"
        assert st.is_healthy is False

    def test_state_from_failed_modality_is_safe(self) -> None:
        """工具报错时 status != "ok"，不能把 error 里的字段当事实。"""
        st = state_from_agent(
            water={"status": "error", "error_code": "INVALID_INPUT"},
            vision={"status": "error", "disease_type": "烂鳃"},
            fusion={"risk_level": "预警"},
        )
        assert st.limiting_factor_label == ""
        assert st.disease_type == ""
        assert st.redline_hits == []


# --------------------------------------------------------------------------- #
# 6. KnowledgeBase —— 端到端（HashEmbeddings）
# --------------------------------------------------------------------------- #


class TestKnowledgeBase:
    def test_meta_sections_not_retrieved(self, kb: KnowledgeBase) -> None:
        """真实语料里这个问题曾让"水霉病怎么处理"的前两名变成工程说明。"""
        res = kb.search(RetrievalQuery(query="字段 对照 联动", category="水质", k=5))
        for c in res.chunks:
            assert "联动" not in c.heading_text

    def test_disease_filter_end_to_end(self, kb: KnowledgeBase) -> None:
        res = kb.search(
            RetrievalQuery(query="水霉 立即措施", category="病害", disease="水霉",
                           allow_disease_null=False, k=3)
        )
        assert res.chunks
        for c in res.chunks:
            assert c.disease == "水霉"

    def test_retrieve_for_agent_returns_context(self, kb: KnowledgeBase) -> None:
        ctx = kb.retrieve_for_agent(
            water={
                "status": "ok",
                "limiting_factor_label": "溶氧",
                "redline_hits": ["DO_REDLINE_PROJECTED"],
                "severity_map": {"dissolved_oxygen": 1.0},
                "trends": {"water_temp": {"current": 28.0}},
            },
            vision={"status": "ok", "disease_type": "烂鳃", "is_healthy": False},
            fusion={"risk_level": "高危"},
        )
        assert ctx.n_chunks > 0
        assert "应急处置" in ctx.by_purpose
        rendered = ctx.render()
        # 按用途分组的结构：让模型知道"这段讲救急、那段讲投料"
        assert "### 【应急处置】" in rendered
        assert "### 【水质调控】" in rendered
        assert "### 【投料管理】" in rendered
        # 每条都带核对状态与可解释的相关度
        assert "未与标准原文核对" in rendered or "已核对原文" in rendered
        assert "向量相似度" in rendered or "词面命中" in rendered
        # 高危场景必须真的召回应急预案里的动作
        assert "增氧机" in rendered
        # 有病害时投料规范也要在（"停料几天、怎么恢复"只写在投料规范里）
        assert "停料" in rendered

    def test_unverified_chunks_flagged_in_prompt(self, kb: KnowledgeBase) -> None:
        """未核对的标准数值必须显式标注，否则模型会当成国标原文引用。"""
        ctx = kb.retrieve_for_agent(
            water={"status": "ok", "limiting_factor_label": "溶氧",
                   "severity_map": {"dissolved_oxygen": 1.0}},
            fusion={"risk_level": "预警"},
        )
        assert ctx.chunks
        assert any("未与标准原文核对" in c.to_prompt_block() for c in ctx.chunks)
        assert ctx.all_unverified is True

    def test_verified_chunk_not_flagged(self, kb: KnowledgeBase) -> None:
        res = kb.search(
            RetrievalQuery(query="烂鳃 停药期", category="病害", disease="烂鳃",
                           allow_disease_null=False, k=3)
        )
        v = [c for c in res.chunks if c.verified]
        assert v
        assert "未与标准原文核对" not in v[0].to_prompt_block()

    def test_round_robin_budget_keeps_low_priority_purpose(self, kb: KnowledgeBase) -> None:
        """max_chunks 截断不该让排最后的「投料管理」整块消失。"""
        ctx = kb.retrieve_for_agent(
            water={
                "status": "ok",
                "limiting_factor_label": "溶氧",
                "redline_hits": ["DO_REDLINE"],
                "severity_map": {"dissolved_oxygen": 1.0},
                "trends": {"water_temp": {"current": 26.0}},
            },
            vision={"status": "ok", "disease_type": "烂鳃", "is_healthy": False},
            fusion={"risk_level": "高危"},
            max_chunks=8,
        )
        assert ctx.n_chunks <= 8
        assert "投料管理" in ctx.by_purpose

    def test_scores_are_explainable(self, kb: KnowledgeBase) -> None:
        """score 是 RRF 融合分（区间很扁），可解释的相关度要看 dense_score。"""
        res = kb.search(RetrievalQuery(query="溶氧 危险线", category="水质", k=3))
        assert res.chunks
        for c in res.chunks:
            assert 0.0 <= c.score <= 1.0
            assert c.dense_score is None or -1.0 <= c.dense_score <= 1.0

    def test_result_json_serialisable(self, kb: KnowledgeBase) -> None:
        ctx = kb.retrieve_for_agent(
            water={"status": "ok", "severity_map": {"dissolved_oxygen": 1.0}},
            fusion={"risk_level": "预警"},
        )
        assert json.loads(json.dumps(ctx.to_dict(), ensure_ascii=False))

    def test_describe_reports_counts(self, kb: KnowledgeBase) -> None:
        info = kb.describe()
        assert info["n_chunks"] == len(kb)
        assert "水质" in info["by_category"]
        assert "烂鳃" in info["by_disease"]

    def test_open_auto_build_false_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(StoreMismatchError):
            KnowledgeBase.open(persist_dir=tmp_path / "nope", embeddings=HashEmbeddings(64))


# --------------------------------------------------------------------------- #
# 7. 与真实语料的集成（缺失时跳过）
# --------------------------------------------------------------------------- #

REAL_CORPUS = Path(__file__).resolve().parents[2] / "knowledge" / "corpus"


@pytest.mark.skipif(not REAL_CORPUS.is_dir(), reason="真实语料目录不存在")
class TestRealCorpus:
    def test_corpus_loads_and_ids_unique(self) -> None:
        docs = load_corpus(REAL_CORPUS)
        assert len(docs) >= 8
        ids = [d.meta.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_every_doc_has_required_meta(self) -> None:
        for d in load_corpus(REAL_CORPUS):
            assert d.meta.title
            assert d.meta.category in {"水质", "病害", "投料", "应急", "标准"}
            assert d.meta.source

    def test_chunking_is_stable(self) -> None:
        docs = load_corpus(REAL_CORPUS)
        a = [(m.chunk_id, t) for m, t in chunk_corpus(docs)]
        b = [(m.chunk_id, t) for m, t in chunk_corpus(docs)]
        assert a == b

    def test_no_duplicate_chunk_ids(self) -> None:
        pairs = chunk_corpus(load_corpus(REAL_CORPUS))
        ids = [m.chunk_id for m, _ in pairs]
        assert len(ids) == len(set(ids))

    def test_tables_survive_chunking(self) -> None:
        pairs = chunk_corpus(load_corpus(REAL_CORPUS))
        table_chunks = [t for m, t in pairs if m.has_table]
        assert len(table_chunks) >= 20
        # 阈值表的表头与数据行必须在同一块
        joined = "\n".join(table_chunks)
        assert "适宜" in joined and "危险" in joined

    def test_disease_docs_cover_three_targets(self) -> None:
        docs = load_corpus(REAL_CORPUS)
        diseases = {d.meta.disease for d in docs if d.meta.category == "病害"}
        assert {"烂鳃", "出血", "水霉"} <= diseases

    def test_unsafe_drug_warning_present(self) -> None:
        """禁用药物清单必须真的在语料里，否则建议层无法拦住违规用药。"""
        pairs = chunk_corpus(load_corpus(REAL_CORPUS))
        blob = "\n".join(t for _m, t in pairs)
        assert "孔雀石绿" in blob
        assert "禁用" in blob

    def test_every_doc_is_verified_with_traceable_provenance(self) -> None:
        """`verified: true` 必须伴随可追溯的核验记录，否则这个标记毫无意义。"""
        for d in load_corpus(REAL_CORPUS):
            meta = d.meta
            assert meta.verified, f"{d.path.name} 未核验"
            assert meta.standard_codes, f"{d.path.name} 没有登记标准编号"
            assert meta.verified_by, f"{d.path.name} 没有核验人"
            assert meta.verified_on, f"{d.path.name} 没有核验日期"
            assert meta.access_source, f"{d.path.name} 没有访问来源"

    def test_standard_codes_look_like_real_designators(self) -> None:
        """标准编号要带年份、要是现行的那几个，防止写错或写回已废止的编号。"""
        retired = {
            "NY 5051-2001",
            "NY 5052-2001",
            "NY 5071-2002",
            "NY 5072-2002",
            "GB/T 22213-2008",
            "GB 11607-2022",
        }
        seen: set[str] = set()
        for d in load_corpus(REAL_CORPUS):
            for code in d.meta.standard_codes:
                assert "-" in code, f"{d.path.name} 的标准编号 {code!r} 没有年份"
                assert code not in retired, f"{d.path.name} 引用了已废止的 {code}"
                seen.add(code)
        assert "GB 11607-1989" in seen

    def test_retired_standards_are_only_discussed_not_cited(self) -> None:
        """已废止的标准只允许出现在两处：停用记录的登记处，或明确写着「已废止」的说明里。

        光断言"不出现"是不行的——退役决定本身必须写下来，
        否则下一个接手的人只会看到引用凭空消失了。
        """
        retired = ("NY 5051", "NY 5071", "NY 5072", "GB 11607-2022")
        markers = ("废止", "已停用", "不再引用", "已退役", "不存在")
        registry = "08_标准与依据.md"  # 停用记录登记处，允许直接列编号
        for d in load_corpus(REAL_CORPUS):
            if d.path.name == registry:
                continue
            for token in retired:
                start = 0
                while (idx := d.body.find(token, start)) != -1:
                    window = d.body[max(0, idx - 100) : idx + 150]
                    assert any(m in window for m in markers), (
                        f"{d.path.name} 里 {token!r} 出现在非停用说明的语境中：{window!r}"
                    )
                    start = idx + len(token)

    def test_cross_disease_handbook_is_recallable_for_each_disease(self) -> None:
        """跨病种手册 disease=null，但必须能被三种病害的过滤检索召回。"""
        docs = load_corpus(REAL_CORPUS)
        handbook = [d for d in docs if d.meta.doc_id == "disease_treatment_handbook"]
        assert len(handbook) == 1
        assert handbook[0].meta.disease is None
        for label in ("烂鳃", "水霉", "出血"):
            q = RetrievalQuery(query="怎么处置", disease=label)  # type: ignore[arg-type]
            assert build_predicate(q)({"disease": GENERIC_DISEASE}), label


# --------------------------------------------------------------------------- #
# 8. 核验元数据：DocMeta → ChunkMeta → 存储 → 检索 → prompt
# --------------------------------------------------------------------------- #


class TestVerificationMetadata:
    def test_verified_on_accepts_yaml_date_object(self) -> None:
        """YAML 会把不加引号的 2026-10-15 解析成 date，不该因此报校验错误。"""
        import datetime

        meta = DocMeta(
            doc_id="x",
            title="t",
            category="标准",
            verified_on=datetime.date(2026, 10, 15),  # type: ignore[arg-type]
        )
        assert meta.verified_on == "2026-10-15"

    def test_verified_on_passes_strings_through(self) -> None:
        meta = DocMeta(doc_id="x", title="t", category="标准", verified_on="2026-01-02")
        assert meta.verified_on == "2026-01-02"

    def test_verification_fields_default_to_empty(self) -> None:
        """不填就不该有值——默认值不能凭空造出一个"已核验"的假象。"""
        meta = DocMeta(doc_id="x", title="t", category="标准")
        assert meta.standard_codes == []
        assert meta.access_source == ""
        assert meta.verified_by == ""
        assert meta.verified_on == ""

    def test_unknown_front_matter_key_still_rejected(self) -> None:
        """`extra="forbid"` 不能因为加了新字段而被放宽。"""
        with pytest.raises(Exception):
            DocMeta(doc_id="x", title="t", category="标准", not_a_field=1)  # type: ignore[call-arg]

    def test_metadata_reaches_chunk_and_store(self, corpus_dir: Path) -> None:
        src = corpus_dir / "01_阈值.md"
        src.write_text(
            DOC_A.replace(
                "verified: false",
                "verified: true\n"
                "standard_codes: [GB 11607-1989]\n"
                "access_source: openstd\n"
                "verified_by: 张三\n"
                "verified_on: 2026-10-15",
            ),
            encoding="utf-8",
        )
        doc = load_document(src)
        assert doc.meta.standard_codes == ["GB 11607-1989"]
        assert doc.meta.verified_on == "2026-10-15"

        raw_chunks = chunk_document(doc)
        assert raw_chunks
        meta = build_chunk_meta(doc, raw_chunks[0], 0)
        assert meta.standard_codes == ["GB 11607-1989"]
        assert meta.verified_by == "张三"
        assert meta.verified_on == "2026-10-15"

        flat = chunk_to_metadata(meta)
        assert flat["standard_codes"] == "GB 11607-1989"
        assert flat["verified_on"] == "2026-10-15"
        assert split_list(flat["standard_codes"]) == ["GB 11607-1989"]

    def test_prompt_block_surfaces_standard_and_date(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="溶氧低于 3 必须立即增氧。",
            heading_text="溶氧分级",
            title="水质阈值",
            category="水质",
            verified=True,
            standard_codes=["GB 11607-1989"],
            verified_on="2026-10-15",
        )
        block = chunk.to_prompt_block()
        assert "已核对原文" in block
        assert "依据 GB 11607-1989" in block
        assert "核验日期 2026-10-15" in block

    def test_prompt_block_omits_provenance_when_absent(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="正文",
            heading_text="小节",
            title="文档",
            category="水质",
        )
        block = chunk.to_prompt_block()
        assert "未与标准原文核对" in block
        assert "依据" not in block

    def test_prompt_block_hides_date_when_unverified(self) -> None:
        """未核验的块不该展示核验日期——那会读成"已核验"。"""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="正文",
            heading_text="小节",
            title="文档",
            category="水质",
            verified=False,
            standard_codes=["GB 11607-1989"],
            verified_on="2026-10-15",
        )
        block = chunk.to_prompt_block()
        assert "依据 GB 11607-1989" in block
        assert "核验日期" not in block
