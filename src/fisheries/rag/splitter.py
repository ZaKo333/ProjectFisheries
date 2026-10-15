"""文档分块：按标题层级切分，**表格整体保留不切断**。

分块是 RAG 里最容易被轻视、却最影响效果的一步。本项目的分块策略基于
一个很具体的约束：

    **阈值表格被切断 = 行的数值失去列的语义。**

例如下面这张表，如果从中间劈开成两个 chunk：

    | 实测溶氧 | 风险等级 | 立即动作          |
    | 3 – 5    | 预警     | 开启增氧机        |
    | ---------|---------|------------------|   ← 切在这里
    | < 3      | 高危     | 全部增氧机满负荷  |

后半块送给大模型，它就只看到"| < 3 | 高危 | 全部增氧机满负荷 |"，
**不知道这三列分别是什么**，于是可能把"高危"当成溶氧值。
这类错误在告警场景下是致命的。

因此本模块的规则是：

1. **以二/三级标题为天然边界**切分（语义完整）；
2. 一个标题下有多个自然段时，按段落边界合并到目标长度；
3. **表格块是不可分割的原子单位**——宁可让这个 chunk 超长，
   也不切表格；
3b. **连续的列表（有序/无序）同样是原子单位**——原因见下；
4. 超长原子块只记录 warning，不做静默截断
   （静默截断 = 悄悄丢知识）；
5. 过短的块**只在标题链相同时**才并入前一块——标题链不同就说明
   它讲的是另一个小节，合并会造成"张冠李戴"的元数据错误。

**为什么列表也必须原子：** 语料里的处置方案都是"第一步 / 第二步 / …"
编号步骤，而且标题明写"按顺序，不要跳步"。实测未加本规则时，
「1.2 立即动作（按顺序，不要跳步）」的 9 条步骤被按长度切成两块，
检索只召回其中一块时，大模型看到的是**从第 5 步开始的残缺流程**——
它会照着"第 5 步"往下讲，却少了前面的"立即全部开机"。
对应急流程来说，缺步骤比不召回更危险。

**为什么不用 LangChain 的 ``RecursiveCharacterTextSplitter``：**
它的分隔符优先级是 ``\\n\\n → \\n → 空格``，**在表格内部也会按 ``\\n`` 切**，
而且它不认识 Markdown 标题，切出来的块经常没有标题上下文。
``MarkdownHeaderTextSplitter`` 认识标题，但它的二级切分仍然会拆表格。
本项目需要的正是"认识标题 **且** 保护表格"，所以自己实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import CITATION_PATTERN, LoadedDoc, stable_hash
from .schemas import ChunkMeta

#: 目标块长度（字符）。中文 1 字约等于 1–1.5 token，
#: 500 字大约是 600–750 token，配合 top-k=4 检索，
#: 注入 prompt 的知识部分约 2500–3000 token，对 7B 模型是可控的。
DEFAULT_MAX_CHARS = 500

#: 相邻块之间的重叠字符数。只在**按段落切分**时生效，
#: 作用是避免答案正好落在切缝上。表格不参与重叠（会破坏表格）。
DEFAULT_OVERLAP_CHARS = 80

#: 小于该长度的块会被合并进相邻块——太短的块（如只有一行的一级标题）
#: 单独入库只会浪费一个检索位。
MIN_CHUNK_CHARS = 60

#: **同一标题下的内容只要不超过这个长度，就整体作为一块，不再按 max_chars 切。**
#:
#: 为什么需要这个上限之外的第二条阈值：实测「1.2 立即动作（按顺序，不要跳步）」
#: 整节只有 549 字，却因为略微超过 max_chars=500 被切成了 407 + 142 两块。
#: 两块共用一个标题，而检索端的去重（``_diversify``，按 ``heading_text`` 去重）
#: 只会保留其中一块——于是"第 1 分钟做什么"被保留，"第 10 分钟做什么"被丢掉，
#: 大模型拿到的是半个流程。**应急步骤缺一半，比检索不到更危险。**
#:
#: 允许多出这 300 字，换来的是"一个标题 = 一块"这个更硬的性质：
#: 标题本身就是语义边界，比字符数更该被尊重。
#: 超过这个长度的章节才退回按 max_chars 切。
SECTION_COHESION_CHARS = 800

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
#: Markdown 列表项：`- xx` / `* xx` / `+ xx` / `1. xx` / `1、xx` / `1) xx`。
#: 数字后**必须紧跟** `.、)）`，所以"3 – 5 mg/L"这类区间不会被误判成列表项。
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}\s*[.、)）])\s+\S")
_CITATION_RE = re.compile(CITATION_PATTERN)

#: 块类型。table 与 list 都是**不可分割**的原子块。
_KIND_PARA = "para"
_KIND_TABLE = "table"
_KIND_LIST = "list"
_ATOMIC_KINDS = (_KIND_TABLE, _KIND_LIST)


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line))

#: 元章节标题的**开头特征**（剥离编号之后）。
#:
#: 判定方式从"标题链里出现某个词"改成"标题的主语就是本项目/本 Agent"。
#: 旧的子串匹配制造了一个代价很高的误伤：
#: ``## 4. 投喂量必须下调的四种情况（本项目 Agent 的决策依据）``
#: 只是括号里解释了一句"这是给 Agent 用的"，整节（含 4.1–4.4 共 5 块）
#: 就被判成元章节、从检索里排除了——**而"溶氧偏低时减料 30%–50%""氨氮超标停料"
#: 恰恰是 Agent 最需要的决策依据**。元信息过滤不能靠"出现过哪个词"，
#: 要看一个标题到底在讲什么。
META_TITLE_PREFIXES: tuple[str, ...] = (
    "与本项目",
    "本项目的",
    "本项目与",
    "与 Agent 的联动",
    "本项目 Agent 的联动",
)

#: 去掉标题开头的编号：`7. ` / `5.1 ` / `4.3、` / `2) ` 等。
_HEADING_NUMBER_RE = re.compile(r"^\s*\d{1,3}(?:\.\d{1,3})*\s*[.、)）]?\s*")


def _is_meta_title(title: str) -> bool:
    """单个标题是否在讲"本项目/本 Agent"而不是在讲养鱼。"""
    stripped = _HEADING_NUMBER_RE.sub("", title).strip()
    return stripped.startswith(META_TITLE_PREFIXES)


def is_meta_section(heading_path: list[str]) -> bool:
    """判断一个块的标题链是否落在元章节里。**只看标题，不看正文。**

    标题链中**任意一级**是元章节，整块就算元章节（子节 5.1 自然继承父节 5）。
    但判定是在每一级标题上分别做的，所以"某领域小节恰好提到本项目"不会被误伤。
    """
    return any(_is_meta_title(part) for part in heading_path)


@dataclass(slots=True)
class RawChunk:
    """切分后的中间产物：标题链 + 正文。尚未定型为 :class:`ChunkMeta`。"""

    heading_path: list[str]
    text: str
    has_table: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Section:
    """标题切分的结果：一个标题及其直属内容（不含子标题的内容）。"""

    level: int
    title: str
    lines: list[str] = field(default_factory=list)


def _split_heading_tree(body: str) -> list[_Section]:
    """把正文按标题切成有序的 section 列表。

    每个 section 只包含**直属**内容：遇到任意标题就开新 section，
    标题链在后续步骤由层级映射重建。
    """
    sections: list[_Section] = []
    current = _Section(level=0, title="")

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        # 表格行不可能匹配标题正则（表格行以 | 开头），无需额外判断。
        if m:
            if current.title or current.lines:
                sections.append(current)
            current = _Section(level=len(m.group(1)), title=m.group(2))
        else:
            current.lines.append(line)

    if current.title or current.lines:
        sections.append(current)

    return sections


def _is_table_block(lines: list[str], start: int) -> int:
    """从 ``start`` 起，连续的表格行有多少行。0 表示该行不是表格。"""
    if start >= len(lines) or not _TABLE_ROW_RE.match(lines[start]):
        return 0
    n = 0
    while start + n < len(lines) and _TABLE_ROW_RE.match(lines[start + n]):
        n += 1
    return n


def _blocks_with_tables(lines: list[str]) -> list[tuple[str, str]]:
    """把行序列切成 ``[(块文本, 块类型), ...]``。

    切块规则：
        * 连续表格行 → 一个 ``table`` 块（**不可再分**）；
        * 一段连续的列表项（含项内缩进续行、以及列表项之间的空行）
          → 一个 ``list`` 块（**不可再分**，理由见模块 docstring）；
        * 其余连续非空行 → 一个 ``para`` 块；
        * 空行 → 段落边界，不产出块。

    列表的判定细节：
        * **列表项之间的空行不算边界**——语料里每个编号步骤之间都空了一行，
          若按空行断开，每条步骤会各成一块，等于没做原子化；
        * **缩进的非列表行算列表项的续行**（Markdown 的常见写法
          "6. 做某事——\\n   原因是……"），不结束列表；
        * **顶格的普通段落**才结束列表块。
    """
    blocks: list[tuple[str, str]] = []
    i = 0
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = "\n".join(buf).strip()
            if text:
                kind = _KIND_LIST if _is_list_item(buf[0]) else _KIND_PARA
                blocks.append((text, kind))
            buf.clear()

    while i < len(lines):
        line = lines[i]

        n = _is_table_block(lines, i)
        if n > 0:
            flush()
            blocks.append(("\n".join(lines[i : i + n]).strip(), _KIND_TABLE))
            i += n
            continue

        if not line.strip():
            # 列表内部的空行：向后看，若下一个非空行仍是列表项，则不算边界。
            if buf and _is_list_item(buf[0]):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and _is_list_item(lines[j]):
                    i += 1
                    continue
            flush()
            i += 1
            continue

        if buf:
            in_list = _is_list_item(buf[0])
            if in_list and not _is_list_item(line) and not line[:1].isspace():
                # 顶格普通段落 → 列表结束
                flush()
            elif not in_list and _is_list_item(line):
                # 段落之后开始的列表自成一块，让编号从第 1 条开始干净
                flush()

        buf.append(line)
        i += 1

    flush()
    return blocks


def _pack_blocks(
    blocks: list[tuple[str, str]],
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, bool, list[str]]]:
    """把块贪心地装进不超过 ``max_chars`` 的分组。

    返回 ``[(合并后的文本, 是否含表格, 警告列表), ...]``。

    关键约束：**表格块与列表块永远不会被切断**。
    如果单个原子块本身就超过 ``max_chars``，它独占一个分组并记警告。
    """
    out: list[tuple[str, bool, list[str]]] = []
    cur: list[str] = []
    cur_len = 0
    cur_has_table = False
    cur_warnings: list[str] = []
    prev_atomic = False

    def flush() -> None:
        nonlocal cur, cur_len, cur_has_table, cur_warnings
        if cur:
            out.append(("\n\n".join(cur), cur_has_table, list(cur_warnings)))
        cur = []
        cur_len = 0
        cur_has_table = False
        cur_warnings = []

    for text, kind in blocks:
        is_atomic = kind in _ATOMIC_KINDS

        if is_atomic and len(text) > max_chars:
            # 不可分割：独占一块，超长只记警告，绝不静默截断。
            flush()
            label = "表格块" if kind == _KIND_TABLE else "列表块"
            out.append(
                (
                    text,
                    kind == _KIND_TABLE,
                    [f"{label} {len(text)} 字超过 max_chars={max_chars}，按不可分割处理"],
                )
            )
            prev_atomic = True
            continue

        if cur and cur_len + len(text) + 2 > max_chars:
            flush()

        # 段落重叠：把上一块的尾巴带上，防止答案正好落在切缝上。
        # 上一块若是表格或列表则不重叠——半张表、半段流程比没有上下文更糟：
        # 重叠出来的"从第 5 步开始"的片段正是我们要消灭的东西。
        if not cur and not is_atomic and overlap_chars > 0 and out and not prev_atomic:
            prev_text, prev_has_table, _ = out[-1]
            if not prev_has_table:
                tail = prev_text[-overlap_chars:].strip()
                if tail:
                    cur.append(tail)
                    cur_len += len(tail)

        cur.append(text)
        cur_len += len(text) + 2
        cur_has_table = cur_has_table or kind == _KIND_TABLE
        prev_atomic = is_atomic

    flush()
    return out


def chunk_document(
    doc: LoadedDoc,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    cohesion_chars: int = SECTION_COHESION_CHARS,
) -> list[RawChunk]:
    """把一个已加载文档切成 :class:`RawChunk` 列表。

    标题链的构造规则：
        * ``#`` 一级标题视为**文档标题**，不进入标题链
          （``meta.title`` 已经承担了这个角色，重复一遍纯属浪费 token）；
        * 从二级标题开始，按层级组成标题链，
          例如 ``## 3. 处置方案`` / ``### 3.1 第一步`` →
          ``['3. 处置方案', '3.1 第一步']``。
    """
    sections = _split_heading_tree(doc.body)
    chunks: list[RawChunk] = []

    # level → 标题文本。用映射而不是定长栈，避免层级跳跃时补出空标题。
    title_map: dict[int, str] = {}

    def emit(secs: list[_Section], heading_path: list[str]) -> None:
        blocks: list[tuple[str, str]] = []
        for sec in secs:
            blocks.extend(_blocks_with_tables(sec.lines))
        if not blocks:
            return

        section_len = sum(len(t) for t, _ in blocks) + 2 * len(blocks)
        if section_len <= cohesion_chars:
            # 整节装得下 → 一块装完，不再按 max_chars 切（见 SECTION_COHESION_CHARS）。
            # 不给重叠：整节就是一块，没有切缝需要掩盖。
            packed = _pack_blocks(blocks, cohesion_chars, overlap_chars=0)
        else:
            packed = _pack_blocks(blocks, max_chars, overlap_chars)

        for text, has_table, warns in packed:
            if (
                len(text) < MIN_CHUNK_CHARS
                and not has_table
                and chunks
                and chunks[-1].heading_path == heading_path
            ):
                # 只在标题链相同时合并，绝不把 A 小节的内容挂到 B 小节名下。
                chunks[-1].text = chunks[-1].text + "\n\n" + text
                chunks[-1].has_table = chunks[-1].has_table or has_table
                chunks[-1].warnings.extend(warns)
                continue
            chunks.append(
                RawChunk(
                    heading_path=list(heading_path),
                    text=text,
                    has_table=has_table,
                    warnings=list(warns),
                )
            )

    pending_intro: list[_Section] = []

    for sec in sections:
        if sec.level == 0:
            # 文档开头、第一个标题之前的引言。
            if any(ln.strip() for ln in sec.lines):
                pending_intro.append(sec)
            continue

        if pending_intro:
            emit(pending_intro, [doc.meta.title])
            pending_intro.clear()

        if sec.level == 1:
            # 一级标题 = 文档标题。它若有直属内容，用文档标题当标题链。
            emit([sec], [doc.meta.title])
            continue

        # 二级及以下：更新层级映射，并清掉比当前更深的层级。
        title_map[sec.level] = sec.title
        for lv in [lv for lv in title_map if lv > sec.level]:
            del title_map[lv]

        heading_path = [title_map[lv] for lv in sorted(title_map) if title_map[lv]]
        emit([sec], heading_path)

    if pending_intro:
        emit(pending_intro, [doc.meta.title])

    return chunks


def build_chunk_meta(doc: LoadedDoc, raw: RawChunk, index: int) -> ChunkMeta:
    """把 :class:`RawChunk` 定型为 :class:`ChunkMeta`。

    ``chunk_id`` 由 ``源文件 + 标题路径 + 序号`` 哈希得到，
    **与正文内容无关**。这样改了正文措辞后 ID 不变，
    向量库可以按 ID 做覆盖更新而不是全量重建。
    """
    heading_text = " > ".join(raw.heading_path) if raw.heading_path else doc.meta.title
    chunk_id = stable_hash(
        doc.meta.source_path or doc.path.name,
        " > ".join(raw.heading_path),
        index,
    )
    citations = sorted(set(_CITATION_RE.findall(raw.text)))

    return ChunkMeta(
        chunk_id=chunk_id,
        doc_id=doc.meta.doc_id,
        title=doc.meta.title,
        category=doc.meta.category,
        disease=doc.meta.disease,
        indicators=list(doc.meta.indicators),
        risk_levels=list(doc.meta.risk_levels),
        verified=doc.meta.verified,
        source=doc.meta.source,
        standard_codes=list(doc.meta.standard_codes),
        access_source=doc.meta.access_source,
        verified_by=doc.meta.verified_by,
        verified_on=doc.meta.verified_on,
        source_path=doc.meta.source_path,
        heading_path=list(raw.heading_path),
        heading_text=heading_text,
        chunk_index=index,
        section_role="meta" if is_meta_section(raw.heading_path) else "domain",
        has_table=raw.has_table,
        citation_keys=citations,
        char_count=len(raw.text),
    )


def build_embedding_text(meta: ChunkMeta, text: str) -> str:
    """构造**用于向量化**的文本（注意：不是送给大模型的文本）。

    前面拼上标题链。原因很实际：chunk 被切出来后常常只有一段正文，
    例如"停止拉网、分池、运输操作，避免造成新的伤口"——
    单看这段，向量模型不知道它属于水霉病处置方案，
    查询"水霉病怎么处理"时相似度上不来。
    拼上标题链后，向量里就带上了主题信息。

    送进 prompt 的仍然是纯正文（见 ``RetrievedChunk.to_prompt_block``），
    避免面包屑在每个块里重复出现、白占上下文。
    """
    breadcrumb = meta.heading_text.strip()
    if not breadcrumb:
        return text
    return f"{breadcrumb}\n{text}"


def chunk_corpus(
    docs: list[LoadedDoc],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    cohesion_chars: int = SECTION_COHESION_CHARS,
) -> list[tuple[ChunkMeta, str]]:
    """把整个语料切块，返回 ``[(ChunkMeta, 正文), ...]``。"""
    out: list[tuple[ChunkMeta, str]] = []
    for doc in docs:
        for i, raw in enumerate(
            chunk_document(
                doc,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                cohesion_chars=cohesion_chars,
            )
        ):
            out.append((build_chunk_meta(doc, raw, i), raw.text))
    return out
