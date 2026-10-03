"""分块策略。

三种策略供 M4 消融实验对比（架构 ADR-010，配置驱动而非代码分支）：

    fixed      定长切分，基线
    recursive  按段落/句子边界递归切分（LangChain 内置）
    heading    按标题层级切分，默认策略

父子块（small-to-big，ADR-005）：
    父块用于补全上下文，子块用于精准检索。
    子块命中后在生成阶段替换为父块，兼顾"找得准"和"看得全"。

表格的特殊处理（ADR-016）：
    表格默认整块不切（table_atomic）。超长表格按行切分时，
    每一片都要重新带上量纲声明与表头——否则切出来的片段里
    数字没有列名也没有单位，等于废片。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.experiment import ChunkingCfg
from app.rag.pdf_parser import ParsedBlock, ParsedDocument

logger = logging.getLogger(__name__)

# 父块上限。超过后强制断开，避免单个块撑爆生成阶段的上下文预算。
PARENT_MAX_CHARS = 3000


@dataclass
class ChunkUnit:
    """与 PG chunks 表及 Milvus fin_chunks 一一对应的最小存储单元。"""

    chunk_uid: str
    content: str
    level: int  # 0=父块 1=子块
    chunk_type: str  # text | table
    heading_path: str
    page_start: int
    page_end: int
    strategy: str
    parent_uid: str | None = None
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    statement_type: str | None = None
    table_flags: list[str] = field(default_factory=list)
    token_count: int = 0


def make_uid(doc_key: str, strategy: str, level: int, seq: int) -> str:
    """生成稳定可复现的块 ID。

    同一文档同一策略重新解析必须得到相同 ID，否则增量更新无从比对。
    """
    raw = f"{doc_key}|{strategy}|{level}|{seq}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{doc_key}-{strategy}-L{level}-{seq:05d}-{digest}"


def est_tokens(text: str) -> int:
    """估算 token 数。

    中文大致一字一 token，英文约四字符一 token。
    这里只用于控制块大小，不需要精确——引入 tiktoken 会让
    分块逻辑依赖具体模型的分词器，反而不利于跨 Provider 切换。
    """
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk + (len(text) - cjk) // 3


def with_heading(text: str, heading: str) -> str:
    """给块补上标题路径前缀。

    检索返回的是孤立的块，没有标题路径就不知道这段话属于哪一节。
    """
    if not heading or text.startswith(heading):
        return text
    return f"{heading}\n{text}"


# ── 表格切分 ────────────────────────────────────────────


def split_table(block: ParsedBlock, max_chars: int) -> list[str]:
    """超长表格按行切分，每片保留量纲声明与表头。

    直接按字符截断会切掉表头和单位，剩下一堆没有列名的数字，
    这种片段检索到了也无法回答问题。
    """
    lines = block.text.splitlines()
    if len(block.text) <= max_chars or len(lines) < 4:
        return [block.text]

    # 表头部分：量纲行、表题行、Markdown 表头与分隔行
    sep_idx = next((i for i, ln in enumerate(lines) if set(ln.strip()) <= set("|- ")), -1)
    if sep_idx < 1:
        return [block.text]
    header = lines[: sep_idx + 1]
    body = lines[sep_idx + 1 :]
    header_text = "\n".join(header)

    parts: list[str] = []
    buf: list[str] = []
    budget = max(max_chars - len(header_text), max_chars // 2)
    for row in body:
        if buf and sum(len(x) + 1 for x in buf) + len(row) > budget:
            parts.append(header_text + "\n" + "\n".join(buf))
            buf = []
        buf.append(row)
    if buf:
        parts.append(header_text + "\n" + "\n".join(buf))
    return parts


# ── 文本切分 ────────────────────────────────────────────


def split_text(text: str, cfg: ChunkingCfg) -> list[str]:
    """按配置的策略切分正文。"""
    if cfg.strategy == "fixed":
        size, overlap = cfg.chunk_size, cfg.overlap
        if len(text) <= size:
            return [text]
        step = max(size - overlap, 1)
        return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]

    # recursive 与 heading 策略的节内切分都用递归切分器：
    # 它按段落→句子→词的顺序回退，尽量不在句中断开。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=True,
    )
    return [s for s in splitter.split_text(text) if s.strip()]


# ── 父块分组 ────────────────────────────────────────────


def group_parents(blocks: list[ParsedBlock], cfg: ChunkingCfg) -> list[list[ParsedBlock]]:
    """把解析块聚合为父块。

    heading 策略：同一标题路径下的连续文本合并为一个父块；
                  表格自成一块，不与正文混合——表格与叙述文字混在一起，
                  切分时很容易把表头和数据分开。
    其他策略：    文本按长度累积，表格仍然独立。
    """
    groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_heading: str | None = None
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len, current_heading
        if current:
            groups.append(current)
        current, current_len, current_heading = [], 0, None

    for blk in blocks:
        if blk.kind == "table":
            flush()
            groups.append([blk])
            continue

        heading_changed = cfg.strategy == "heading" and current_heading not in (None, blk.heading_path)
        if heading_changed or current_len + len(blk.text) > PARENT_MAX_CHARS:
            flush()

        if not current:
            current_heading = blk.heading_path
        current.append(blk)
        current_len += len(blk.text)

    flush()
    return groups


# ── 主流程 ──────────────────────────────────────────────


def chunk_document(doc: ParsedDocument, doc_key: str, cfg: ChunkingCfg) -> list[ChunkUnit]:
    """把解析结果切分为可入库的块。

    parent_child 关闭时只产出一层（level=1），用于基线实验对比。
    """
    units: list[ChunkUnit] = []
    parent_seq = child_seq = 0

    for group in group_parents(doc.blocks, cfg):
        head = group[0]
        is_table = head.kind == "table"
        heading = head.heading_path
        text = "\n".join(b.text for b in group)
        page_start = min(b.page_start for b in group)
        page_end = max(b.page_end for b in group)

        parent_uid: str | None = None
        if cfg.parent_child:
            parent_uid = make_uid(doc_key, cfg.strategy, 0, parent_seq)
            units.append(
                ChunkUnit(
                    chunk_uid=parent_uid,
                    content=with_heading(text, heading),
                    level=0,
                    chunk_type=head.kind,
                    heading_path=heading,
                    page_start=page_start,
                    page_end=page_end,
                    strategy=cfg.strategy,
                    unit=head.unit,
                    currency=head.currency,
                    statement_type=head.statement_type,
                    table_flags=list(head.flags),
                    token_count=est_tokens(text),
                )
            )
            parent_seq += 1

        # 子块
        if is_table and cfg.table_atomic:
            pieces = split_table(head, max_chars=cfg.chunk_size * 4)
        elif is_table:
            pieces = split_text(text, cfg)
        else:
            pieces = split_text(text, cfg)

        for piece in pieces:
            body = with_heading(piece, heading)
            units.append(
                ChunkUnit(
                    chunk_uid=make_uid(doc_key, cfg.strategy, 1, child_seq),
                    content=body,
                    level=1,
                    chunk_type=head.kind,
                    heading_path=heading,
                    page_start=page_start,
                    page_end=page_end,
                    strategy=cfg.strategy,
                    parent_uid=parent_uid,
                    unit=head.unit,
                    currency=head.currency,
                    statement_type=head.statement_type,
                    table_flags=list(head.flags),
                    token_count=est_tokens(body),
                )
            )
            child_seq += 1

    logger.info(
        "分块 %s（策略=%s）：父块 %d，子块 %d",
        doc_key,
        cfg.strategy,
        sum(1 for u in units if u.level == 0),
        sum(1 for u in units if u.level == 1),
    )
    return units
