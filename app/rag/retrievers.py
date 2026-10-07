"""检索链：混合检索 → 重排 → 父块还原。

对应架构文档 §6.2 步骤 5–7。几条关键约定：

  1. **标量过滤必须下推到检索层**。实测 BM25 对年份完全无感
     （2023 与 2024 两条得分相同），公司与报告期只能靠过滤条件卡住，
     指望检索器自己区分是不行的。
  2. **strategy 必须参与过滤**。不同实验的块共存于同一集合，
     不隔离就会把 fixed 策略的块混进 heading 策略的结果里，
     消融实验的对比直接失效（ADR-010）。
  3. **父块还原要去重**。多个子块常同属一个父块，不按 parent_uid
     去重会让同一段内容重复进上下文，成本翻倍而信息量不变。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker
from sqlalchemy import select

from app.config.experiment import Experiment
from app.providers.registry import get_embeddings
from app.store.milvus_schema import collection_names
from app.store.models import Chunk
from app.store.pg import session_factory

logger = logging.getLogger(__name__)

OUTPUT_FIELDS = [
    "chunk_uid",
    "content",
    "doc_key",
    "company_code",
    "report_year",
    "period",
    "chunk_type",
    "statement_type",
    "unit",
    "heading_path",
    "page_start",
    "parent_uid",
]


@dataclass
class QueryFilters:
    """从问题中抽取的结构化过滤条件。"""

    company_codes: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    statement_type: str | None = None
    chunk_type: str | None = None

    def is_empty(self) -> bool:
        return not (self.company_codes or self.years or self.statement_type or self.chunk_type)


@dataclass
class RetrievedChunk:
    chunk_uid: str
    content: str
    score: float
    company_code: str = ""
    report_year: int = 0
    unit: str | None = None
    chunk_type: str = "text"
    heading_path: str = ""
    page_start: int = 0
    page_end: int = 0
    parent_uid: str | None = None
    doc_key: str = ""
    expanded: bool = False  # 是否已被替换为父块

    @classmethod
    def from_hit(cls, hit: dict) -> RetrievedChunk:
        e = hit.get("entity", hit)
        return cls(
            chunk_uid=e.get("chunk_uid", ""),
            content=e.get("content", ""),
            score=float(hit.get("distance", 0.0)),
            company_code=e.get("company_code", ""),
            report_year=int(e.get("report_year", 0) or 0),
            unit=e.get("unit") or None,
            chunk_type=e.get("chunk_type", "text"),
            heading_path=e.get("heading_path", ""),
            page_start=int(e.get("page_start", 0) or 0),
            page_end=int(e.get("page_start", 0) or 0),
            parent_uid=e.get("parent_uid") or None,
            doc_key=e.get("doc_key", ""),
        )


# 年报通常披露最近三年数据，因此 2024 年的数字可能出现在
# 2024、2025 或 2026 年的年报中。
REPORT_COVERS_YEARS = 3


def expand_report_years(years: list[int]) -> list[int]:
    """把「数据年度」映射为可能承载它的「报告年度」。

    这是个极易踩的坑：report_year 是**文档的报告年度**，
    而用户问的是**数据的年度**，两者不是一回事。
    一份 2025 年报含 2023–2025 三年数据，但 report_year 只有 2025，
    用 `report_year == 2024` 过滤会直接返回空集。

    实测该缺陷导致 61 题中约四分之一被误判为「检索无结果」而拒答，
    且因为在检索阶段就空了，相关度阈值根本没被触及——
    调阈值完全无效，排查时极易被误导。
    """
    out: set[int] = set()
    for y in years:
        out.update(range(y, y + REPORT_COVERS_YEARS))
    return sorted(out)


def build_expr(filters: QueryFilters, strategy: str) -> str:
    """构造 Milvus 标量过滤表达式。

    strategy 始终参与，保证实验之间互不污染。
    """
    parts = [f'strategy == "{strategy}"']
    if filters.company_codes:
        codes = ", ".join(f'"{c}"' for c in filters.company_codes)
        parts.append(f"company_code in [{codes}]")
    if filters.years:
        years = ", ".join(str(y) for y in expand_report_years(filters.years))
        parts.append(f"report_year in [{years}]")
    if filters.statement_type:
        parts.append(f'statement_type == "{filters.statement_type}"')
    if filters.chunk_type:
        parts.append(f'chunk_type == "{filters.chunk_type}"')
    return " and ".join(parts)


SUMMARY_CHUNK_TYPE = "table_summary"


async def force_summary_table(client, filters: QueryFilters, exp, limit: int = 2):
    """按块类型直接点名召回「近三年主要会计数据」表。

    取代原先往关键词查询里拼表名的做法。那个做法有两个毛病：
      1. 表名各家写法不同（上交所「近三年主要会计数据和财务指标」、
         深交所「主要会计数据和财务指标」），写死一种只覆盖一半公司——
         实测海康威视因此一直召回不到（`lessons.md` 7.13）
      2. 本质上还是在和词频赛跑，而这张表每个科目只出现一次，天生吃亏

    按标签取就没有这些问题：解析阶段已经确定哪张表是它了。
    """
    expr_parts = [f'chunk_type == "{SUMMARY_CHUNK_TYPE}"', f'strategy == "{exp.chunking.strategy}"']
    if filters.company_codes:
        codes = ", ".join(f'"{c}"' for c in filters.company_codes)
        expr_parts.append(f"company_code in [{codes}]")
    try:
        rows = await asyncio.to_thread(
            client.query,
            collection_name=collection_names()[0],
            filter=" and ".join(expr_parts),
            output_fields=OUTPUT_FIELDS,
            limit=limit,
        )
    except Exception as exc:
        logger.warning("汇总表强制召回失败：%s", exc)
        return []
    # query 返回裸字段、没有 distance；给个固定分值让它能与检索结果合流。
    # 取 0.02 是刻意的：高于相关度阈值（0.010）因而不会把整轮判成拒答，
    # 又低于真正命中的 RRF 分值，排序时不挤掉本来更相关的块。
    return [RetrievedChunk.from_hit({"entity": r, "distance": 0.02}) for r in rows]


async def retrieve(
    client: MilvusClient,
    query: str,
    filters: QueryFilters,
    exp: Experiment,
    keyword_query: str | None = None,
) -> list[RetrievedChunk]:
    """按实验配置执行检索。

    mode=dense  只用向量，语义相近但可能漏掉精确的专有名词
    mode=sparse 只用 BM25，精确但不懂语义
    mode=hybrid 两者 RRF 融合
    """
    cfg = exp.retrieval
    name, _ = collection_names()
    expr = build_expr(filters, exp.chunking.strategy)
    kw = keyword_query or query

    if cfg.mode in ("dense", "hybrid"):
        vector = await get_embeddings().aembed_query(query)
    else:
        vector = None

    def _search() -> list[dict]:
        if cfg.mode == "dense":
            res = client.search(
                name,
                data=[vector],
                anns_field="dense_vector",
                limit=cfg.top_k_dense,
                filter=expr,
                output_fields=OUTPUT_FIELDS,
                search_params={"params": {"ef": 64}},
            )
            return list(res[0])
        if cfg.mode == "sparse":
            res = client.search(
                name,
                data=[kw],
                anns_field="sparse_vector",
                limit=cfg.top_k_sparse or cfg.top_k_dense,
                filter=expr,
                output_fields=OUTPUT_FIELDS,
            )
            return list(res[0])

        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="dense_vector",
            param={"ef": 64},
            limit=cfg.top_k_dense,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[kw],
            anns_field="sparse_vector",
            param={},
            limit=cfg.top_k_sparse,
            expr=expr,
        )
        res = client.hybrid_search(
            name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=max(cfg.top_k_dense, cfg.top_k_sparse),
            output_fields=OUTPUT_FIELDS,
        )
        return list(res[0])

    hits = await asyncio.to_thread(_search)
    chunks = [RetrievedChunk.from_hit(h) for h in hits]
    logger.debug("检索 mode=%s 命中 %d 条，filter=%s", cfg.mode, len(chunks), expr)
    return chunks


async def rerank(
    chunks: list[RetrievedChunk], query: str, exp: Experiment
) -> tuple[list[RetrievedChunk], list[str]]:
    """重排，返回 (结果, 降级标记)。

    关闭时按原序截断到 top_n——这样开关重排只改变排序方式，
    不改变送入生成的块数，消融实验才是在比较排序质量本身。

    服务不可用时保留原序继续，并标记 degraded：重排是质量增强
    而非必需环节，为它牺牲可用性不划算（架构 §10.1）。
    """
    top_n = exp.rerank.top_n or len(chunks)
    if not exp.rerank.enabled or not chunks:
        return chunks[:top_n], []

    from app.providers.reranker import RerankUnavailable, rerank_documents

    try:
        ranked = await rerank_documents(
            query, [c.content for c in chunks], top_n=top_n, model=exp.rerank.model
        )
    except RerankUnavailable as exc:
        logger.warning("重排不可用，保留原序：%s", exc)
        return chunks[:top_n], ["rerank_skipped"]

    out: list[RetrievedChunk] = []
    for idx, score in ranked:
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            # 用重排分数覆盖检索分数：后续相关度判定应基于重排结果，
            # 否则阈值比较的仍是 RRF 分值，与实际排序脱节
            c.score = score
            out.append(c)
    logger.debug("重排 %d → %d 条，最高分 %.4f", len(chunks), len(out), out[0].score if out else 0)
    return out, []


async def expand_parents(
    chunks: list[RetrievedChunk], exp: Experiment
) -> tuple[list[RetrievedChunk], list[str]]:
    """父块还原（small-to-big）。

    返回 (结果, 降级标记)。关闭父子块时原样返回。
    """
    cfg = exp.parent_expansion
    if not cfg.enabled or not chunks:
        return chunks, []

    parent_uids = [c.parent_uid for c in chunks if c.parent_uid]
    if not parent_uids:
        return chunks, []

    async with session_factory()() as session:
        rows = (
            await session.execute(select(Chunk).where(Chunk.chunk_uid.in_(set(parent_uids))))
        ).scalars().all()
    parents = {r.chunk_uid: r for r in rows}

    result: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    budget = cfg.token_budget
    used = 0
    dropped = 0

    for c in chunks:
        pid = c.parent_uid
        if cfg.dedup_by_parent and pid and pid in seen_parents:
            # 同一父块只保留一次：重复进上下文不增加信息，只增加成本
            continue

        parent = parents.get(pid) if pid else None
        if parent is None:
            item, cost = c, len(c.content)
        else:
            seen_parents.add(pid)
            item = RetrievedChunk(
                chunk_uid=parent.chunk_uid,
                content=parent.content,
                score=c.score,
                company_code=c.company_code,
                report_year=c.report_year,
                unit=parent.unit or c.unit,
                chunk_type=parent.chunk_type,
                heading_path=parent.heading_path or c.heading_path,
                page_start=parent.page_start or c.page_start,
                page_end=parent.page_end or parent.page_start or c.page_end,
                parent_uid=None,
                doc_key=c.doc_key,
                expanded=True,
            )
            cost = len(parent.content)

        # 表格父块不中途截断：切一半的表格没有使用价值
        if used + cost > budget and result:
            dropped += 1
            continue
        result.append(item)
        used += cost

    degraded = [f"context_truncated:{dropped}"] if dropped else []
    logger.debug(
        "父块还原：输入 %d → 输出 %d（去重后），字符预算 %d/%d，丢弃 %d",
        len(chunks),
        len(result),
        used,
        budget,
        dropped,
    )
    return result, degraded


def max_score(chunks: list[RetrievedChunk]) -> float:
    return max((c.score for c in chunks), default=0.0)
