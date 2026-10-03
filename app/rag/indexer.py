"""入库流水线：解析 → 分块 → 向量化 → 双写。

对应架构文档 §6.1。几条不可动摇的约定：

  1. **只有子块进 Milvus**。父块仅用于检索命中后补全上下文，不需要向量，
     按 parent_uid 从 PG 取回即可。实测父块最大 10636 字节，
     超过 Milvus VARCHAR 的 8192 字节上限（注意是字节不是字符，
     中文 UTF-8 占 3 字节），若强行写入会直接插入失败。
  2. **先写 PG 再写 Milvus**。失败时以 PG 为准做补偿重建，
     反过来会留下无法回溯的孤立向量。
  3. **幂等**：同一文件内容重复上传不重复消耗 embedding。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from sqlalchemy import delete, select

from app.config.experiment import Experiment, load_experiment
from app.providers.registry import get_embeddings, get_registry
from app.rag.chunker import ChunkUnit, chunk_document
from app.rag.pdf_parser import parse_pdf
from app.store.milvus_schema import MAX_CONTENT_LEN, collection_names
from app.store.models import Chunk, Document
from app.store.pg import session_factory

logger = logging.getLogger(__name__)

EMBED_MAX_RETRIES = 3
# 文件名形如 600519_贵州茅台_xxx2025年年度报告.pdf
FILENAME_PAT = re.compile(r"^(\d{6})_([^_]+)_.*?(\d{4})\s*年")


@dataclass
class IngestResult:
    doc_id: int
    doc_key: str
    status: str
    chunk_count: int = 0
    child_count: int = 0
    embed_cost: float = 0.0
    skipped: bool = False
    error: str | None = None


# ── 元信息与幂等 ────────────────────────────────────────


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def infer_meta(path: Path) -> tuple[str, str, int]:
    """从文件名推断 (股票代码, 公司名, 年度)。

    下载脚本统一了命名，推断失败说明文件来路不明，应当显式报错
    而不是猜一个默认值——错误的 doc_key 会污染整个检索过滤。
    """
    m = FILENAME_PAT.match(path.name)
    if not m:
        raise ValueError(f"无法从文件名推断公司与年度：{path.name}（期望 代码_公司名_...YYYY年）")
    return m.group(1), m.group(2), int(m.group(3))


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """按字节截断。Milvus 的 VARCHAR 长度限制以字节计。"""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    return raw[:max_bytes].decode("utf-8", errors="ignore"), True


# ── 向量化 ──────────────────────────────────────────────


async def embed_texts(texts: list[str], batch_size: int) -> tuple[list[list[float]], int]:
    """批量向量化，返回 (向量列表, 消耗的字符数)。

    失败时指数退避重试；整批失败才放弃，避免单条异常拖垮整篇文档。
    """
    embedder = get_embeddings()
    vectors: list[list[float]] = []
    char_count = sum(len(t) for t in texts)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                vectors.extend(await embedder.aembed_documents(batch))
                break
            except Exception as exc:
                if attempt == EMBED_MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "embedding 第 %d/%d 批失败（第 %d 次重试，%ds 后）：%s",
                    start // batch_size + 1,
                    (len(texts) + batch_size - 1) // batch_size,
                    attempt + 1,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)
    return vectors, char_count


def embed_cost(char_count: int) -> float:
    """embedding 成本。中文约一字一 token，用字符数近似。"""
    spec = get_registry().embed_spec()
    return round(char_count / 1000 * spec.price_in, 6)


# ── 写入 ────────────────────────────────────────────────


async def _save_chunks_pg(session, doc_id: int, units: list[ChunkUnit]) -> None:
    await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
    session.add_all(
        [
            Chunk(
                chunk_uid=u.chunk_uid,
                doc_id=doc_id,
                parent_uid=u.parent_uid,
                level=u.level,
                chunk_type=u.chunk_type,
                heading_path=u.heading_path or None,
                page_start=u.page_start,
                page_end=u.page_end,
                token_count=u.token_count,
                content=u.content,
                strategy=u.strategy,
                unit=u.unit,
                currency=u.currency,
                period=u.period,
                statement_type=u.statement_type,
                table_flags=u.table_flags or None,
            )
            for u in units
        ]
    )


def _write_milvus(client, rows: list[dict], doc_key: str, strategy: str) -> None:
    """重写该文档该策略下的全部子块向量。

    先删后插而非 upsert：块数会随策略变化，残留的旧块会污染检索结果。
    """
    name, _ = collection_names()
    client.delete(name, filter=f'doc_key == "{doc_key}" and strategy == "{strategy}"')
    if rows:
        client.insert(name, rows)
    client.flush(name)


# ── 主流程 ──────────────────────────────────────────────


async def ingest_pdf(
    path: str | Path,
    milvus_client,
    exp: Experiment | None = None,
    max_pages: int | None = None,
    force: bool = False,
) -> IngestResult:
    path = Path(path)
    exp = exp or load_experiment()
    cfg = exp.chunking

    code, name, year = infer_meta(path)
    doc_key = f"{code}_{year}_annual"
    digest = file_hash(path)

    factory = session_factory()

    # ── 幂等检查 ──
    async with factory() as session:
        existing = (
            await session.execute(select(Document).where(Document.doc_key == doc_key))
        ).scalar_one_or_none()

        if existing and existing.content_hash == digest and existing.status == "ready" and not force:
            logger.info("%s 内容未变且已就绪，跳过（force=True 可强制重建）", doc_key)
            return IngestResult(
                doc_id=existing.id,
                doc_key=doc_key,
                status="ready",
                chunk_count=existing.chunk_count or 0,
                skipped=True,
            )

        if existing:
            doc = existing
            doc.content_hash = digest
            doc.file_path = str(path)
        else:
            doc = Document(
                doc_key=doc_key,
                company_code=code,
                company_name=name,
                report_year=year,
                report_type="annual",
                file_path=str(path),
                content_hash=digest,
            )
            session.add(doc)
        doc.status = "parsing"
        doc.error_code = doc.error_msg = None
        await session.commit()
        doc_id = doc.id

    async def set_status(status: str, **fields) -> None:
        async with factory() as session:
            d = await session.get(Document, doc_id)
            d.status = status
            for k, v in fields.items():
                setattr(d, k, v)
            await session.commit()

    try:
        # ── 解析 ──
        parsed = await asyncio.to_thread(parse_pdf, path, max_pages)
        await set_status("chunking", page_count=parsed.page_count)

        # ── 分块 ──
        units = chunk_document(parsed, doc_key, cfg)
        # 报告期是文档级属性，统一在此赋值，分块阶段无从得知
        for u in units:
            u.period = u.period or f"FY{year}"
        children = [u for u in units if u.level == 1]
        if not children:
            raise ValueError("分块结果为空，疑似解析失败")
        await set_status("embedding")

        # ── 向量化：只对子块 ──
        texts, truncated = [], 0
        for u in children:
            text, cut = truncate_utf8(u.content, MAX_CONTENT_LEN)
            if cut:
                truncated += 1
                u.content = text
            texts.append(text)
        if truncated:
            logger.warning("%s 有 %d 个子块超出 Milvus 字段上限，已截断", doc_key, truncated)

        vectors, char_count = await embed_texts(texts, cfg_batch(exp))
        cost = embed_cost(char_count)

        # ── 双写：先 PG 后 Milvus ──
        async with factory() as session:
            await _save_chunks_pg(session, doc_id, units)
            await session.commit()

        rows = [
            {
                "chunk_uid": u.chunk_uid,
                "dense_vector": vec,
                "content": u.content,
                "doc_key": doc_key,
                "company_code": code,
                "report_year": year,
                "period": u.period or f"FY{year}",
                "chunk_type": u.chunk_type,
                "statement_type": u.statement_type or "",
                "strategy": u.strategy,
                "unit": u.unit or "",
                "heading_path": (u.heading_path or "")[:500],
                "page_start": u.page_start,
                "parent_uid": u.parent_uid or "",
            }
            for u, vec in zip(children, vectors, strict=True)
        ]
        await asyncio.to_thread(_write_milvus, milvus_client, rows, doc_key, cfg.strategy)

        await set_status("ready", chunk_count=len(units), index_cost=Decimal(str(cost)))
        logger.info(
            "%s 入库完成：%d 块（子块 %d），embedding 成本 ¥%.6f",
            doc_key,
            len(units),
            len(children),
            cost,
        )
        return IngestResult(
            doc_id=doc_id,
            doc_key=doc_key,
            status="ready",
            chunk_count=len(units),
            child_count=len(children),
            embed_cost=cost,
        )

    except Exception as exc:
        logger.exception("%s 入库失败", doc_key)
        await set_status("failed", error_code="FC-1001", error_msg=f"{type(exc).__name__}: {exc}")
        return IngestResult(
            doc_id=doc_id, doc_key=doc_key, status="failed", error=f"{type(exc).__name__}: {exc}"
        )


def cfg_batch(exp: Experiment) -> int:
    from app.config.settings import get_settings

    return int(get_settings().yaml_get("ingestion.embed_batch_size", 64))
