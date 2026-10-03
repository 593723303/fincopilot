"""M1a 存储层 schema 测试。

不连接数据库，只校验表与集合的结构定义——这些字段都是架构决策的载体，
后续重构时被误删会导致设计悄悄失效（而功能测试未必能发现）。
"""

from __future__ import annotations

from app.store.models import Base, Chunk, Document, FinancialMetric


def _cols(model) -> set[str]:
    return {c.name for c in model.__table__.columns}


# ── 表格元数据：中文年报数值题的核心（ADR-016）──────────────


def test_chunk_has_table_metadata_fields():
    """量纲与口径必须是结构化字段，不能留在文本里让模型自己猜。"""
    cols = _cols(Chunk)
    for field in ("unit", "currency", "period", "statement_type", "table_flags"):
        assert field in cols, f"缺少表格元数据字段 {field}，数值题会读错量级或科目"


def test_chunk_supports_parent_child():
    """父子块：子块用于精准检索，父块用于补全上下文（ADR-005）。"""
    cols = _cols(Chunk)
    assert "parent_uid" in cols
    assert "level" in cols


def test_chunk_has_strategy_for_ablation():
    """分块策略必须随块存储，否则不同实验的块混在一起无法隔离（ADR-010）。"""
    assert "strategy" in _cols(Chunk)


def test_chunk_uid_is_unique():
    """chunk_uid 是 PG 与 Milvus 之间唯一的关联键，必须唯一。"""
    assert Chunk.__table__.c.chunk_uid.unique is True


# ── 指标抽取质量：防止「权威的错误数字」（ADR-018）──────────


def test_financial_metric_has_provenance_fields():
    """每条指标都要能回链到原文页，否则错误无法发现。"""
    cols = _cols(FinancialMetric)
    for field in ("source_doc_id", "source_page", "source_chunk_uid"):
        assert field in cols, f"缺少来源回链字段 {field}"


def test_financial_metric_has_quality_fields():
    cols = _cols(FinancialMetric)
    for field in ("confidence", "verify_status", "verify_source"):
        assert field in cols, f"缺少质量校验字段 {field}"


def test_financial_metric_identity_is_unique():
    """同一公司同一期同一科目只能有一条记录。"""
    names = {c.name for c in FinancialMetric.__table__.constraints if c.name}
    assert "uq_metric_identity" in names


def test_financial_metric_keeps_raw_unit():
    """归一后仍要保留原始单位，否则抽取错误无从追溯。"""
    cols = _cols(FinancialMetric)
    assert "unit" in cols and "raw_unit" in cols


# ── 文档幂等与状态机 ────────────────────────────────────


def test_document_has_content_hash_for_idempotency():
    """重复上传同一文件不应重复消耗 embedding。"""
    assert "content_hash" in _cols(Document)


def test_document_tracks_index_cost():
    """入库成本要可核算，否则语料扩容的代价不可见。"""
    assert "index_cost" in _cols(Document)


def test_all_tables_registered():
    assert set(Base.metadata.tables) == {"documents", "chunks", "financial_metrics"}


# ── Milvus collection 定义 ──────────────────────────────


def test_milvus_uses_chinese_analyzer():
    """中文不配分词器时整句会被当成一个 token，BM25 形同虚设。"""
    from app.store.milvus_schema import CHINESE_ANALYZER

    assert CHINESE_ANALYZER == {"type": "chinese"}


def test_milvus_collection_names_configured():
    from app.store.milvus_schema import collection_names

    chunks, cache = collection_names()
    assert chunks == "fin_chunks"
    # 语义缓存必须独立集合，与文档向量混存会污染检索（ADR-008）
    assert cache == "query_cache"
    assert chunks != cache
