"""PostgreSQL 表结构定义（SQLAlchemy 2.x ORM）。

对应架构文档 §7.1。M1 只建离线入库所需的三张表；
会话相关表在 M2 建，评估相关表在 M3 建——当前用不到的不提前建（决策阶梯）。
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# 文档入库状态机：pending → parsing → chunking → embedding → ready
#                              ↘ failed（任一阶段失败）
DOC_STATUSES = ("pending", "parsing", "chunking", "embedding", "ready", "failed")


class Document(Base):
    """年报文档主表。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_key: Mapped[str] = mapped_column(String(64), unique=True, comment="如 600519_2024_annual")

    company_code: Mapped[str] = mapped_column(String(16), index=True)
    company_name: Mapped[str] = mapped_column(String(128))
    report_year: Mapped[int] = mapped_column(Integer, index=True)
    report_type: Mapped[str] = mapped_column(String(16), comment="annual | h1 | q1 | q3")

    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(Text)
    # 幂等去重依据：同一份文件重复上传不重复消耗 embedding
    content_hash: Mapped[str] = mapped_column(String(64), index=True)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 本次入库的 embedding 花费，用于核算语料扩容成本
    index_cost: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)

    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    error_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(f"status IN {DOC_STATUSES}", name="ck_documents_status"),
        Index("ix_documents_company_year", "company_code", "report_year"),
    )


class Chunk(Base):
    """文档块。向量存于 Milvus，此处保存原文与结构关系。"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 与 Milvus 主键一致，是两个存储之间的唯一关联
    chunk_uid: Mapped[str] = mapped_column(String(128), unique=True)
    doc_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    # 父子块（small-to-big）：子块用于精准检索，父块用于补全上下文
    parent_uid: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    level: Mapped[int] = mapped_column(SmallInteger, comment="0=父块 1=子块")

    chunk_type: Mapped[str] = mapped_column(String(16), comment="text | table")
    heading_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text)

    # 消融实验隔离：不同分块策略的块共存于同一张表而互不干扰
    strategy: Mapped[str] = mapped_column(String(16), index=True)

    # ── 表格块专属元数据（chunk_type='table' 时必填）──
    # 中文年报的数值题主要错在量纲与口径，而不是检索召回。
    # 把单位和科目结构化出来，不留在文本里让模型自己猜（ADR-016）。
    unit: Mapped[str | None] = mapped_column(String(8), nullable=True, comment="元|万元|亿元")
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    period: Mapped[str | None] = mapped_column(String(16), nullable=True, comment="FY2024|H1-2024")
    statement_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="balance|income|cashflow|other"
    )
    # 解析风险标记，如 cross_page、merged_cell，供排查数值题错误时归因
    table_flags: Mapped[list[str] | None] = mapped_column(ARRAY(String(24)), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        CheckConstraint("level IN (0, 1)", name="ck_chunks_level"),
        CheckConstraint("chunk_type IN ('text', 'table')", name="ck_chunks_type"),
        Index("ix_chunks_doc_strategy", "doc_id", "strategy"),
    )


class FinancialMetric(Base):
    """结构化财务指标，Text2SQL 的数据源。

    抽取过程本身会出错，而错误的结构化数据会产出「看起来精确的错误数字」——
    比检索幻觉更难察觉，因为它带着数据库查询的权威外观。
    因此每条记录都必须可回链到原文、带置信度、并与外部数据交叉校验（ADR-018）。
    """

    __tablename__ = "financial_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    company_code: Mapped[str] = mapped_column(String(16), index=True)
    company_name: Mapped[str] = mapped_column(String(128))
    report_year: Mapped[int] = mapped_column(Integer)
    period: Mapped[str] = mapped_column(String(16), comment="FY | H1 | Q1 | Q3")

    metric_code: Mapped[str] = mapped_column(String(48), comment="revenue | net_profit | ...")
    # 存科目全称，不存简称：营业收入 ≠ 营业总收入，净利润 ≠ 归母净利润
    metric_name: Mapped[str] = mapped_column(String(128))
    value: Mapped[Decimal | None] = mapped_column(Numeric(20, 4), nullable=True)
    unit: Mapped[str] = mapped_column(String(8), comment="归一后单位")
    raw_unit: Mapped[str | None] = mapped_column(String(8), nullable=True, comment="抽取时原始单位")

    # ── 质量保障字段，缺一不可 ──
    source_doc_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_chunk_uid: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)
    verify_status: Mapped[str] = mapped_column(
        String(16), default="unverified",
        comment="unverified | cross_checked | mismatch | manual_ok",
    )
    verify_source: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="如 akshare")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "company_code", "report_year", "period", "metric_code", name="uq_metric_identity"
        ),
        CheckConstraint(
            "verify_status IN ('unverified', 'cross_checked', 'mismatch', 'manual_ok')",
            name="ck_metric_verify_status",
        ),
        Index("ix_metrics_lookup", "company_code", "report_year", "metric_code"),
    )
