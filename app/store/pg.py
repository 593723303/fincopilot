"""PostgreSQL 连接管理（SQLAlchemy 2.x async + asyncpg）。

承载：文档与块元数据、财务指标、会话、评估结果、LangGraph checkpoint。
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_pg() -> None:
    global _engine, _session_factory
    settings = get_settings()
    _engine = create_async_engine(
        settings.postgres_dsn,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        echo=False,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("PostgreSQL 连接池已创建")


async def close_pg() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("PostgreSQL 连接池已关闭")


def session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("PostgreSQL 尚未初始化，请检查应用 lifespan")
    return _session_factory


async def ping_pg() -> tuple[bool, str]:
    """就绪探针用。返回 (是否连通, 说明)。"""
    if _engine is None:
        return False, "未初始化"
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
