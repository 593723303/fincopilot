"""LangGraph Checkpointer：把会话状态落到 PostgreSQL。

为什么需要：

  1. **多轮对话**——「它去年的呢」要靠上一轮的状态做指代消解，
     进程重启后不能丢
  2. **HITL 中断恢复**——`interrupt()` 把执行停在中途等人确认，
     恢复时必须读回停下来那一刻的完整状态
  3. **排查**——一次 bad case 发生在哪个节点、当时 state 是什么，
     checkpoint 是唯一完整的现场

用 PG 而不是内存：内存版重启即失，等于前两条都做不到。
代价是多一个 psycopg 依赖（项目主路径用 asyncpg，
langgraph 的 PG checkpointer 只支持 psycopg，两者可以共存）。
"""

from __future__ import annotations

import asyncio
import logging
import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_saver = None
_cm = None


def _ensure_selector_loop_policy() -> None:
    """Windows 默认的 ProactorEventLoop 跑不了 psycopg 的异步模式。

    报错是 `Psycopg cannot use the 'ProactorEventLoop' to run in async mode`。
    只在 Windows 上切换，且只在事件循环尚未创建时有效——
    所以必须在应用启动早期调用，而不是等到第一次用 checkpointer。
    """
    if sys.platform != "win32":
        return
    policy = asyncio.get_event_loop_policy()
    if isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy):
        return
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logger.info("已切换为 WindowsSelectorEventLoopPolicy（psycopg 异步模式所需）")


def _psycopg_dsn() -> str:
    """把 SQLAlchemy 的 asyncpg DSN 转成 psycopg 能用的形式。"""
    dsn = get_settings().postgres_dsn
    return dsn.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )


async def init_checkpointer():
    """建立 checkpointer 并建表。连不上时退回内存版，不阻断启动。

    退回是有意的：checkpoint 丢了只影响多轮与恢复，
    而让整个服务起不来会把一个可降级的问题变成全面不可用。
    但必须留痕——静默降级会让人以为多轮正常工作。
    """
    global _saver, _cm
    if _saver is not None:
        return _saver
    _ensure_selector_loop_policy()
    try:
        _cm = AsyncPostgresSaver.from_conn_string(_psycopg_dsn())
        _saver = await _cm.__aenter__()
        await _saver.setup()
        logger.info("Checkpointer 已就绪（PostgreSQL）")
    except Exception as exc:
        logger.warning("PostgreSQL checkpointer 不可用，退回内存版：%s", exc)
        _saver, _cm = InMemorySaver(), None
    return _saver


async def close_checkpointer() -> None:
    global _saver, _cm
    if _cm is not None:
        try:
            await _cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("关闭 checkpointer 失败：%s", exc)
    _saver, _cm = None, None


def checkpointer():
    return _saver
