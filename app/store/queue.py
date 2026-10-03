"""arq 任务队列连接。

架构 §4.2：API 与 Worker 是同一镜像的两种角色。
API 负责投递任务后立即返回，解析入库这类分钟级工作交给 Worker，
避免阻塞在线请求。
"""

from __future__ import annotations

import logging

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

QUEUE_NAME = "fincopilot:ingest"

_pool: ArqRedis | None = None


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url)


async def init_queue() -> None:
    global _pool
    _pool = await create_pool(redis_settings(), default_queue_name=QUEUE_NAME)
    logger.info("任务队列连接已建立")


async def close_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("任务队列连接已关闭")


def queue() -> ArqRedis:
    if _pool is None:
        raise RuntimeError("任务队列尚未初始化，请检查应用 lifespan")
    return _pool


async def ping_queue() -> tuple[bool, str]:
    if _pool is None:
        return False, "未初始化"
    try:
        await _pool.ping()
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
