"""Redis 连接管理。

承载：两级缓存、会话短期记忆、限流计数、离线任务队列。
架构 §3：Redis 是弱依赖——不可用时缓存与会话降级，服务仍可运行。
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis, from_url

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: Redis | None = None


async def init_redis() -> None:
    global _client
    settings = get_settings()
    _client = from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    logger.info("Redis 客户端已创建")


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis 连接已关闭")


def redis() -> Redis:
    if _client is None:
        raise RuntimeError("Redis 尚未初始化，请检查应用 lifespan")
    return _client


async def ping_redis() -> tuple[bool, str]:
    if _client is None:
        return False, "未初始化"
    try:
        await _client.ping()
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
