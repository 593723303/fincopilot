"""Milvus 连接管理。

承载：文档块向量（dense + sparse 混合检索）与语义缓存。
架构 §3：Milvus 是强依赖——不可用时不降级，直接报错。
宁可明确失败，也不能让模型脱离文档自由发挥（ADR-009）。

pymilvus 的 MilvusClient 是同步 SDK，因此 ping 放到线程池执行，
避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import logging

from pymilvus import MilvusClient

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None


async def init_milvus() -> None:
    global _client
    settings = get_settings()

    def _connect() -> MilvusClient:
        kwargs = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        return MilvusClient(**kwargs)

    _client = await asyncio.to_thread(_connect)
    logger.info("Milvus 客户端已创建：%s", settings.milvus_uri)


async def close_milvus() -> None:
    global _client
    if _client is not None:
        await asyncio.to_thread(_client.close)
        _client = None
        logger.info("Milvus 连接已关闭")


def milvus() -> MilvusClient:
    if _client is None:
        raise RuntimeError("Milvus 尚未初始化，请检查应用 lifespan")
    return _client


async def ping_milvus() -> tuple[bool, str]:
    if _client is None:
        return False, "未初始化"
    try:
        collections = await asyncio.to_thread(_client.list_collections)
        return True, f"ok, collections={len(collections)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
