"""离线任务 Worker 入口。

    arq app.worker.Settings

与 API 共用同一镜像，仅启动命令不同（架构 §4.2）。
负载特征不同是拆分的理由：在线请求 IO 密集、响应要快；
文档解析 CPU 密集、分钟级耗时，放在一起会互相拖累。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from app.config.experiment import load_experiment
from app.config.settings import get_settings
from app.rag.indexer import ingest_pdf
from app.store.milvus import close_milvus, init_milvus
from app.store.pg import close_pg, init_pg
from app.store.queue import QUEUE_NAME, redis_settings

logger = logging.getLogger(__name__)


async def ingest_document(
    ctx: dict,
    file_path: str,
    exp_id: str | None = None,
    max_pages: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """解析并入库一份年报。

    返回值会被 arq 持久化，可通过 job_id 查询结果。
    """
    exp = load_experiment(exp_id)
    result = await ingest_pdf(
        Path(file_path),
        ctx["milvus"],
        exp=exp,
        max_pages=max_pages,
        force=force,
    )
    return {
        "doc_id": result.doc_id,
        "doc_key": result.doc_key,
        "status": result.status,
        "chunk_count": result.chunk_count,
        "child_count": result.child_count,
        "embed_cost": result.embed_cost,
        "skipped": result.skipped,
        "error": result.error,
    }


async def startup(ctx: dict) -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )
    await init_pg()
    await init_milvus()
    # Worker 是独立进程，需要自己持有 Milvus 客户端
    ctx["milvus"] = MilvusClient(
        uri=settings.milvus_uri, token=settings.milvus_token or None
    )
    logger.info("Worker 启动完成，队列=%s", QUEUE_NAME)


async def shutdown(ctx: dict) -> None:
    client = ctx.get("milvus")
    if client is not None:
        client.close()
    await close_milvus()
    await close_pg()
    logger.info("Worker 已停止")


class Settings:
    functions = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown
    queue_name = QUEUE_NAME
    # 解析是 CPU 密集任务，并发过高只会互相争抢
    max_jobs = int(get_settings().yaml_get("ingestion.worker_concurrency", 2))
    # 单份年报可能数百页，给足超时
    job_timeout = 1800
    keep_result = 3600
    # arq 读取的是类属性而非实例属性
    redis_settings = redis_settings()
