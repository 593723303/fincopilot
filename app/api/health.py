"""健康探针。

/healthz  进程存活，不碰任何外部依赖 —— 给容器编排用
/readyz   三个存储是否连通 + 模型 Key 是否配置 —— 给人看，M0 的验收入口
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.config.experiment import load_experiment
from app.config.settings import get_settings
from app.providers.registry import configured_profiles
from app.store.milvus import ping_milvus
from app.store.pg import ping_pg
from app.store.queue import ping_queue
from app.store.redis_client import ping_redis

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    settings = get_settings()
    return {"status": "ok", "app": settings.yaml_get("app.name", "FinCopilot"), "env": settings.app_env}


@router.get("/readyz")
async def readyz(response: Response) -> dict:
    settings = get_settings()
    exp = load_experiment()

    pg_ok, pg_msg = await ping_pg()
    redis_ok, redis_msg = await ping_redis()
    milvus_ok, milvus_msg = await ping_milvus()
    queue_ok, queue_msg = await ping_queue()

    stores = {
        "postgres": {"ok": pg_ok, "detail": pg_msg},
        "redis": {"ok": redis_ok, "detail": redis_msg},
        "milvus": {"ok": milvus_ok, "detail": milvus_msg},
        "task_queue": {"ok": queue_ok, "detail": queue_msg},
    }

    # Milvus 与 PostgreSQL 是强依赖；Redis 不可用只降级不阻断（架构 §3）
    ready = pg_ok and milvus_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "ready": ready,
        "stores": stores,
        "models_configured": configured_profiles(),
        "langfuse": settings.langfuse_enabled,
        "experiment": {"exp_id": exp.exp_id, "description": exp.description},
    }
