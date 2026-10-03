"""FastAPI 入口。

架构 §4.2：同一镜像通过启动命令区分 API / Worker 两种角色。
    API    uvicorn app.main:app
    Worker arq app.worker.Settings   （M1 引入）
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health
from app.config.experiment import load_experiment
from app.config.settings import get_settings
from app.store.milvus import close_milvus, init_milvus
from app.store.pg import close_pg, init_pg
from app.store.redis_client import close_redis, init_redis

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _setup_logging(settings.log_level)

    exp = load_experiment()
    logger.info("启动 %s，环境=%s，实验=%s", settings.yaml_get("app.name"), settings.app_env, exp.exp_id)

    # 连接失败不阻止启动 —— 让 /readyz 把问题暴露出来，
    # 比启动时崩溃更利于定位（尤其本地只起了部分容器时）
    for name, fn in (("PostgreSQL", init_pg), ("Redis", init_redis), ("Milvus", init_milvus)):
        try:
            await fn()
        except Exception as exc:
            logger.warning("%s 初始化失败（可用 /readyz 查看详情）：%s", name, exc)

    yield

    await close_milvus()
    await close_redis()
    await close_pg()
    logger.info("已关闭全部连接")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.yaml_get("app.name", "FinCopilot"),
        version=settings.yaml_get("app.version", "0.1.0"),
        description="财报智能问答与分析服务",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.yaml_get("server.cors_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    return app


app = create_app()
