"""Langfuse 接入。

架构 §10.3：没有 trace 就无法归因 bad case，因此从第一行代码起就接上。
未配置 Key 时静默降级为空回调——本地开发不应被可观测组件阻塞（§3 弱依赖）。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langfuse_handler() -> Any | None:
    """返回 LangChain 回调处理器；未配置或导入失败时返回 None。"""
    settings = get_settings()
    if not settings.langfuse_enabled:
        logger.info("Langfuse 未配置，本次运行不上报 trace（不影响功能）")
        return None
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:  # pragma: no cover - 版本差异时的兜底路径
        try:
            from langfuse.langchain import CallbackHandler  # type: ignore[no-redef]
        except ImportError:
            logger.warning("langfuse 已安装但未找到 CallbackHandler，跳过 trace 上报")
            return None
    try:
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse 初始化失败，跳过 trace 上报：%s", exc)
        return None


def callbacks(extra: list | None = None) -> list:
    """统一的回调列表构造。所有模型调用都应带上它。"""
    handler = get_langfuse_handler()
    result = [handler] if handler else []
    if extra:
        result.extend(extra)
    return result


def trace_metadata(**kwargs: Any) -> dict[str, Any]:
    """自定义属性。exp_id 必须始终在内——否则无法区分不同实验的 trace。"""
    settings = get_settings()
    meta = {"exp_id": settings.exp_id, "env": settings.app_env}
    meta.update({k: v for k, v in kwargs.items() if v is not None})
    return meta
