"""重排服务。

百炼的 rerank 不走 OpenAI 兼容端点，因此无法复用 Provider 的
配置表 + ChatOpenAI 工厂（ADR-004 的抽象只覆盖兼容接口），
这里用 HTTP 直连。其余约定保持一致：超时、重试、成本记账、
失败时降级而非报错。

降级策略（架构 §10.1）：重排服务不可用时跳过重排、保留原序，
并标记 degraded——重排是质量增强而非必需环节，
为它牺牲可用性不划算。
"""

from __future__ import annotations

import logging
import os

import httpx

from app.providers.registry import get_registry

logger = logging.getLogger(__name__)

DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
TIMEOUT = 20.0
MAX_RETRIES = 2
# 单文档超长会拖慢重排且收益递减，截断到足以判断相关性的长度
MAX_DOC_CHARS = 2000


class RerankUnavailable(RuntimeError):
    """重排服务不可用。调用方应降级而非中断。"""


async def rerank_documents(
    query: str, documents: list[str], top_n: int, model: str | None = None
) -> list[tuple[int, float]]:
    """对候选文档重排，返回 [(原始下标, 相关性分数)]，按分数降序。

    抛出 RerankUnavailable 时调用方应保留原序继续。
    """
    if not documents:
        return []

    model = model or get_registry().rerank_model() or "gte-rerank-v2"
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        from app.config.settings import get_settings

        api_key = get_settings().dashscope_api_key
    if not api_key:
        raise RerankUnavailable("未配置 DASHSCOPE_API_KEY")

    payload = {
        "model": model,
        "input": {
            "query": query[:1000],
            "documents": [d[:MAX_DOC_CHARS] for d in documents],
        },
        "parameters": {"return_documents": False, "top_n": min(top_n, len(documents))},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(DASHSCOPE_RERANK_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                raise RerankUnavailable(f"HTTP {resp.status_code}: {resp.text[:160]}")
            data = resp.json()
            results = (data.get("output") or {}).get("results") or []
            if not results:
                raise RerankUnavailable(f"响应中无 results：{str(data)[:160]}")
            return [
                (int(r["index"]), float(r.get("relevance_score", 0.0)))
                for r in results
                if "index" in r
            ]
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                logger.warning("重排第 %d 次失败，重试：%s", attempt + 1, exc)
                continue
    raise RerankUnavailable(str(last_err))
