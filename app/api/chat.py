"""问答接口。

流式版本遵循架构 §8.2 的事件协议。推送的不只是文本 token，
还包括路由分支与节点进度——Agent 的多步推理对用户可见，
同时这也是排查问题时最直接的信息来源。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config.experiment import load_experiment
from app.graph.build import get_graph
from app.graph.nodes.rag import FINAL_ANSWER_TAG
from app.graph.state import new_state
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])

# 对用户有意义的节点，内部辅助节点不外推
VISIBLE_NODES = {
    "analyze_query": "理解问题",
    "retrieve": "检索年报",
    "grade": "评估相关度",
    "generate": "生成回答",
    "verify": "核验引用",
}


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _validate_exp(exp_id: str | None):
    if not exp_id:
        return
    try:
        load_experiment(exp_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """非流式问答。供评估脚本与自动化测试调用。"""
    _validate_exp(req.exp_id)
    conv_id = req.conv_id or uuid.uuid4().hex
    started = time.perf_counter()

    try:
        result = await get_graph().ainvoke(new_state(question=req.question, conv_id=conv_id))
    except Exception as exc:
        logger.exception("问答失败")
        raise HTTPException(
            status_code=502,
            detail=f"{type(exc).__name__}: {exc}。请检查 .env 中的 API Key 与存储连接。",
        ) from exc

    return ChatResponse(
        conv_id=conv_id,
        answer=result.get("answer", ""),
        refused=result.get("refused", False),
        refuse_reason=result.get("refuse_reason"),
        citations=result.get("citations", []),
        degraded=result.get("degraded", []),
        cached=result.get("cached", False),
        usage=result.get("usage", {}),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


async def _event_stream(question: str, conv_id: str) -> AsyncIterator[dict]:
    started = time.perf_counter()
    graph = get_graph()
    final: dict = {}
    first_token_ms: int | None = None

    yield _sse("meta", {"conv_id": conv_id, "exp_id": load_experiment().exp_id})

    try:
        async for ev in graph.astream_events(
            new_state(question=question, conv_id=conv_id), version="v2"
        ):
            kind = ev.get("event")
            name = ev.get("name", "")

            if kind == "on_chain_start" and name in VISIBLE_NODES:
                yield _sse("step", {"node": name, "label": VISIBLE_NODES[name], "status": "running"})

            elif kind == "on_chain_end" and name in VISIBLE_NODES:
                out = ev.get("data", {}).get("output") or {}
                payload = {"node": name, "label": VISIBLE_NODES[name], "status": "done"}

                if name == "analyze_query" and isinstance(out, dict):
                    route = out.get("route")
                    if route:
                        yield _sse("route", {"branch": route, "filters": out.get("filters")})
                if name == "retrieve" and isinstance(out, dict):
                    payload["hits"] = len(out.get("retrieved") or [])
                yield _sse("step", payload)

            elif kind == "on_chat_model_stream":
                # 只转发最终回答的 token，路由分析的输出不推给用户
                if FINAL_ANSWER_TAG not in (ev.get("tags") or []):
                    continue
                chunk = ev.get("data", {}).get("chunk")
                text = getattr(chunk, "content", "") or ""
                if not text:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                yield _sse("token", {"text": text})

            elif kind == "on_chain_end" and name == "LangGraph":
                final = ev.get("data", {}).get("output") or {}

        # 引用在生成之后才完成核验，故在结尾统一推送
        for c in final.get("citations") or []:
            yield _sse("citation", c)

        if final.get("refused"):
            yield _sse(
                "refused",
                {"reason": final.get("refuse_reason"), "answer": final.get("answer", "")},
            )
            # 拒答路径没有流式 token，整段补推一次
            if final.get("answer"):
                yield _sse("token", {"text": final["answer"]})

        yield _sse(
            "done",
            {
                "usage": final.get("usage", {}),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "first_token_ms": first_token_ms,
                "cached": final.get("cached", False),
                "degraded": final.get("degraded", []),
                "citation_count": len(final.get("citations") or []),
            },
        )
    except Exception as exc:
        logger.exception("流式问答失败")
        yield _sse("error", {"code": "FC-3001", "message": f"{type(exc).__name__}: {exc}"})


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式问答。事件协议见架构 §8.2。"""
    _validate_exp(req.exp_id)
    conv_id = req.conv_id or uuid.uuid4().hex
    return EventSourceResponse(_event_stream(req.question, conv_id))
