"""问答接口。

M0 只提供非流式版本，用于打通链路；SSE 流式与 Agent 分支在 M2 / M5 补齐。
接口协议（架构 §8.2）已定稿，后续扩展不改变现有字段语义。
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException

from app.config.experiment import load_experiment
from app.graph.build import get_graph
from app.graph.state import new_state
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if req.exp_id:
        try:
            load_experiment(req.exp_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    conv_id = req.conv_id or uuid.uuid4().hex
    started = time.perf_counter()

    state = new_state(question=req.question, conv_id=conv_id)
    try:
        result = await get_graph().ainvoke(state)
    except Exception as exc:
        # 模型未配置 Key 时在这里失败，给出可操作的提示而不是裸 500
        raise HTTPException(
            status_code=502,
            detail=f"模型调用失败：{type(exc).__name__}: {exc}。请检查 .env 中的 API Key 是否已填写。",
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
