from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    conv_id: str | None = Field(None, description="会话 ID，留空则新建")
    exp_id: str | None = Field(None, description="指定实验配置，留空用默认")


class ChatResponse(BaseModel):
    conv_id: str
    answer: str
    refused: bool = False
    refuse_reason: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    cached: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
