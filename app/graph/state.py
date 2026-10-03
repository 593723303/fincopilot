"""LangGraph 状态定义。

架构 §5.2 的完整字段在此落地。M0 只用到其中少数几个，
但结构先定好——后续节点逐个填充，避免中途改状态结构导致全图返工。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Route = Literal["chat", "rag", "agent"]


class GraphState(TypedDict, total=False):
    # ── 会话上下文 ──
    conv_id: str
    trace_id: str
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 查询处理 ──
    question: str
    rewritten: str | None
    route: Route
    filters: dict[str, Any]

    # ── 检索结果（M2 起使用）──
    retrieved: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    relevance: float
    retry_count: int

    # ── Agent 执行（M5 起使用）──
    tool_calls: int
    tool_errors: int
    scratchpad: list[dict[str, Any]]

    # ── 输出 ──
    answer: str
    refused: bool
    refuse_reason: str | None
    needs_approval: bool
    degraded: list[str]
    cached: bool
    usage: dict[str, Any]


def new_state(question: str, conv_id: str, trace_id: str = "") -> GraphState:
    return GraphState(
        conv_id=conv_id,
        trace_id=trace_id,
        question=question,
        messages=[],
        filters={},
        retrieved=[],
        citations=[],
        relevance=0.0,
        retry_count=0,
        tool_calls=0,
        tool_errors=0,
        scratchpad=[],
        answer="",
        refused=False,
        refuse_reason=None,
        needs_approval=False,
        degraded=[],
        cached=False,
        usage={},
    )
