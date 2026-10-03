"""LangGraph 主图装配。

M2 的拓扑（Agent 分支在 M5 接入）：

    START → guard_in → analyze_query ─┬─ chat ──→ chat_node ─────────────→ END
                                      └─ rag ──→ retrieve → grade ─┬─ retry → retrieve
                                                                   ├─ refuse → END
                                                                   └─ ok → generate → verify → END

路由判为 agent 时，当前降级走 rag 并标记 degraded——
能力未实现就老实降级，不假装支持。
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.config.experiment import load_experiment
from app.graph.nodes.query import analyze_query
from app.graph.nodes.rag import (
    chat_node,
    generate_node,
    grade_node,
    retrieve_node,
    verify_node,
)
from app.graph.state import GraphState

logger = logging.getLogger(__name__)

MAX_QUESTION_LEN = 500


async def guard_in(state: GraphState) -> GraphState:
    """入口护栏。限流在 API 层做，这里只校验内容本身。"""
    question = (state.get("question") or "").strip()
    if not question:
        return {"refused": True, "refuse_reason": "empty_question", "answer": "请输入问题。"}
    if len(question) > MAX_QUESTION_LEN:
        return {
            "refused": True,
            "refuse_reason": "question_too_long",
            "answer": f"问题过长，请控制在 {MAX_QUESTION_LEN} 字以内。",
        }
    return {"question": question}


def branch_after_analyze(state: GraphState) -> str:
    if state.get("refused"):
        return "end"
    route = state.get("route", "rag")
    if route == "chat":
        return "chat"
    if route == "agent":
        # M5 之前没有 Agent 能力，降级到 rag 并留痕，便于评估时区分
        degraded = list(state.get("degraded") or [])
        if "agent_not_available" not in degraded:
            degraded.append("agent_not_available")
            state["degraded"] = degraded
        return "rag"
    return "rag"


def branch_after_grade(state: GraphState) -> str:
    if state.get("refused"):
        return "end"
    exp = load_experiment()
    relevance = state.get("relevance", 0.0)
    if relevance < exp.generation.relevance_threshold:
        # grade 已把 retry_count 加过，这里只负责选边
        if state.get("retry_count", 0) <= exp.generation.max_retry_on_low_relevance:
            return "retry"
        return "end"
    return "generate"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("guard_in", guard_in)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("chat", chat_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "guard_in")
    graph.add_edge("guard_in", "analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        branch_after_analyze,
        {"chat": "chat", "rag": "retrieve", "end": END},
    )
    graph.add_edge("chat", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        branch_after_grade,
        {"retry": "retrieve", "generate": "generate", "end": END},
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


_compiled = None


def get_graph():
    """编译一次复用。M5 接入 Checkpointer 后需改为按会话传入。"""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        logger.info("LangGraph 主图已编译（M2：chat / rag 双分支）")
    return _compiled
