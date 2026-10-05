"""LangGraph 主图装配。

拓扑：

    START → guard_in → analyze_query ─┬─ chat  ──→ chat_node ────────────────→ END
                                      ├─ agent ──→ agent_node（ReAct 循环）──→ END
                                      └─ rag   ──→ retrieve → grade ─┬─ retry → retrieve
                                                                     ├─ refuse → END
                                                                     └─ ok → generate → verify → END

三条分支的分工：chat 不检索；rag 单次检索即可回答；
agent 处理需要多次检索或计算的问题（跨表加总、同比、跨公司对比）。
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.graph.checkpoint import checkpointer
from app.graph.nodes.agent import agent_node
from app.graph.nodes.query import analyze_query
from app.graph.nodes.rag import (
    cache_lookup_node,
    cache_store_node,
    chat_node,
    generate_node,
    grade_node,
    retrieve_node,
    verify_node,
)
from app.graph.state import GraphState, experiment_of

logger = logging.getLogger(__name__)

MAX_QUESTION_LEN = 500


async def guard_in(state: GraphState) -> GraphState:
    """入口护栏。限流在 API 层做，这里只校验内容本身。

    顺带把用户这轮的提问写进会话历史。各节点只往 messages 里追加
    AIMessage，谁都没有追加用户的那一句——于是多轮里
    `state["messages"]` 全是助手自己说过的话，
    「它去年的呢」既找不到上文的公司，`resolve_companies` 的
    「用户是否真的提过这家公司」也永远校验不到。
    """
    question = (state.get("question") or "").strip()
    if not question:
        return {"refused": True, "refuse_reason": "empty_question", "answer": "请输入问题。"}
    if len(question) > MAX_QUESTION_LEN:
        return {
            "refused": True,
            "refuse_reason": "question_too_long",
            "answer": f"问题过长，请控制在 {MAX_QUESTION_LEN} 字以内。",
        }
    return {"question": question, "messages": [HumanMessage(content=question)]}


def branch_after_analyze(state: GraphState) -> str:
    if state.get("refused"):
        return "end"
    route = state.get("route", "rag")
    # 缓存命中的直接到结尾，不再走任何分支
    if state.get("cached"):
        return "end"
    if route == "chat":
        return "chat"
    if route == "agent":
        return "agent"
    return "rag"


def branch_after_grade(state: GraphState) -> str:
    if state.get("refused"):
        return "end"
    exp = experiment_of(state)
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
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("cache_store", cache_store_node)
    graph.add_node("chat", chat_node)
    graph.add_node("agent", agent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    graph.add_edge(START, "guard_in")
    graph.add_edge("guard_in", "analyze_query")
    # 查缓存放在查询分析之后：实体要先解析出来才能算 entity_key，
    # 而语义缓存靠它做硬约束（同一问法不同年份的向量相似度极高）
    graph.add_edge("analyze_query", "cache_lookup")

    graph.add_conditional_edges(
        "cache_lookup",
        branch_after_analyze,
        {"chat": "chat", "rag": "retrieve", "agent": "agent", "end": END},
    )
    graph.add_edge("chat", "cache_store")
    # Agent 自带引用，无需再走 verify 的编号核验
    graph.add_edge("agent", "cache_store")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        branch_after_grade,
        {"retry": "retrieve", "generate": "generate", "end": END},
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", "cache_store")
    graph.add_edge("cache_store", END)

    return graph


_compiled = None


def get_graph():
    """编译一次复用。

    带 checkpointer 编译：多轮对话的指代消解与 HITL 恢复都依赖它。
    checkpointer 未初始化时（评估脚本、单元测试）照常编译，
    只是不落盘——这两种场景本来就是单轮的。
    """
    global _compiled
    if _compiled is None:
        saver = checkpointer()
        _compiled = build_graph().compile(checkpointer=saver) if saver else build_graph().compile()
        logger.info(
            "LangGraph 主图已编译（checkpointer=%s）",
            type(saver).__name__ if saver else "无",
        )
    return _compiled
