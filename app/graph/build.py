"""LangGraph 主图装配。

M0 只有一条最小路径：guard_in → answer → END。
它的作用不是回答得好，而是证明「配置 → Provider → LangGraph → Langfuse」
这条链路是通的。M2 起在此基础上接入路由、检索、生成与 Agent 分支。
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config.experiment import load_experiment
from app.graph.state import GraphState
from app.observability.cost import TokenUsage, usage_from_response
from app.observability.tracing import callbacks, trace_metadata
from app.providers.registry import get_chat

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "你是一个财报分析助手。当前为开发骨架阶段，尚未接入年报检索能力。\n"
    "若用户询问具体财务数据，请直接说明尚未接入资料库，不要凭记忆作答，更不要编造数字。"
)


async def guard_in(state: GraphState) -> GraphState:
    """入口护栏。M0 只做长度校验，限流与注入检测在 M2 补齐。"""
    question = (state.get("question") or "").strip()
    max_len = 500
    if not question:
        return {"refused": True, "refuse_reason": "问题为空", "answer": "请输入问题。"}
    if len(question) > max_len:
        return {
            "refused": True,
            "refuse_reason": "问题过长",
            "answer": f"问题过长，请控制在 {max_len} 字以内。",
        }
    return {"question": question}


async def answer(state: GraphState) -> GraphState:
    """调用模型作答，并记账 token 与成本。"""
    if state.get("refused"):
        return {}

    exp = load_experiment()
    profile = exp.generation.profile
    llm = get_chat(profile)

    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=state["question"])]
    response = await llm.ainvoke(
        messages,
        config={
            "callbacks": callbacks(),
            "metadata": trace_metadata(conv_id=state.get("conv_id"), node="answer", profile=profile),
        },
    )

    prompt_tokens, completion_tokens = usage_from_response(response)
    usage = TokenUsage()
    usage.add(profile, prompt_tokens, completion_tokens)

    text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "answer": text,
        "messages": [AIMessage(content=text)],
        "usage": usage.to_dict(),
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("guard_in", guard_in)
    graph.add_node("answer", answer)

    graph.add_edge(START, "guard_in")
    graph.add_edge("guard_in", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


_compiled = None


def get_graph():
    """编译一次复用。M5 接入 Checkpointer 后此处需要改为按需传入 checkpointer。"""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
        logger.info("LangGraph 主图已编译（M0 最小图）")
    return _compiled
