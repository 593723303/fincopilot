"""RAG 分支节点：检索 → 相关度判定 → 带引用生成 → 核验。

生成环节的约束是整条链路的收口：前面做的量纲抽取、口径对齐，
只有在 prompt 里强制模型声明单位与科目全称，才真正兑现为正确答案。
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from app.config.experiment import load_experiment
from app.graph.state import GraphState
from app.observability.cost import TokenUsage, merge_usage, usage_from_response
from app.observability.tracing import callbacks, trace_metadata
from app.providers.registry import get_chat
from app.rag.retrievers import (
    QueryFilters,
    RetrievedChunk,
    expand_parents,
    max_score,
    rerank,
    retrieve,
)
from app.store.milvus import milvus

logger = logging.getLogger(__name__)

CITE_PAT = re.compile(r"\[(\d+)\]")

# 流式接口据此区分最终回答与内部分析：两者都是 LLM 调用，
# 不打标就会把路由判定的中间结果推给用户
FINAL_ANSWER_TAG = "final_answer"

_SYSTEM = """你是财报分析助手。只能依据提供的资料回答，不得使用资料之外的知识。

硬性要求：
1. 每个事实性陈述后标注来源编号，如 [1]；多个来源写作 [1][2]。
2. **涉及数值时必须写明单位与科目全称**，例如「营业总收入 1,476.94 万元」，
   不可只写数字。单位以资料中【单位：X】的声明为准，不要自行换算或省略。
3. 科目名称按资料原文写全，不要简写（「归属于上市公司股东的净利润」不可写成「净利润」）。
4. 资料中没有依据的内容，直接说明「所提供资料中未找到相关信息」，不要推测或补充常识。
5. 不提供任何投资建议、买卖推荐或价格预测。

回答简洁，先给结论再给依据。"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        head = c.heading_path or "（无章节信息）"
        src = f"{c.doc_key} 第{c.page_start}页 · {head}"
        parts.append(f"[{i}]（来源：{src}）\n{c.content}")
    return "\n\n".join(parts)


def _citations(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "idx": i,
            "chunk_uid": c.chunk_uid,
            "doc_key": c.doc_key,
            "company_code": c.company_code,
            "report_year": c.report_year,
            "page": c.page_start,
            "heading": c.heading_path,
            "unit": c.unit,
            "chunk_type": c.chunk_type,
            "snippet": c.content[:160],
        }
        for i, c in enumerate(chunks, 1)
    ]


def _merge_usage(state: GraphState, usage: TokenUsage) -> dict:
    return merge_usage(state.get("usage"), usage)


async def retrieve_node(state: GraphState) -> GraphState:
    if state.get("refused"):
        return {}

    exp = load_experiment()
    raw = state.get("filters") or {}
    retry = state.get("retry_count", 0)

    # statement_type 刻意不参与过滤。实测两个问题：
    #   1. 只有三大报表能从标题识别出类型，其余表格为 None，
    #      硬过滤会把它们全部排除——「九、分季度主要财务数据」就因此落空
    #   2. LLM 对该字段判断不可靠，连「你能做什么」都会被填成 cashflow
    # 抽取结果仍保留在 state 中，M4 有评估集后再验证它是否真有正面作用。
    #
    # 重试必须改变检索条件，否则只是原样重跑。相关度不足最常见的原因
    # 是年份抽错，因此重试时放宽年份，公司始终保留。
    filters = QueryFilters(
        company_codes=raw.get("company_codes") or [],
        years=[] if retry >= 1 else (raw.get("years") or []),
        statement_type=None,
    )
    if retry:
        logger.info("第 %d 次重试：放宽过滤条件至 %s", retry, filters)

    query = state.get("rewritten") or state["question"]

    chunks = await retrieve(milvus(), query, filters, exp, keyword_query=query)
    degraded: list[str] = list(state.get("degraded") or [])

    if not chunks:
        # 检索为空时不调用生成模型：没有资料就不可能有依据，
        # 调用只会烧钱并诱导模型编造（架构 §10.2 空检索短路）
        logger.info("检索无结果，过滤条件=%s", filters)
        return {
            "retrieved": [],
            "citations": [],
            "relevance": 0.0,
            "refused": True,
            "refuse_reason": "FC-2001",
            "answer": "已收录的年报中没有检索到与该问题相关的内容。",
            "degraded": degraded,
        }

    chunks = await rerank(chunks, query, exp)
    chunks, expand_degraded = await expand_parents(chunks, exp)
    degraded.extend(expand_degraded)

    return {
        "retrieved": [c.__dict__ for c in chunks],
        "relevance": max_score(chunks),
        "degraded": degraded,
    }


async def grade_node(state: GraphState) -> GraphState:
    """相关度判定。

    低于阈值时先尝试一次改写重试，仍不达标则拒答——
    宁可明确失败，也不让模型脱离文档自由发挥（ADR-009）。
    """
    if state.get("refused"):
        return {}

    exp = load_experiment()
    threshold = exp.generation.relevance_threshold
    relevance = state.get("relevance", 0.0)
    retry = state.get("retry_count", 0)

    if relevance >= threshold:
        return {}

    if retry < exp.generation.max_retry_on_low_relevance:
        logger.info("相关度 %.3f 低于阈值 %.3f，触发改写重试", relevance, threshold)
        return {"retry_count": retry + 1}

    return {
        "refused": True,
        "refuse_reason": "FC-2002",
        "answer": "检索到的内容与问题关联度不足，无法给出有依据的回答。",
    }


async def generate_node(state: GraphState, config: RunnableConfig) -> GraphState:
    if state.get("refused"):
        return {}

    exp = load_experiment()
    profile = exp.generation.profile
    chunks = [RetrievedChunk(**d) for d in (state.get("retrieved") or [])]
    question = state.get("rewritten") or state["question"]

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"资料：\n\n{_format_context(chunks)}\n\n问题：{question}"),
    ]
    # 必须用 astream 而非 ainvoke：astream_events 只在模型以流式方式
    # 调用时才产生 on_chat_model_stream 事件，用 ainvoke 则前端收不到任何
    # token，界面会在生成期间一直空白（M2 实测）。
    # stream_usage 让服务端在流末尾带回 token 用量——成本是核心指标，
    # 不能因为改成流式就丢掉计量。
    llm = get_chat(profile, stream_usage=True)
    # 必须与注入的 config 合并，不能整个替换。
    # 运行时 config 里带着事件传播用的 callbacks，覆盖掉会让
    # astream_events 捕获不到任何 on_chat_model_* 事件，
    # 同时 Langfuse 的调用链也会断开（M2 实测：只剩 on_chain_* 事件）。
    merged = merge_configs(
        config,
        {
            "callbacks": callbacks(),
            # 这个 tag 是流式接口区分「最终回答」与「路由分析」的依据：
            # 两者都是 LLM 调用，不打标就会把内部分析结果推给用户
            "tags": [FINAL_ANSWER_TAG],
            "metadata": trace_metadata(
                node="generate",
                profile=profile,
                conv_id=state.get("conv_id"),
                n_chunks=len(chunks),
            ),
        },
    )

    response = None
    async for piece in llm.astream(messages, config=merged):
        response = piece if response is None else response + piece

    if response is None:
        return {"refused": True, "refuse_reason": "FC-3001", "answer": "模型未返回内容。"}

    text = response.content if isinstance(response.content, str) else str(response.content)
    usage = TokenUsage()
    prompt_tokens, completion_tokens = usage_from_response(response)
    if prompt_tokens == 0 and completion_tokens == 0:
        # 服务端未回传用量时按字符数估算，宁可粗略也不能让成本统计出现空洞
        prompt_tokens = sum(len(str(m.content)) for m in messages)
        completion_tokens = len(text)
        logger.debug("流式响应未带用量，按字符数估算")
    usage.add(profile, prompt_tokens, completion_tokens)

    return {
        "answer": text,
        "citations": _citations(chunks),
        "messages": [AIMessage(content=text)],
        "usage": _merge_usage(state, usage),
    }


async def verify_node(state: GraphState) -> GraphState:
    """核验引用编号。

    模型可能引用不存在的编号。越界引用若原样返回，用户点开会落空，
    比没有引用更糟——它看起来可信却无法追溯。
    """
    answer = state.get("answer") or ""
    citations = state.get("citations") or []
    if not answer or not citations:
        return {}

    valid = {c["idx"] for c in citations}
    used = {int(m) for m in CITE_PAT.findall(answer)}
    invalid = used - valid
    degraded = list(state.get("degraded") or [])

    if invalid:
        logger.warning("回答中存在越界引用 %s，有效范围 1-%d", sorted(invalid), len(citations))
        for idx in invalid:
            answer = answer.replace(f"[{idx}]", "")
        degraded.append(f"invalid_citation:{len(invalid)}")

    # 只保留被实际引用的来源，避免前端展示一堆未被使用的条目
    cited = used & valid
    kept = [c for c in citations if c["idx"] in cited] if cited else citations

    return {"answer": answer, "citations": kept, "degraded": degraded}


async def chat_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """闲聊与能力说明分支，不检索。"""
    if state.get("refused"):
        return {}

    exp = load_experiment()
    profile = exp.router.profile
    llm = get_chat(profile)
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "你是财报分析助手，基于已收录的 A 股年报回答财务问题。"
                    "用户当前的问题不需要查阅资料，简短回应即可，"
                    "并说明你能做什么。不要编造任何财务数据。"
                )
            ),
            HumanMessage(content=state["question"]),
        ],
        config=merge_configs(
            config,
            {
                "callbacks": callbacks(),
                "tags": [FINAL_ANSWER_TAG],
                "metadata": trace_metadata(node="chat", profile=profile),
            },
        ),
    )
    usage = TokenUsage()
    usage.add(profile, *usage_from_response(response))
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "answer": text,
        "messages": [AIMessage(content=text)],
        "usage": _merge_usage(state, usage),
    }
