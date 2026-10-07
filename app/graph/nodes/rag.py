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

from app.graph.state import GraphState, experiment_of
from app.observability.cost import TokenUsage, merge_usage, usage_from_response
from app.observability.tracing import callbacks, trace_metadata
from app.providers.registry import get_chat
from app.rag.cache import entity_key, get_exact, get_semantic, put_exact, put_semantic
from app.rag.retrievers import (
    QueryFilters,
    RetrievedChunk,
    expand_parents,
    force_summary_table,
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
   不可只写数字。**单位与数字都照抄资料原文，一律不要换算。**
   资料写「31,629,416,193.83 元」就原样给出，不要改写成「316.29 亿元」，
   更不要写成「31.63 亿元」——换算是最常见的出错点，实测就错过一次 10 倍。
   需要直观感受时，可以在原值之后补一句约数，但主数值必须是原文那个。
3. 科目名称按资料原文写全，不要简写（「归属于上市公司股东的净利润」不可写成「净利润」）。
4. **财务数据默认取合并报表口径**。年报里同一科目往往有三处：
   「近三年主要会计数据」摘要表、合并报表、母公司报表（章节名含「母公司」）。
   除非问题明确问母公司，否则不得引用母公司数字——
   两者差异可以很大（实测经营活动现金流净额相差三倍）。
   摘要表与合并报表都可用，优先摘要表，因为它同时列出往年数据。
5. 公司基本信息（股票代码、注册地址、联系电话、邮编、公司全称等）
   同样在年报里（通常在「公司简介」一节），这些都属于可回答范围，
   不要以「只能回答财务问题」为由拒答。
6. **三年并列表必须按列头确认年度**。「主要会计数据」表并排列出
   本年 / 上年 / 增减 / 前年四列，第二个数字是上一年、最后一个才是前年。
   取数前先在表头里找到问题问的那个年份，再沿该列往下取；
   表格有合并单元格时列可能对不齐，拿不准就明说未找到，不要取最左边那个。
   表格出现「重述后 / 重述前」「调整后 / 调整前」子列头时，
   往年的列被拆成了两列，数据行的列数比表头多，中间各列对应哪一年无法确定。
   **这时只有最左边那个数值列可以确定是本年**（本年不存在调整前后之分）；
   问的若不是本年，就说明无法确定，不要硬取一个。
7. 「营业收入」与「营业总收入」是两个不同科目，金额不同，问哪个取哪个，
   不要互相替代；「净利润」与「归属于上市公司股东的净利润」同理。
8. 问「同比/环比变化了多少」「增长了多少」时答案给百分比，
   需要的话再补绝对额；只给绝对额等于没回答这个问题。
9. **问题里的期间或科目在资料中不存在时，必须点明它不存在**，
   不得用相近的数据顶替。例如「第五季度」——一年只有四个季度，
   这时要回答「不存在第五季度」，而不是把第四季度的数给出去。
10. 资料中没有依据的内容，直接说明「所提供资料中未找到相关信息」，不要推测或补充常识。
11. 不提供任何投资建议、买卖推荐或价格预测。
12. **比较优劣、给出选择建议的问题一律不答**：「哪只股票更值得投资」
    「该不该买」「哪家更有前景」——直接说明不提供投资建议，
    不要先把两家的财务数据罗列一遍再说不给建议，罗列本身就构成了倾向性。
    客观的单项数值对比（「哪家营业收入更高」）可以答，因为它只是读数。

回答简洁，先给结论再给依据。"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        head = c.heading_path or "（无章节信息）"
        src = f"{c.doc_key} 第{c.page_start}页 · {head}"
        parts.append(f"[{i}]（来源：{src}）\n{c.content}")
    return "\n\n".join(parts)


def build_citations(chunks: list[RetrievedChunk]) -> list[dict]:
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


# 疑问句的壳：对 BM25 没有信息量，却实打实占据词频权重
_FILLER = ("是多少", "多少", "是什么", "请问", "的具体数值", "吗", "呢", "？", "?", "，", ",")


# 年报的三大报表只有「本期 / 上期」两列。比报告年度早两年及以上的数据，
# 整份年报里只存在于「近三年主要会计数据」这张汇总表中。
# 不点名这张表的话，检索会被标题几乎逐字相同的现金流量表附注淹没——
# 实测格力「2023年经营活动产生的现金流量净额」里，汇总表 p7 排第 20 名
# （正好被 top_n=15 截掉），加上这个提示后升到第 12 名。
# 用两个交易所共有的措辞。这张表的标题按交易所不同：
#   上交所：七、近三年主要会计数据和财务指标
#   深交所：六、主要会计数据和财务指标
# 写死「近三年主要会计数据」只命中上交所那半边——实测海康威视的汇总表
# 用它仍进不了前 20，换成共有子串后升到第 3 名。四家实测名次：
#   不加提示       海康>20  格力20  恒瑞19  茅台16
#   近三年主要会计数据  海康>20  格力12  恒瑞 5  茅台 6
#   主要会计数据和财务指标 海康 3  格力 9  恒瑞 5  茅台 6
SUMMARY_TABLE_HINT = "主要会计数据和财务指标"
# 早于报告年度就触发。原本设为 2（只对「本期、上期之外」触发），
# 后来实测发现：汇总表在**任何年份**的朴素查询下都进不了前 20，
# 当年的题之所以还对，是模型从三大报表里也能答出来，属于侥幸。
# 上期（2024）的题则会失败——三大报表用「本期/上期」标列，
# 只有汇总表写着具体年份。因此改为 1。
# 当年（2025）不触发：那一列在哪张表里都是第一列，不需要额外引导。
YEARS_BEYOND_STATEMENTS = 1


def needs_summary_table(years: list[int], report_years: list[int]) -> bool:
    """问的年份是否早到只能从多年汇总表里取。"""
    if not years or not report_years:
        return False
    return min(years) <= max(report_years) - YEARS_BEYOND_STATEMENTS


async def keyword_query_of(question: str, codes: list[str], years: list[int] | None = None) -> str:
    """BM25 用的关键词查询：剥掉公司名与疑问词。

    公司名已经由 company_code 标量过滤处理过了，再留在关键词里纯属噪声——
    这份文档里每一页都写着「贵州茅台」，该词在文档内的区分度为零，
    却会把真正有区分度的词（年份、科目名）的权重稀释掉。

    实测对比（问 2023 年营业收入，答案在 p6「近三年主要会计数据」）：
        「贵州茅台2023年营业收入是多少？」 → p6 进不了前 20，结果拒答
        「2023年营业收入」                 → p6 排第 1
    稠密检索仍用完整问题，语义匹配不受这种词频稀释影响。

    没抽到公司代码时原样返回：此时过滤器兜不住，公司名还得留着。
    """
    from app.graph.nodes.query import available_corpus, unambiguous_aliases

    corpus = await available_corpus()
    report_years = [y for code, _n, y in corpus if not codes or code in codes]
    hint = f" {SUMMARY_TABLE_HINT}" if needs_summary_table(years or [], report_years) else ""
    if not codes:
        return question + hint
    # 只剥唯一别名。通名型别名（银行 / 医药 / 电力）剥掉的是查询里
    # 最有信息量的词——问招行的「银行业务收入」会被剥成「业务收入」
    names = {alias for alias, owner in unambiguous_aliases(corpus).items() if owner in codes}
    out = question
    # 长别名优先，否则先删掉「茅台」会把「贵州茅台」剩下半截
    for alias in sorted(filter(None, names), key=len, reverse=True):
        out = out.replace(alias, " ")
    for word in _FILLER:
        out = out.replace(word, " ")
    out = " ".join(out.split())
    return (out or question) + hint


async def cache_lookup_node(state: GraphState) -> GraphState:
    """查缓存。命中则带着答案直接走到结尾。

    放在查询分析之后：实体（公司、年份）要先解析出来，
    才能算 entity_key——语义缓存靠它做硬约束。
    """
    if state.get("refused"):
        return {}
    exp = experiment_of(state)
    raw = state.get("filters") or {}
    question = state.get("rewritten") or state["question"]
    ekey = entity_key(raw.get("company_codes") or [], raw.get("years") or [], question)

    hit = await get_exact(question, exp, ekey)
    source = "exact"
    if hit is None:
        hit = await get_semantic(milvus(), question, exp, ekey)
        source = "semantic"
    if not hit:
        return {}

    logger.info("缓存命中（%s）：%s", source, question[:30])
    return {
        "answer": hit.get("answer", ""),
        "citations": hit.get("citations", []),
        "cached": True,
        "degraded": [*(state.get("degraded") or []), f"cache_hit:{source}"],
    }


async def cache_store_node(state: GraphState) -> GraphState:
    """把答案写入两级缓存。

    拒答、空答、降级产生的答案一律不缓存——
    把一次偶发失败固化下来，比不缓存糟得多。
    """
    if state.get("cached") or state.get("refused"):
        return {}
    answer = (state.get("answer") or "").strip()
    if not answer:
        return {}
    degraded = state.get("degraded") or []
    if any(d.startswith(("agent_forced_answer", "agent_empty_answer")) for d in degraded):
        return {}

    exp = experiment_of(state)
    raw = state.get("filters") or {}
    question = state.get("rewritten") or state["question"]
    ekey = entity_key(raw.get("company_codes") or [], raw.get("years") or [], question)
    payload = {"answer": answer, "citations": state.get("citations") or []}
    await put_exact(question, exp, ekey, payload)
    await put_semantic(milvus(), question, exp, ekey, payload)
    return {}


async def retrieve_node(state: GraphState) -> GraphState:
    if state.get("refused"):
        return {}

    exp = experiment_of(state)
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
    keyword_query = await keyword_query_of(query, filters.company_codes, raw.get("years") or [])
    if keyword_query != query:
        logger.debug("关键词查询去噪：%s → %s", query, keyword_query)

    chunks = await retrieve(milvus(), query, filters, exp, keyword_query=keyword_query)

    # 问往年数据时，把「近三年主要会计数据」表按标签直接取回来，
    # 不和词频赛跑——那张表每个科目只出现一次，实测在朴素查询下
    # 一次都没进过前 20（lessons 7.16）。
    years = raw.get("years") or []
    from app.graph.nodes.query import available_corpus as _corpus

    report_years = [y for _c, _n, y in await _corpus()]
    if needs_summary_table(years, report_years):
        forced = await force_summary_table(milvus(), filters, exp)
        known = {c.chunk_uid for c in chunks}
        extra = [c for c in forced if c.chunk_uid not in known]
        if extra:
            logger.debug("强制召回汇总表 %d 条", len(extra))
            # 放在最前：它是往年数据唯一的权威来源，
            # 排在后面会被 top_n 截断扔掉——那正是原来失败的方式
            chunks = extra + chunks
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

    chunks, rerank_degraded = await rerank(chunks, query, exp)
    degraded.extend(rerank_degraded)
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

    exp = experiment_of(state)
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

    exp = experiment_of(state)
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
        "citations": build_citations(chunks),
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

    exp = experiment_of(state)
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
