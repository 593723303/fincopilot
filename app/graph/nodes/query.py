"""查询理解：指代消解 + 实体抽取 + 路由判定。

三件事合并为一次 LLM 调用。架构原设计是改写与路由两步，
但两者都用轻模型、都需要同样的上下文（会话历史 + 已入库公司），
合并后省一次往返与一次计费，判定质量没有损失（决策阶梯）。

路由的兜底方向是刻意的：置信度不足时走 agent 而非 rag。
agent 是 rag 的能力超集，判错只多花成本；反向判错会直接答错。
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config.experiment import load_experiment
from app.graph.state import GraphState
from app.observability.cost import TokenUsage, merge_usage, usage_from_response
from app.observability.tracing import callbacks, trace_metadata
from app.providers.registry import get_chat
from app.store.models import Document
from app.store.pg import session_factory

logger = logging.getLogger(__name__)

YEAR_PAT = re.compile(r"(20\d{2})\s*年")
# 不能用 \b：Python 的词边界基于 \w，而中文字符在 Unicode 模式下也算 \w，
# 所以「600519的净利润」里数字与汉字之间并没有词边界，\b(\d{6})\b 会漏匹配。
# 改用数字前后断言，既能贴着汉字匹配，又不会命中更长数字的片段。
CODE_PAT = re.compile(r"(?<!\d)(\d{6})(?!\d)")


class QueryAnalysis(BaseModel):
    """查询分析结果。"""

    rewritten: str = Field(description="指代消解后的完整问题，若无需改写则原样返回")
    company_codes: list[str] = Field(default_factory=list, description="涉及的股票代码，六位数字")
    years: list[int] = Field(default_factory=list, description="涉及的报告年度")
    statement_type: Literal["balance", "income", "cashflow", "equity"] | None = Field(
        None, description="明确指向某张报表时填写，否则留空"
    )
    route: Literal["chat", "rag", "agent"] = Field(description="问题应走的处理分支")
    confidence: float = Field(ge=0.0, le=1.0, description="路由判定的置信度")


_SYSTEM = """你是财报问答系统的查询分析器。根据对话历史与可用语料，完成三件事：

1. 指代消解：把「它」「该公司」「这一年」等代词还原为具体实体，输出完整问题。
2. 实体抽取：识别问题涉及的股票代码与报告年度。只能从「可用语料」中选择，
   语料里没有的公司不要臆造代码。
3. 路由判定：
   - chat  ：寒暄、询问系统能力，不需要查资料
   - rag   ：单一公司、单一年度的事实查询，一次检索即可回答
   - agent ：跨公司或跨年度对比、需要计算比率或趋势、需要作图

可用语料：
{corpus}

判定不确定时，route 填 agent 并给出较低的 confidence。"""


async def available_corpus() -> list[tuple[str, str, int]]:
    """已入库且就绪的语料清单，作为实体抽取的候选集。"""
    async with session_factory()() as session:
        rows = (
            await session.execute(
                select(Document.company_code, Document.company_name, Document.report_year)
                .where(Document.status == "ready")
                .order_by(Document.company_code, Document.report_year)
            )
        ).all()
    return [(r[0], r[1], r[2]) for r in rows]


def company_aliases(name: str) -> set[str]:
    """公司简称。

    用户几乎不会写全称：说「茅台」而非「贵州茅台」，说「宁德」而非
    「宁德时代新能源科技股份有限公司」。整名匹配会全数落空。
    """
    base = re.sub(r"(股份有限公司|有限公司|集团|控股|科技|新能源|股份)", "", name).strip()
    out = {name, base}
    if len(base) >= 4:
        out.add(base[-2:])  # 贵州茅台 → 茅台
        out.add(base[:2])  # 宁德时代 → 宁德
    return {a for a in out if len(a) >= 2}


def fallback_analysis(question: str, corpus: list[tuple[str, str, int]]) -> QueryAnalysis:
    """规则兜底：LLM 不可用时也要能检索。

    按公司名与年份做字面匹配，命中则走 rag，否则交给 agent。
    """
    codes = {c for c in CODE_PAT.findall(question)}
    for code, name, _year in corpus:
        if name and any(alias in question for alias in company_aliases(name)):
            codes.add(code)
    years = {int(y) for y in YEAR_PAT.findall(question)}
    known_years = {y for _c, _n, y in corpus}
    years &= known_years or years
    return QueryAnalysis(
        rewritten=question,
        company_codes=sorted(codes),
        years=sorted(years),
        statement_type=None,
        route="rag" if codes else "agent",
        confidence=0.5 if codes else 0.3,
    )


async def analyze_query(state: GraphState, config: RunnableConfig) -> GraphState:
    if state.get("refused"):
        return {}

    question = state["question"]
    corpus = await available_corpus()
    corpus_text = (
        "\n".join(f"- {code} {name} {year}年年报" for code, name, year in corpus) or "（暂无语料）"
    )

    exp = load_experiment()
    profile = exp.router.profile
    history = state.get("messages") or []

    try:
        llm = get_chat(profile).with_structured_output(QueryAnalysis, include_raw=True)
        messages = [
            SystemMessage(content=_SYSTEM.format(corpus=corpus_text)),
            *history[-6:],
            HumanMessage(content=question),
        ]
        out = await llm.ainvoke(
            messages,
            config=merge_configs(
                config,
                {
                    "callbacks": callbacks(),
                    "metadata": trace_metadata(node="analyze_query", profile=profile),
                },
            ),
        )
        analysis: QueryAnalysis = out["parsed"]
        raw = out.get("raw")
        usage = TokenUsage()
        if raw is not None:
            usage.add(profile, *usage_from_response(raw))
    except Exception as exc:
        logger.warning("查询分析失败，回退到规则匹配：%s", exc)
        analysis = fallback_analysis(question, corpus)
        usage = TokenUsage()

    # 置信度不足时走能力超集，宁可多花成本也不要答错
    route = analysis.route
    if analysis.confidence < exp.router.confidence_threshold and route != "chat":
        route = exp.router.fallback_branch

    merged = merge_usage(state.get("usage"), usage)

    logger.info(
        "查询分析：route=%s conf=%.2f codes=%s years=%s",
        route,
        analysis.confidence,
        analysis.company_codes,
        analysis.years,
    )
    return {
        "rewritten": analysis.rewritten,
        "route": route,
        "filters": {
            "company_codes": analysis.company_codes,
            "years": analysis.years,
            "statement_type": analysis.statement_type,
        },
        "usage": merged,
    }
