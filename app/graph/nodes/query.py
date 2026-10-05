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

from app.graph.state import GraphState, experiment_of
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
    # 让模型输出**公司名**而不是股票代码。模型抄名字很准，抄六位数字很不准：
    # 语料扩到 12 家后，实测「贵州茅台的股票代码是多少」被抽成 600900
    # （长江电力），茅台的题全在长江电力的年报里检索，整类题塌掉。
    # 名字→代码的映射是确定性的，交给程序做。
    companies: list[str] = Field(
        default_factory=list, description="问题涉及的公司名，必须逐字抄自「可用语料」列表"
    )
    company_codes: list[str] = Field(
        default_factory=list, description="若问题里直接写了六位股票代码就填，否则留空"
    )
    years: list[int] = Field(default_factory=list, description="涉及的报告年度")
    statement_type: Literal["balance", "income", "cashflow", "equity"] | None = Field(
        None, description="明确指向某张报表时填写，否则留空"
    )
    route: Literal["chat", "rag", "agent"] = Field(description="问题应走的处理分支")
    confidence: float = Field(ge=0.0, le=1.0, description="路由判定的置信度")


_SYSTEM = """你是财报问答系统的查询分析器。根据对话历史与可用语料，完成三件事：

1. 指代消解：把「它」「该公司」「这一年」等代词还原为具体实体，输出完整问题。
2. 实体抽取：识别问题涉及的**公司名**与报告年度。
   公司名必须逐字抄自下面「可用语料」里的名字，不要改写、不要翻译、
   不要自己推断股票代码——代码由系统查表得到。
   语料里没有的公司，companies 留空（后续会据此拒答）。
3. 路由判定：
   - chat  ：**仅限**寒暄与询问系统自身能力（「你好」「你能做什么」）。
             任何针对某家具体公司的提问都不是 chat，哪怕它不是财务数据——
             股票代码、公司全称、注册地址、邮政编码、联系电话、董秘姓名
             都写在年报的「公司简介」一节里，必须查资料，不得直接拒答
   - rag   ：答案能从年报的某一页**直接读到**，不需要做任何算术
   - agent ：答案必须**算**出来，或要多轮检索才能凑齐数据

   判定的唯一标准是「最终那个数字能不能从一页表格里直接抄下来」，
   **不是**看涉及几家公司、几个年度。只涉及一家公司一个年度的问题，
   同样可能需要 agent：

   - 「2025年营业收入是多少」          → rag（表里就是这个数）
   - 「2025年归母净利润占营收的比例」   → agent（要做除法）
   - 「四个季度营业收入合计」          → agent（要做加法）
   - 「归母净利润与扣非归母相差多少」   → agent（要做减法）
   - 「营业收入比2023年增长了多少」     → agent（要跨列取数再算增幅）
   - 「毛利率 / 净利率 / 同比 / 环比 / 占比 / 平均 / 相差 / 合计」 → 一律 agent

   反过来，下面这些**不要**判成 agent，它们都是直接读的：
   - 「营业收入变动的原因是什么」「销售费用为什么增加」 → rag
     （原因、说明、是什么、有哪些这类定性问题，答案是年报里的一段话）
   - 「2025年第一季度营业收入是多少」 → rag（年报有「分季度主要财务数据」表）
   - 「2025年营业收入同比上年增减了多少」 → rag
     （「主要会计数据」表里直接印着「本期比上年同期增减(%)」这一列）
   不要因为句子里出现「变动」「增减」「原因」就判成要做算术——
   年报里本来就印着这些数和这些话

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


# 截出来的两字别名里，这些是行业通名而非公司标识。
# 语料只有茅台和宁德时代时无所谓；扩到 12 家后，「招商银行」会产出别名
# 「银行」、「恒瑞医药」产出「医药」、「长江电力」产出「电力」、
# 「中国平安」产出「中国」——而别名有两处用途，两处都会被它们毁掉：
#   1. 规则兜底按别名认公司：问「宁德时代在银行的存款」会被判成问招商银行
#   2. keyword_query_of 把别名从 BM25 查询里剥掉：问招行的「银行业务收入」
#      会被剥成「业务收入」，把最有信息量的词删了
GENERIC_NAME_PARTS = frozenset(
    {
        "中国", "中华", "国际", "发展", "实业", "集团", "股份", "控股",
        "银行", "证券", "保险", "电力", "能源", "石油", "医药", "制药",
        "生物", "电器", "科技", "电子", "通信", "传媒", "化工", "汽车",
        "食品", "地产", "环保", "水泥", "物流", "建设", "工业",
    }
)


def company_aliases(name: str) -> set[str]:
    """公司简称。

    用户几乎不会写全称：说「茅台」而非「贵州茅台」，说「宁德」而非
    「宁德时代新能源科技股份有限公司」。整名匹配会全数落空。

    截短会撞上行业通名，因此过滤 GENERIC_NAME_PARTS；
    是否与同语料的其它公司冲突，由 unambiguous_aliases 再把一道关。
    """
    base = re.sub(r"(股份有限公司|有限公司|集团|控股|科技|新能源|股份)", "", name).strip()
    out = {name, base}
    if len(base) >= 4:
        out.add(base[-2:])  # 贵州茅台 → 茅台
        out.add(base[:2])  # 宁德时代 → 宁德
    return {a for a in out if len(a) >= 2 and a not in GENERIC_NAME_PARTS}


def unambiguous_aliases(corpus: list[tuple[str, str, int]]) -> dict[str, str]:
    """别名 → 公司代码，只保留在**当前语料内唯一**的别名。

    通名过滤是静态的，这一层是动态的：同一个别名指向两家公司时，
    用它去认公司必然有一半是错的，不如不认。
    语料越大越容易撞名，这道关只会越来越重要。
    """
    owners: dict[str, set[str]] = {}
    for code, name, _year in corpus:
        if not name:
            continue
        for alias in company_aliases(name):
            owners.setdefault(alias, set()).add(code)
    return {alias: next(iter(codes)) for alias, codes in owners.items() if len(codes) == 1}


def aliases_by_code(corpus: list[tuple[str, str, int]]) -> dict[str, set[str]]:
    """代码 → 该公司在本语料内**唯一**的别名集合。"""
    out: dict[str, set[str]] = {}
    for alias, code in unambiguous_aliases(corpus).items():
        out.setdefault(code, set()).add(alias)
    return out


def resolve_companies(
    analysis: QueryAnalysis,
    corpus: list[tuple[str, str, int]],
    question: str = "",
    history: str = "",
) -> list[str]:
    """把模型给出的公司名解析成股票代码。

    为什么不直接让模型输出代码：模型抄名字很准，抄六位数字很不准。
    语料扩到 12 家后实测「贵州茅台的股票代码是多少」被抽成 600900
    （长江电力），该公司全部题目在别家的年报里检索，整类题塌掉——
    而日志里只看到「检索不到」，看不出是认错了公司。

    名字→代码是确定性映射，程序做不会错。模型直接写了六位代码的
    （用户原话里就有代码）仍然采信，但必须在语料内。

    最后一道闸是「问句里必须真的出现过」。模型面对语料外的公司时，
    会**贴到名字最相近的已收录公司**——实测问「海天味业2025年营业收入」
    返回了海康威视的真实数字，数字真、引用真、格式对，
    用户没有任何办法发现自己拿到的是别家财报。而且它是概率性的：
    单独调用这个节点时能正确返回空，走完整图时就贴错了。

    所以不能靠提示词约束，只能用代码判定：
    解析出的每个公司，其别名必须在问句（或指代消解后的问句）里出现过，
    否则就是模型凭空补的，一律丢弃。
    """
    known = {code for code, _n, _y in corpus}
    aliases = unambiguous_aliases(corpus)
    # 只采信**用户原话里真的出现过**的代码。模型即使被告知不要推断代码，
    # 仍会往这个字段里填一个，而填错的概率不低——实测问茅台时它填 600900。
    # 问句里没有的代码一律当作猜测丢弃。
    haystack = f"{question} {history}"
    asked = set(CODE_PAT.findall(haystack))
    codes = {c for c in analysis.company_codes if c in known and c in asked}
    for name in analysis.companies:
        name = (name or "").strip()
        if not name:
            continue
        if name in known:  # 模型把代码填进了名字字段
            codes.add(name)
            continue
        if hit := aliases.get(name):
            codes.add(hit)
            continue
        # 名字没精确命中时退一步做包含匹配：模型可能写了全称或少写一个字
        for alias, code in aliases.items():
            if alias in name or name in alias:
                codes.add(code)
                break

    # 硬校验：留下的每家公司，都必须能在**用户写过的文本**里找到它的某个别名。
    #
    # 这里刻意不包含 analysis.rewritten。第一版把改写后的问句也算进来，
    # 结果完全拦不住——模型正是在「指代消解」这一步把公司名换掉的：
    # 问「海天味业2025年营业收入」，它改写成「海康威视2025年营业收入」，
    # 于是校验自然通过。**用模型的输出去校验模型的输出，等于没校验。**
    #
    # 代价是指代类追问（「它去年的呢」）要靠历史消息兜住，由调用方传入。
    by_code = aliases_by_code(corpus)
    kept = {c for c in codes if c in asked or any(a in haystack for a in by_code.get(c, ()))}
    if kept != codes:
        logger.info("丢弃问句中未出现的公司：%s", sorted(codes - kept))
    return sorted(kept)


def fallback_analysis(question: str, corpus: list[tuple[str, str, int]]) -> QueryAnalysis:
    """规则兜底：LLM 不可用时也要能检索。

    按公司名与年份做字面匹配，命中则走 rag，否则交给 agent。
    """
    codes = {c for c in CODE_PAT.findall(question)}
    # 用唯一别名而非全部别名：撞名的别名认出来的公司有一半是错的
    for alias, code in unambiguous_aliases(corpus).items():
        if alias in question:
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

    exp = experiment_of(state)
    profile = exp.router.profile
    # guard_in 已经把本轮提问追加进 messages，这里要去掉最后一条，
    # 否则同一个问题会在提示词里出现两次
    history = (state.get("messages") or [])[:-1]

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
    # 历史里只取用户自己说过的话。把助手的回答也算进来，
    # 等于又让模型的输出参与校验——助手上一轮提到过的公司会被当成用户问过。
    user_said = " ".join(
        str(getattr(m, "content", "")) for m in history if getattr(m, "type", "") == "human"
    )
    codes = resolve_companies(analysis, corpus, question, user_said)

    # 模型识别出了公司名，却一个都没能解析成已收录的公司——
    # 说明问的是语料外的公司。此时**必须拒答，不能把公司过滤留空**：
    # 留空会让检索退化成「全语料搜索」，捞回某一家的真实数字交给模型，
    # 于是问「海天味业2025年营业收入」会答出海康威视的 925 亿。
    # 数字真、引用真、格式对，用户无从发现拿到的是别家财报。
    if analysis.companies and not codes:
        logger.info("公司 %s 不在语料内，直接拒答", analysis.companies)
        return {
            "rewritten": analysis.rewritten,
            "route": route,
            "filters": {"company_codes": [], "years": analysis.years, "statement_type": None},
            "refused": True,
            "refuse_reason": "FC-2002",
            # 文案里不能复述 analysis.companies——它可能正是模型贴错的那个名字。
            # 实测问「海天味业」时这里会写成「未收录『海康威视』」，
            # 等于把模型的幻觉直接念给用户听。只说范围，不说它以为的是谁。
            "answer": (
                "您询问的公司不在已收录范围内，无法回答。当前可查询："
                + "、".join(sorted({n for _c, n, _y in corpus if n}))
                + "。"
            ),
            "usage": merged,
        }

    logger.info(
        "查询分析：route=%s conf=%.2f companies=%s codes=%s years=%s",
        route,
        analysis.confidence,
        analysis.companies,
        codes,
        analysis.years,
    )
    return {
        "rewritten": analysis.rewritten,
        "route": route,
        "filters": {
            "company_codes": codes,
            "years": analysis.years,
            "statement_type": analysis.statement_type,
        },
        "usage": merged,
    }
