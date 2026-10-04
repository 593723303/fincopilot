"""检索链与查询理解的测试。

纯函数为主，不依赖运行中的存储与模型服务。
"""

from __future__ import annotations

import pytest

from app.config.experiment import Experiment, ParentExpansionCfg, load_experiment
from app.graph.build import branch_after_analyze, branch_after_grade
from app.graph.nodes.query import (
    QueryAnalysis,
    company_aliases,
    fallback_analysis,
    resolve_companies,
    unambiguous_aliases,
)
from app.rag.retrievers import QueryFilters, RetrievedChunk, build_expr, max_score

CORPUS = [
    ("600519", "贵州茅台", 2025),
    ("300750", "宁德时代新能源科技股份有限公司", 2025),
]


# ── 过滤表达式 ──────────────────────────────────────────


def test_expr_always_includes_strategy():
    """strategy 必须参与过滤，否则不同实验的块会混在一起（ADR-010）。"""
    expr = build_expr(QueryFilters(), "heading")
    assert 'strategy == "heading"' in expr


def test_expr_with_company_and_year():
    expr = build_expr(QueryFilters(company_codes=["600519"], years=[2025]), "fixed")
    assert 'company_code in ["600519"]' in expr
    assert "report_year in [2025, 2026, 2027]" in expr
    assert 'strategy == "fixed"' in expr


def test_report_years_expand_to_coverage():
    """report_year 是文档的报告年度，不是数据的年度。

    一份 2025 年报含 2023–2025 三年数据，按 report_year == 2024 过滤
    会直接返回空集——实测该缺陷导致约四分之一题目被误拒。
    """
    from app.rag.retrievers import expand_report_years

    assert expand_report_years([2024]) == [2024, 2025, 2026]
    assert expand_report_years([]) == []


def test_expr_for_past_year_includes_newer_reports():
    """问 2023 年数据时，必须能匹配到 2025 年的年报。"""
    expr = build_expr(QueryFilters(years=[2023]), "heading")
    assert "2025" in expr


def test_expr_multiple_companies():
    expr = build_expr(QueryFilters(company_codes=["600519", "300750"]), "heading")
    assert '"600519"' in expr and '"300750"' in expr


def test_expr_with_statement_type():
    expr = build_expr(QueryFilters(statement_type="income"), "heading")
    assert 'statement_type == "income"' in expr


def test_filters_is_empty():
    assert QueryFilters().is_empty() is True
    assert QueryFilters(years=[2025]).is_empty() is False


# ── 公司简称匹配 ────────────────────────────────────────


def test_aliases_cover_short_name():
    """用户说「茅台」而非「贵州茅台」，整名匹配会全数落空。"""
    assert "茅台" in company_aliases("贵州茅台")


def test_aliases_strip_corporate_suffix():
    aliases = company_aliases("宁德时代新能源科技股份有限公司")
    assert "宁德时代" in aliases
    assert "宁德" in aliases


def test_aliases_exclude_single_char():
    assert all(len(a) >= 2 for a in company_aliases("贵州茅台"))


# ── 规则兜底 ────────────────────────────────────────────


def test_fallback_matches_short_company_name():
    r = fallback_analysis("茅台2025年营业收入", CORPUS)
    assert "600519" in r.company_codes
    assert 2025 in r.years
    assert r.route == "rag"


def test_fallback_matches_stock_code():
    r = fallback_analysis("600519的净利润", CORPUS)
    assert "600519" in r.company_codes


def test_fallback_unknown_company_goes_to_agent():
    """识别不出主体时交给能力超集，而不是硬查一遍。"""
    r = fallback_analysis("帮我算个比率", CORPUS)
    assert r.company_codes == []
    assert r.route == "agent"


def test_fallback_is_valid_analysis():
    r = fallback_analysis("茅台2025年营收", CORPUS)
    assert isinstance(r, QueryAnalysis)
    assert 0.0 <= r.confidence <= 1.0


# ── 路由分支 ────────────────────────────────────────────


def test_branch_refused_goes_to_end():
    assert branch_after_analyze({"refused": True}) == "end"


def test_branch_chat():
    assert branch_after_analyze({"route": "chat"}) == "chat"


def test_branch_agent_goes_to_agent():
    """M5 起 Agent 分支已实现，不再降级到 rag。

    这条原本断言的是「降级并留痕」。M5 落地后它从「保护性断言」
    变成了「锁死旧行为」——改掉而不是删掉，是为了留下行为变更的痕迹。
    """
    state: dict = {"route": "agent", "degraded": []}
    assert branch_after_analyze(state) == "agent"
    assert "agent_not_available" not in state["degraded"]


def test_branch_after_grade_generates_when_relevant():
    exp = load_experiment("exp02_heading")
    state = {"relevance": exp.generation.relevance_threshold + 0.1, "retry_count": 0}
    assert branch_after_grade(state) == "generate"


def test_branch_after_grade_retries_when_low():
    state = {"relevance": 0.0, "retry_count": 1}
    assert branch_after_grade(state) == "retry"


def test_branch_after_grade_ends_when_exhausted():
    state = {"relevance": 0.0, "retry_count": 99}
    assert branch_after_grade(state) == "end"


# ── 父块还原 ────────────────────────────────────────────


def _chunk(uid: str, parent: str | None, text: str = "正文") -> RetrievedChunk:
    return RetrievedChunk(chunk_uid=uid, content=text, score=0.5, parent_uid=parent)


@pytest.mark.asyncio
async def test_expand_disabled_returns_as_is():
    from app.rag.retrievers import expand_parents

    exp = Experiment(exp_id="t", parent_expansion=ParentExpansionCfg(enabled=False))
    chunks = [_chunk("a", "p1"), _chunk("b", "p1")]
    out, degraded = await expand_parents(chunks, exp)
    assert out == chunks
    assert degraded == []


@pytest.mark.asyncio
async def test_expand_without_parent_uid_returns_as_is():
    from app.rag.retrievers import expand_parents

    exp = Experiment(exp_id="t", parent_expansion=ParentExpansionCfg(enabled=True))
    chunks = [_chunk("a", None), _chunk("b", None)]
    out, _ = await expand_parents(chunks, exp)
    assert len(out) == 2


# ── 配置一致性 ──────────────────────────────────────────


def test_hybrid_experiment_uses_lower_threshold():
    """RRF 分值量级远小于余弦相似度，阈值必须随融合方式调整，
    照搬 dense 的阈值会导致全部拒答。"""
    dense = load_experiment("exp01_baseline")
    hybrid = load_experiment("exp02_heading")
    assert dense.retrieval.mode == "dense"
    assert hybrid.retrieval.mode == "hybrid"
    assert hybrid.generation.relevance_threshold < dense.generation.relevance_threshold


def test_max_score_on_empty():
    assert max_score([]) == 0.0


def test_max_score_picks_highest():
    assert max_score([_chunk("a", None), RetrievedChunk("b", "x", 0.9)]) == 0.9


# ── 关键词查询去噪（run 0021 定位的检索失败） ──────────────


@pytest.mark.asyncio
async def test_keyword_query_strips_company_and_filler(monkeypatch):
    """公司名已由标量过滤处理，留在关键词里只会稀释年份与科目名的权重。"""
    from app.graph.nodes import rag as rag_mod

    async def fake_corpus():
        return [("600519", "贵州茅台", 2025)]

    monkeypatch.setattr("app.graph.nodes.query.available_corpus", fake_corpus)
    got = await rag_mod.keyword_query_of("贵州茅台2023年营业收入是多少？", ["600519"])
    assert "贵州茅台" not in got
    assert "是多少" not in got
    assert "2023" in got and "营业收入" in got


@pytest.mark.asyncio
async def test_keyword_query_kept_intact_without_company_code(monkeypatch):
    """没抽到公司代码时过滤器兜不住，公司名必须留着。

    注意这里仍要桩掉语料：为了判断「该年份是否只能从汇总表取数」，
    现在即使没有公司代码也会读一次语料清单。
    """
    from app.graph.nodes import rag as rag_mod

    async def fake_corpus():
        return [("600519", "贵州茅台", 2025)]

    monkeypatch.setattr("app.graph.nodes.query.available_corpus", fake_corpus)
    q = "贵州茅台2024年营业收入是多少？"
    assert await rag_mod.keyword_query_of(q, []) == q


# ── Agent：文本形式的工具调用必须被拦下（run 0025 实测缺陷） ──


@pytest.mark.parametrize(
    "text",
    [
        'calculate\n{"expression": "82320067101.68 - 82293107655.25"}\n</tool_call>',
        "<tool_call>retrieve_report",
        '{"company": "600519", "year": 2025}',
    ],
)
def test_malformed_tool_call_detected(text):
    """小模型会把工具调用当正文吐出来，不拦就会原样返给用户。"""
    from app.graph.nodes.agent import MALFORMED_CALL

    assert MALFORMED_CALL.search(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "贵州茅台2025年营业收入为168,838,102,514.79元。",
        "两者相差 26,959,446.43 元。",
        "该表格的 expression 列为空。",
    ],
)
def test_malformed_tool_call_does_not_misfire(text):
    from app.graph.nodes.agent import MALFORMED_CALL

    assert MALFORMED_CALL.search(text) is None


# ── 公司别名：扩到 12 家后暴露的通名问题 ──────────────────


CORPUS_12 = [
    ("600519", "贵州茅台", 2025),
    ("300750", "宁德时代", 2025),
    ("600036", "招商银行", 2025),
    ("601318", "中国平安", 2025),
    ("601857", "中国石油", 2024),
    ("600276", "恒瑞医药", 2025),
    ("600900", "长江电力", 2025),
    ("000651", "格力电器", 2025),
]


@pytest.mark.parametrize("word", ["银行", "医药", "电力", "电器", "中国"])
def test_generic_industry_words_are_not_aliases(word):
    """行业通名不能当公司标识。

    它们有两处危害：规则兜底会把「宁德时代在银行的存款」判成问招商银行；
    keyword_query_of 会把招行问题里的「银行」从 BM25 查询中剥掉。
    """
    assert word not in unambiguous_aliases(CORPUS_12)


def test_aliases_still_cover_the_natural_short_names():
    aliases = unambiguous_aliases(CORPUS_12)
    for short, code in [
        ("茅台", "600519"),
        ("宁德", "300750"),
        ("招商", "600036"),
        ("平安", "601318"),
        ("恒瑞", "600276"),
        ("格力", "000651"),
    ]:
        assert aliases.get(short) == code


def test_ambiguous_alias_is_dropped():
    """同一别名指向两家公司时，用它认公司必然有一半是错的。"""
    corpus = [("000001", "平安银行", 2025), ("601318", "中国平安", 2025)]
    assert "平安" not in unambiguous_aliases(corpus)


# ── 公司解析：模型不该负责抄六位代码 ──────────────────────


def _analysis(**kw) -> QueryAnalysis:
    base = dict(
        rewritten="q",
        companies=[],
        company_codes=[],
        years=[],
        statement_type=None,
        route="rag",
        confidence=0.9,
    )
    base.update(kw)
    return QueryAnalysis(**base)


def test_company_name_resolves_to_code():
    """模型输出公司名，代码由程序查表——名字好抄，代码不好抄。"""
    got = resolve_companies(_analysis(companies=["贵州茅台"]), CORPUS_12, "贵州茅台的营业收入")
    assert got == ["600519"]


def test_model_guessed_code_is_discarded():
    """问句里没出现过的代码一律当猜测丢弃。

    实测语料扩到 12 家后，问贵州茅台时模型把 company_codes 填成 600900
    （长江电力），茅台全部题目在别家年报里检索，整类题塌掉；
    而日志只显示「检索不到」，看不出是认错了公司。
    """
    a = _analysis(companies=["贵州茅台"], company_codes=["600900"])
    assert resolve_companies(a, CORPUS_12, "贵州茅台的营业收入") == ["600519"]


def test_code_written_by_the_user_is_trusted():
    """用户原话里就有代码时仍然采信。"""
    a = _analysis(company_codes=["600519"])
    assert resolve_companies(a, CORPUS_12, "600519的净利润是多少？") == ["600519"]


def test_unknown_company_resolves_to_nothing():
    """语料外的公司解析为空，后续据此拒答，而不是硬塞一家。"""
    a = _analysis(companies=["五粮液"])
    assert resolve_companies(a, CORPUS_12, "五粮液2025年营业收入") == []


# ── 更早年度只能从汇总表取数 ──────────────────────────────


def test_summary_hint_for_year_beyond_statements():
    """三大报表只有本期/上期两列，2023 年的数只在「近三年主要会计数据」里。"""
    from app.graph.nodes.rag import needs_summary_table

    assert needs_summary_table([2023], [2025]) is True


@pytest.mark.parametrize("year", [2024, 2025])
def test_no_summary_hint_for_current_or_prior_year(year):
    """本期与上期在三大报表里就有，不必把查询往汇总表上引。"""
    from app.graph.nodes.rag import needs_summary_table

    assert needs_summary_table([year], [2025]) is False


def test_summary_hint_needs_both_sides():
    from app.graph.nodes.rag import needs_summary_table

    assert needs_summary_table([], [2025]) is False
    assert needs_summary_table([2023], []) is False
