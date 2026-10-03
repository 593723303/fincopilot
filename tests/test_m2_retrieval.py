"""检索链与查询理解的测试。

纯函数为主，不依赖运行中的存储与模型服务。
"""

from __future__ import annotations

import pytest

from app.config.experiment import Experiment, ParentExpansionCfg, load_experiment
from app.graph.build import branch_after_analyze, branch_after_grade
from app.graph.nodes.query import QueryAnalysis, company_aliases, fallback_analysis
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
    assert "report_year in [2025]" in expr
    assert 'strategy == "fixed"' in expr


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


def test_branch_agent_degrades_to_rag_with_mark():
    """M5 之前没有 Agent 能力，降级要留痕以便评估时区分。"""
    state: dict = {"route": "agent", "degraded": []}
    assert branch_after_analyze(state) == "rag"
    assert "agent_not_available" in state["degraded"]


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
