"""PDF 解析器测试。

主体是纯函数测试，不依赖语料，可在 CI 中运行；
末尾的集成测试在 data/raw 无 PDF 时自动跳过。

这里的每个用例都对应一个真实踩过的坑，不是凭空设计的边界条件。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.pdf_parser import (
    PROSE_UNIT_PAT,
    STATEMENT_SCOPE,
    TableContext,
    inline_unit,
    is_numeric_table,
    looks_tabular,
    normalize_cell,
    parse_pdf,
    render_table,
    scale_to_yuan,
)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


# ── 单元格规范化 ────────────────────────────────────────


def test_normalize_removes_internal_newline():
    """「归属于上市公司股东的\n净利润」不处理会与无换行版本被当成两个科目。"""
    assert normalize_cell("归属于上市公司股东的\n净利润") == "归属于上市公司股东的净利润"


def test_normalize_handles_fullwidth_space():
    assert normalize_cell("营业　收入") == "营业 收入"


def test_normalize_none_and_empty():
    assert normalize_cell(None) == ""
    assert normalize_cell("   ") == ""


# ── 数值表判定（bug1：文字型表被标了量纲）────────────────


def test_text_table_is_not_numeric():
    """备查文件目录这类文字表不该被标注量纲。"""
    rows = [
        ["备查文件目录", "载有公司负责人签名并盖章的会计报表"],
        ["", "载有会计师事务所盖章的审计报告原件"],
    ]
    assert is_numeric_table(rows) is False


def test_financial_table_is_numeric():
    rows = [
        ["", "第一季度", "第二季度"],
        ["营业收入", "50,600,957,885.78", "38,788,396,531.06"],
        ["净利润", "26,847,474,238.76", "18,555,488,059.34"],
    ]
    assert is_numeric_table(rows) is True


def test_tiny_table_is_not_numeric():
    """内容太少时不做判断，避免误判。"""
    assert is_numeric_table([["合计", "1"]]) is False


# ── 表内单位声明（bug2：单位写在表格里而非上方）──────────


def test_inline_unit_detected_and_row_skipped():
    """持股表把「单位：股」写在表格第一行，必须提取并跳过该行。"""
    rows = [["单位：股", "", ""], ["姓名", "职务", "年初持股数"], ["陈华", "董事", "1,000"]]
    unit, currency, skip = inline_unit(rows)
    assert unit == "股"
    assert skip == 1


def test_inline_unit_keeps_row_with_real_content():
    """若该行还有实义内容，则不能整行丢弃。"""
    rows = [["项目（单位：万元）", "金额"], ["营业收入", "1,234"]]
    unit, _, skip = inline_unit(rows)
    assert unit == "万元"
    assert skip == 0


def test_inline_unit_absent():
    rows = [["项目", "金额"], ["营业收入", "1,234"]]
    assert inline_unit(rows) == (None, None, 0)


def test_non_amount_unit_recognized():
    """「股」「%」等非金额单位必须能识别，否则会被金额量纲错误覆盖。"""
    for text, expected in (("单位：股", "股"), ("单位：%", "%"), ("单位：吨", "吨")):
        unit, _, _ = inline_unit([[text, ""], ["a", "1"]])
        assert unit == expected


# ── 量纲折算 ────────────────────────────────────────────


def test_scale_to_yuan():
    """跨公司对比的前提：茅台用元、宁德用千元，不折算会差 1000 倍。"""
    assert scale_to_yuan(1.0, "元") == 1
    assert scale_to_yuan(1.0, "千元") == 1_000
    assert scale_to_yuan(1.0, "万元") == 10_000
    assert scale_to_yuan(1.0, "亿元") == 100_000_000


def test_scale_rejects_non_amount_unit():
    """「股」不是金额量纲，不能折算成元。"""
    assert scale_to_yuan(1.0, "股") is None
    assert scale_to_yuan(1.0, None) is None


# ── 渲染（bug3：单位必须进正文）──────────────────────────


def test_render_puts_unit_in_text():
    """模型只读文本，单位只存元数据等于没有。"""
    rows = [["项目", "金额"], ["营业收入", "1,234"]]
    out = render_table(rows, TableContext(unit="万元", currency="人民币"), "第三节 > 一、概况")
    assert "单位：万元" in out.splitlines()[0]
    assert "币种：人民币" in out.splitlines()[0]
    assert "| 项目 | 金额 |" in out


def test_render_without_unit_has_no_unit_line():
    out = render_table([["A", "B"], ["c", "d"]], TableContext(), "")
    assert "单位" not in out


# ── 集成：真实年报 ──────────────────────────────────────

_pdfs = sorted(RAW_DIR.glob("*.pdf")) if RAW_DIR.exists() else []
needs_corpus = pytest.mark.skipif(
    not _pdfs, reason="data/raw 无语料，先执行 python -m scripts.fetch_reports"
)


@needs_corpus
def test_parse_real_report_produces_blocks():
    doc = parse_pdf(_pdfs[0], max_pages=30)
    assert doc.page_count == 30
    assert doc.table_blocks, "真实年报前 30 页应当含表格"
    assert all(b.heading_path is not None for b in doc.blocks)


@needs_corpus
def test_unit_is_written_into_block_text():
    """解析出单位的表，正文首行必须带量纲声明 —— 模型只读文本。"""
    doc = parse_pdf(_pdfs[0], max_pages=40)
    tagged = [t for t in doc.table_blocks if t.unit]
    assert tagged, "前 40 页应当存在带量纲的数值表"
    for t in tagged:
        first = t.text.splitlines()[0]
        assert "【单位：" in first, f"p{t.page_start} 的量纲未写进正文：{first[:40]}"


@needs_corpus
def test_text_tables_not_tagged_with_unit():
    """文字型表格不应被标注量纲（备查文件目录、释义表等）。"""
    doc = parse_pdf(_pdfs[0], max_pages=40)
    untagged = [t for t in doc.table_blocks if not t.unit]
    # 真实年报前 40 页必然混有文字型表格；若一个都没有，
    # 说明兜底逻辑又在无条件给所有表打量纲了
    assert untagged, "所有表都被标了量纲，文字型表的过滤可能已失效"


@needs_corpus
def test_unit_source_recorded_in_flags():
    """单位来源必须记录：inherited 的可信度远低于 table_inline。"""
    doc = parse_pdf(_pdfs[0], max_pages=40)
    tagged = [t for t in doc.table_blocks if t.unit]
    assert tagged
    for t in tagged:
        assert any(f.startswith("unit_src:") for f in t.flags)


# ── 报表口径（合并 / 母公司） ────────────────────────────


@pytest.mark.parametrize(
    "line",
    ["合并现金流量表", "母公司资产负债表", "合并所有者权益变动表", "母公司利润表"],
)
def test_statement_scope_recognized(line):
    assert STATEMENT_SCOPE.search(line) is not None


@pytest.mark.parametrize("line", ["现金流量表补充资料", "分季度主要财务数据", "合并范围变更"])
def test_statement_scope_ignores_other_titles(line):
    assert STATEMENT_SCOPE.search(line) is None


def test_scope_written_into_table_text():
    """口径必须写进正文。

    检索返回的是文本，模型看不到元数据——口径只存元数据里，
    合并与母公司的数字就会被混用（实测两者经营现金流净额相差近一倍）。
    """
    ctx = TableContext(unit="元", scope="母公司现金流量表")
    out = render_table([["项目", "2025年度"], ["经营活动现金流量净额", "1"]], ctx, "第八节 > 二、财务报表")
    assert "母公司现金流量表" in out.splitlines()[1]


def test_scope_replaces_heading_fallback():
    """有口径时不再回落到章节名，避免同一位置出现两个标题。"""
    ctx = TableContext(unit="元", scope="合并利润表")
    out = render_table([["项目", "本期"]], ctx, "第八节财务报告 > 二、财务报表")
    assert "二、财务报表" not in out


# ── 括号式单位声明（银行 / 保险年报的写法） ──────────────


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("（人民币百万元，特别注明除外）", "百万元"),
        ("（除特别注明外，货币单位均以人民币百万元列示）", "百万元"),
        ("（除特别注明外，金额单位为人民币百万元）", "百万元"),
        ("（人民币千元）", "千元"),
    ],
)
def test_prose_unit_declaration(line, expected):
    """招商银行前 120 页 39 张表，单位无一被「单位：X」识别到。"""
    m = PROSE_UNIT_PAT.search(line)
    assert m is not None and m.group(1) == expected


@pytest.mark.parametrize(
    "line",
    [
        "人民币7,159,767百万元，占总资产的54.78%",
        "（本公司于2024年完成重组）",
        "（简称「公司」）",
    ],
)
def test_prose_unit_ignores_amounts_in_prose(line):
    """括号里带数字的是具体金额，不是整张表的量纲声明。"""
    assert PROSE_UNIT_PAT.search(line) is None


def test_looks_tabular_needs_grouped_numbers():
    """页码、年份到处都是，带千分位的数字基本只出现在金额表里。"""
    assert looks_tabular("1,745,679 " * 12) is True
    assert looks_tabular("2025年年度报告 第 183 页 公司于2024年完成重组") is False


# ── 三年并列表按年份展开 ────────────────────────────────


def _three_year_rows():
    return [
        ["主要会计数据", "2025年", "2024年", "本期比上年同期增减(%)", "2023年"],
        ["营业收入", "168,838,102,514.79", "170,899,152,276.34", "-1.21", "147,693,604,994.14"],
    ]


def test_year_breakdown_pairs_each_value_with_its_year():
    """把列对齐这件事在解析阶段做完，模型只需字符串匹配。

    实测模型经常数错列——问 2024 年给 2025 年的数，flash 与 plus 都会错，
    加提示词规则也压不住。
    """
    out = render_table(_three_year_rows(), TableContext(unit="元"), "x")
    assert "【按年份】营业收入：2025年=168,838,102,514.79" in out
    assert "2024年=170,899,152,276.34" in out
    assert "2023年=147,693,604,994.14" in out


def test_year_breakdown_skips_the_change_rate_column():
    """「本期比上年同期增减(%)」列头里也写着年份，但它不是某一年的数值列。"""
    out = render_table(_three_year_rows(), TableContext(unit="元"), "x")
    assert "=-1.21" not in out


def test_no_breakdown_when_columns_are_relative():
    """只有「本期/上期」的表没有年份可对齐，不输出——宁可没有，不能给错的。"""
    rows = [["项目", "本期发生额", "上期发生额"], ["营业收入", "1,000", "2,000"]]
    assert "【按年份】" not in render_table(rows, TableContext(unit="元"), "x")


def test_no_breakdown_when_row_has_too_few_numbers():
    """数值不足两个就对不出年份，整行跳过。"""
    rows = [
        ["主要会计数据", "2025年", "2024年", "2023年"],
        ["是否适用", "是", "", ""],
    ]
    assert "【按年份】" not in render_table(rows, TableContext(), "x")


def test_no_breakdown_when_a_year_spans_two_columns():
    """资产负债表的「2025年末 / 2025年初」两列都归到 2025，展开出来有歧义。

    「2025年=43,904,550；2025年=10,000,000」读的人无从判断哪个是哪个——
    歧义的输出比没有更糟。
    """
    rows = [
        ["项目", "2025年12月31日", "2025年1月1日", "2024年12月31日"],
        ["短期借款", "43,904,550", "10,000,000", "31,008,549"],
    ]
    assert "【按年份】" not in render_table(rows, TableContext(unit="千元"), "x")
