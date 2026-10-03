"""PDF 解析器测试。

主体是纯函数测试，不依赖语料，可在 CI 中运行；
末尾的集成测试在 data/raw 无 PDF 时自动跳过。

这里的每个用例都对应一个真实踩过的坑，不是凭空设计的边界条件。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.pdf_parser import (
    TableContext,
    inline_unit,
    is_numeric_table,
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
