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
    HeadingTracker,
    TableContext,
    WordTable,
    count_merged_cells,
    inline_unit,
    is_numeric_table,
    is_summary_table,
    looks_tabular,
    merge_wrapped_labels,
    multi_value_rows,
    normalize_cell,
    parse_pdf,
    render_table,
    scale_to_yuan,
    year_label_count,
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


# ── 无边框表格的坐标重建（P1-5 / P1-8） ──────────────────


def test_wrapped_label_merged_back():
    """折成两行的科目名要并回数据行，整词必须能完整出现。

    取自广发证券年报第 19 页的真实版面：科目名折断后，
    「经营活动产生的现金流量净额」这串字在整块里一次都不出现，
    BM25 因此打不中，检索转而选了列宽些、科目名完整的母公司表，
    最终答出「未找到」——而数据就在这张表里。
    """
    rows = [
        ["项目", "2025 年", "2024 年", "2023 年"],
        ["", "调整前", "调整后", "调整后"],
        ["经营活动产生的现金流", "-27,780,960,722.97", "9,970,809,011.81", "-8,918,975,156.38"],
        ["量净额（元）", "", "", ""],
    ]
    out = merge_wrapped_labels(rows)
    assert len(out) == 3
    assert out[2][0] == "经营活动产生的现金流量净额（元）"
    # 子列头行紧跟在表头之后，上一行没有数值格，不能被并走
    assert out[1][1] == "调整前"


def test_wrapped_label_merge_leaves_real_rows_alone():
    """三类容易误伤的行都不能动。"""
    # ① 占位符撑起来的真数据行：`-` 不是中文，整行不算续行
    rows = [
        ["营业收入", "1,000.00", "2,000.00"],
        ["其他收益", "-", "-"],
    ]
    assert merge_wrapped_labels(rows)[1][0] == "其他收益"

    # ② 多格子列头（五格），超过两格的上限
    rows = [
        ["营业总收入", "35,492,783,045.20", "27,198,789,118.97"],
        ["调整前", "调整后", "调整后"],
    ]
    assert len(merge_wrapped_labels(rows)) == 2

    # ③ 表头里的「2025 年」不是数值，其后的子列头不会被并进表头
    rows = [["项目", "2025 年", "2024 年"], ["增减", "", ""]]
    assert len(merge_wrapped_labels(rows)) == 2


def test_merged_cell_count_ignores_long_single_numbers():
    """一个长数字不是两个数。

    早先的正则「两个千分位数之间夹任意字符」会因回溯把
    349,079,082,852 拆成 349,079 与 082,852，于是位数够多的正常数字
    全被判成粘连，紫金矿业的坐标重建因此一直被拒绝。
    """
    assert count_merged_cells([["349,079,082,852"]]) == 0
    assert count_merged_cells([["168,838,102,514.79"]]) == 0


def test_merged_cell_count_catches_real_merges():
    """中间夹着小数的真粘连要认出来。"""
    assert count_merged_cells([["303,639,957,153 14.96 293,403,242,878"]]) == 1
    assert count_merged_cells([["337,488 0.01 339,123"]]) == 1


def test_merged_cell_count_can_skip_the_catch_all_column():
    """重建表的第 0 列是兜底列，会收走带数字的正文行，不该计入。"""
    rows = [["正文 1,234 和 5,678", "100,000"]]
    assert count_merged_cells(rows) == 1
    assert count_merged_cells(rows, skip_first_col=True) == 0


def test_year_label_count_takes_max_per_row_not_total():
    """按行取最大，不能把几行的年份标签加总。

    坐标重建按 y 切行，会把「2025年」「2024年」「2023年」拆到不同行——
    总数没变，表头却已经不可用。实测中国建筑就是这样被毁掉的。
    """
    intact = [["主要会计数据", "2025年", "2024年", "2023年"]]
    broken = [["本期比上年", "2024年"], ["2025年"], ["2023年"]]
    assert year_label_count(intact) == 3
    assert year_label_count(broken) == 1


def test_decimal_section_numbering_is_recognized():
    """招商银行用「第二章」与「2.1」编号，只认「第X节」会让整篇没有标题路径。"""
    t = HeadingTracker()
    assert t.feed("第二章 会计数据和财务指标摘要") is True
    assert t.feed("2.1 本集团主要会计数据和财务指标") is True
    assert "2.1" in t.path()


def test_bare_decimal_is_not_a_heading():
    """孤立的「2.1」不是标题，后面必须跟实义文字。"""
    t = HeadingTracker()
    assert t.feed("2.1") is False
    assert t.feed("1.23") is False


def test_year_header_from_above_needs_two_years():
    """从表格上方捞表头：找不到两个以上年份就返回空，不猜。"""
    from app.rag.pdf_parser import YEAR_LABEL

    # 这个判据本身很简单，锁住的是「不足两个年份不采用」这条纪律
    assert len(YEAR_LABEL.findall("2025年 2024年 增减(%) 2023年")) == 3
    assert len(YEAR_LABEL.findall("本集团主要会计数据和财务指标")) == 0
    assert len(YEAR_LABEL.findall("2025年度报告（A股）")) == 1


def test_multi_value_rows_counts_real_table_rows():
    """「一行里有两个以上数值」是判断「这是不是一张表」的直接判据。"""
    rows = [
        ["项目", "2025 年", "2024 年"],
        ["总资产", "13,898,471", "12,957,827"],
        ["总负债", "12,482,483", "11,653,115"],
        ["注：本公司对非经常性损益项目的确认依照规定执行", "", ""],
    ]
    # 表头的「2025 年」不是数值格，正文说明也不是——只有两行真数据
    assert multi_value_rows(rows) == 2


def test_word_table_quacks_like_pymupdf_table():
    """拼出来的表要能冒充 pymupdf 的 Table 接上原流程。"""
    t = WordTable((0.0, 1.0, 2.0, 3.0), [["a", "1,000"], ["b", "2,000"]])
    assert t.extract() == [["a", "1,000"], ["b", "2,000"]]
    assert t.bbox == (0.0, 1.0, 2.0, 3.0)


def test_summary_table_detected_from_header_row():
    """标题落在表第一行时也要认出汇总表。

    按词坐标拼出来的表会把表格上方的标题收进第一格的兜底列，
    于是标题既不在 heading 里也不在 caption 里。中国平安第 14 页就是这样：
    caption 被取成紧挨着的「12月31日」，真标题躺在表的第一行。
    """
    rows = [
        ["（人民币百万元） 财务摘要 主要会计数据及财务指标", "2025年", "2024年", "2023年"],
        ["总资产", "13,898,471", "12,957,827", "11,583,417"],
    ]
    assert is_summary_table("", "12月31日", rows)
    # 标题不出现时不能误判
    assert not is_summary_table("", "12月31日", [["项目", "2025年", "2024年"], ["总资产", "1", "2"]])
