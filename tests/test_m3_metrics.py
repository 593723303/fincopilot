"""评估指标测试。

用例大多直接取自真实评估运行中出现过的回答——
评估器自身的缺陷会直接伪造指标（实测把误答率虚高到 50%），
所以这些判定规则必须被测试锁住。
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    bootstrap_ci,
    extract_amounts,
    fact_match,
    intervals_overlap,
    is_refusal,
    numeric_match,
    recall_at_k,
    stat,
    to_yuan,
    unit_declared,
)

# ── 量纲归一 ────────────────────────────────────────────


def test_to_yuan_scales():
    assert to_yuan(1, "元") == 1
    assert to_yuan(1, "千元") == 1e3
    assert to_yuan(1, "万元") == 1e4
    assert to_yuan(1, "亿元") == 1e8


def test_to_yuan_unknown_unit_returns_none():
    """量纲未知时不猜测——猜错会让错误答案被判成正确。"""
    assert to_yuan(1, "股") is None
    assert to_yuan(1, None) is None


def test_extract_amounts_handles_thousand_separator():
    out = extract_amounts("营业收入为 50,600,957,885.78 元")
    assert (50600957885.78, "元") in out


def test_extract_amounts_picks_up_both_forms():
    """模型常同时给原值与换算值，两者都要能参与匹配。"""
    out = extract_amounts("研发投入为 22,146,581 千元（即 221.47 亿元）")
    values = {v for v, _ in out}
    assert 22146581.0 in values
    assert 221.47 in values


# ── 数值判定 ────────────────────────────────────────────


def test_numeric_match_exact():
    ok, declared, _ = numeric_match("营业收入为 50,600,957,885.78 元", 50600957885.78, "元")
    assert ok is True
    assert declared is True


def test_numeric_match_across_units():
    """标准答案用元，模型用万元作答，归一后应判对。"""
    ok, _, _ = numeric_match("营业收入为 16,883,810.25 万元", 168838102514.79, "元")
    assert ok is True


def test_numeric_match_within_tolerance():
    ok, _, _ = numeric_match("约 168,838,000,000 元", 168838102514.79, "元", tolerance=0.005)
    assert ok is True


def test_numeric_match_rejects_wrong_value():
    ok, _, detail = numeric_match("研发投入为 109,689,625.09 元", 674657764.92, "元")
    assert ok is False
    assert "109689625" in detail.replace(",", "")


def test_numeric_match_reports_nearest_not_first():
    """回答里常混有年份等无关数字，报告「最接近」不能拿第一个候选。"""
    answer = "2025年该指标为 674,657,000.00 元"
    ok, _, detail = numeric_match(answer, 999999999.0, "元")
    assert ok is False
    assert "2025" not in detail.split("最接近")[-1].split("，")[0]


def test_numeric_match_no_number():
    ok, declared, detail = numeric_match("资料中未提及", 100.0, "元")
    assert ok is False
    assert declared is False


# ── 拒答判定 ────────────────────────────────────────────


def test_system_refusal_flag_wins():
    assert is_refusal("任意内容", True) is True


@pytest.mark.parametrize(
    "answer",
    [
        "所提供资料中未找到相关信息。",
        "我无法提供投资建议。作为财报分析助手，我可以帮助您解读财务报表。",
        "我无法提供腾讯控股2025年的营业收入预测，因为这属于未来数据。",
        "已收录的年报中没有检索到与该问题相关的内容。",
    ],
)
def test_detects_real_refusals(answer):
    assert is_refusal(answer, False) is True


@pytest.mark.parametrize(
    "answer",
    [
        # 正确回答，末尾附带能力说明——全文匹配会把它误判为拒答
        "贵州茅台的股票代码是600519。\n\n我可以帮助您分析财报数据，但无法提供实时行情或编造财务信息。",
        "贵州茅台2025年营业收入为 168,838,102,514.79 元 [1]。如需其他指标我无法直接给出，请补充说明。",
    ],
)
def test_answer_with_trailing_disclaimer_is_not_refusal(answer):
    """真正的拒答开门见山；末尾的免责声明不构成拒答。"""
    assert is_refusal(answer, False) is False


# ── 事实与量纲声明 ──────────────────────────────────────


def test_fact_match_requires_all_keys():
    ok, _ = fact_match("主要是酱香系列酒产品结构调整影响", "酱香系列酒产品结构调整")
    assert ok is True


def test_fact_match_fails_when_key_missing():
    ok, _ = fact_match("主要是产品涨价", "酱香系列酒产品结构调整")
    assert ok is False


def test_unit_declared_accepts_exact_unit():
    assert unit_declared("营业收入为 50,600,957,885.78 元", "元") is True


def test_unit_declared_fails_when_missing():
    assert unit_declared("营业收入为 50600957885.78", "千元") is False


def test_unit_declared_when_no_unit_expected():
    assert unit_declared("股票代码是600519", None) is True


# ── 检索指标 ────────────────────────────────────────────


def test_recall_hits():
    uids = ["600519_2025_annual-heading-L1-00007-abc"]
    assert recall_at_k(uids, ["600519_2025_annual"]) == 1.0


def test_recall_miss():
    assert recall_at_k(["300750_2025_annual-x"], ["600519_2025_annual"]) == 0.0


def test_recall_none_when_no_expectation():
    """没有标注应命中文档时不参与统计，而不是记 0 分。"""
    assert recall_at_k(["x"], None) is None


# ── 统计 ────────────────────────────────────────────────


def test_bootstrap_is_reproducible():
    """同一批数据必须得到相同区间，否则无法判断指标变化来自改动还是随机性。"""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert bootstrap_ci(values, rounds=200) == bootstrap_ci(values, rounds=200)


def test_bootstrap_interval_contains_mean():
    values = [1.0] * 7 + [0.0] * 3
    mean, lo, hi = bootstrap_ci(values, rounds=500)
    assert lo <= mean <= hi


def test_small_sample_gives_wide_interval():
    """小样本的区间必须足够宽——这正是评估集要扩到数百条的理由。"""
    _, lo, hi = bootstrap_ci([1.0, 0.0, 1.0], rounds=500)
    assert hi - lo > 0.5


def test_overlapping_intervals_mean_no_improvement():
    a = stat("a", [1.0, 0.0, 1.0, 1.0], rounds=300)
    b = stat("b", [1.0, 1.0, 0.0, 1.0], rounds=300)
    assert intervals_overlap(a, b) is True


def test_bootstrap_empty():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
