"""语义缓存的实体硬约束测试。

缓存答错比不缓存糟得多：数字格式正确、引用齐全，
从输出上看不出它答的是另一年或另一个科目。
"""

from __future__ import annotations

import pytest

from app.rag.cache import entity_key, exact_key, metric_words


def test_entity_key_is_order_independent():
    """同一组实体只能有一种写法，否则缓存永远打不中。"""
    a = entity_key(["600519", "300750"], [2025, 2024], "营业收入")
    b = entity_key(["300750", "600519"], [2024, 2025], "营业收入")
    assert a == b


def test_entity_key_separates_years():
    """仅差年份的两个问题实测相似度 0.86，但不能只靠阈值挡。"""
    a = entity_key(["600519"], [2025], "营业收入是多少")
    b = entity_key(["600519"], [2024], "营业收入是多少")
    assert a != b


def test_entity_key_separates_metrics():
    """仅差科目实测相似度 0.9224，距 0.97 阈值只剩 0.05——最危险的一类。

    最初的 entity_key 只约束公司与年份，这两个问题的 key 完全相同。
    """
    a = entity_key(["600519"], [2025], "贵州茅台2025年营业收入是多少？")
    b = entity_key(["600519"], [2025], "贵州茅台2025年净利润是多少？")
    assert a != b


def test_entity_key_same_for_paraphrases():
    """问法不同、实体相同 → key 必须一致，才轮得到向量相似度发挥作用。"""
    a = entity_key(["600519"], [2025], "贵州茅台2025年销售费用是多少？")
    b = entity_key(["600519"], [2025], "贵州茅台2025年的销售费用为多少？")
    assert a == b


def test_metric_words_prefers_the_longer_match():
    """「营业总收入」与「营业收入」是不同科目，不能混。"""
    assert "营业总收入" in metric_words("宁德时代2025年营业总收入")


def test_unknown_metric_does_not_loosen_the_key():
    """认不出科目时退回公司+年份约束，不会比原来更宽。"""
    assert metric_words("贵州茅台2025年的某项神秘指标") == []


@pytest.mark.parametrize("exp_id", ["exp00_debug", "exp04_wider"])
def test_exact_key_is_scoped_by_experiment(exp_id):
    """不同实验的检索参数不同、答案不同，共用一个键会让消融互相污染。"""
    k = exact_key("贵州茅台2025年营业收入", exp_id, "c=600519|y=2025|m=营业收入")
    other = exact_key("贵州茅台2025年营业收入", "exp99", "c=600519|y=2025|m=营业收入")
    assert k != other
