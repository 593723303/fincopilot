"""评估集生成脚本的纪律测试。

评估集错了，所有指标都是假的——生成脚本比被测系统更需要测试。
"""

from __future__ import annotations

from scripts.build_eval_v2 import usable


def _row(metric: str, year: int, value: float, *, corroborated=True, unit="元") -> dict:
    return {
        "metric": metric,
        "year": year,
        "value": value,
        "raw": f"{value:,.2f}",
        "unit": unit,
        "page": 6,
        "heading": "第二节 > 七、近三年主要会计数据",
        "source_line": "| x |",
        "corroborated": corroborated,
    }


def test_drops_rows_not_corroborated():
    """没在原始页面文本里对上的行不能出题。

    底稿与被评估系统共用同一个解析器，不独立对账的话，
    解析器错了标准答案与系统回答会一起错，评估照样全绿。
    """
    rows = [_row("营业收入", 2025, 1.0, corroborated=False), _row("营业收入", 2024, 2.0)]
    assert [r["year"] for r in usable(rows)] == [2024]


def test_drops_rows_without_unit():
    """没有量纲的数字出不了题：读者分不清是元还是万元。"""
    rows = [_row("营业收入", 2025, 1.0, unit=None), _row("营业收入", 2024, 2.0)]
    assert [r["year"] for r in usable(rows)] == [2024]


def test_drops_whole_group_when_values_repeat_across_years():
    """同一科目跨年数值相同，年份就分不出来，整组弃用。

    这类题答对答错无法区分：模型随便报一年都"对"。
    """
    rows = [_row("总资产", 2025, 100.0), _row("总资产", 2024, 100.0)]
    assert usable(rows) == []


def test_keeps_other_metrics_when_one_group_is_dropped():
    """弃用是按科目分组的，不能牵连别的科目。"""
    rows = [
        _row("总资产", 2025, 100.0),
        _row("总资产", 2024, 100.0),
        _row("营业收入", 2025, 10.0),
        _row("营业收入", 2024, 20.0),
    ]
    assert {r["metric"] for r in usable(rows)} == {"营业收入"}
