"""评估集生成脚本的纪律测试。

评估集错了，所有指标都是假的——生成脚本比被测系统更需要测试。
"""

from __future__ import annotations

from scripts.build_eval_v2 import usable
from scripts.extract_key_metrics import corroborate


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


# ── 原文对账必须覆盖整张表的页码区间 ────────────────────


class _FakePdf:
    """只提供 get_text 与 page_count 的最小替身。"""

    def __init__(self, pages: list[str]) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def __getitem__(self, idx: int):
        text = self._pages[idx]
        return type("P", (), {"get_text": lambda self=None, _t=text: _t})()


def test_corroborate_finds_value_on_continuation_page():
    """跨页合并的表保留首页页码，目标行却常在续页上。

    实测茅台合并现金流量表 page_start=64，而「经营活动产生的现金流量净额」
    印在第 65 页。只查首页会把它判成对不上而整条丢弃，
    口径辨析题因此一道都生成不出来。
    """
    pdf = _FakePdf(["第64页没有这个数", "61,522,204,989.35 在第65页"])
    assert corroborate(pdf, 1, "61,522,204,989.35", 2) is True
    assert corroborate(pdf, 1, "61,522,204,989.35") is False


def test_corroborate_rejects_value_absent_from_the_table():
    """对账的作用是证伪：这个数压根不在这张表覆盖的页上。"""
    pdf = _FakePdf(["无关内容", "也无关"])
    assert corroborate(pdf, 1, "12,345.67", 2) is False


def test_corroborate_ignores_thousand_separators():
    """原文可能不带千分位，不能因为写法不同就判成对不上。"""
    pdf = _FakePdf(["61522204989.35"])
    assert corroborate(pdf, 1, "61,522,204,989.35") is True


# ── 留出集纪律 ──────────────────────────────────────────


def test_refuse_items_skip_companies_already_ingested():
    """已入库的公司不能再当「应拒答」占位。

    建 v3 时若把五粮液、比亚迪入库，v2 里那两道拒答题就失效了——
    所以选留出集公司时必须避开这些占位名。
    """
    from scripts.build_eval_v2 import OUT_OF_CORPUS, refuse_items

    all_names = {c for c, _ in OUT_OF_CORPUS}
    assert refuse_items(all_names) == []
    assert len(refuse_items(set())) == len(OUT_OF_CORPUS)


def test_out_of_corpus_placeholders_are_not_in_the_corpus():
    """占位公司一旦被下载进 data/raw，拒答题就名存实亡。

    这条测试是给未来的自己看的：扩语料时很容易顺手把五粮液加进去。

    2026-10-08 它真的抓到了一次——建 v5 时下载了比亚迪，
    而比亚迪当时正在 default 组里。失败方式很安静：出题脚本会
    自动跳过已入库的公司，不报错，只是拒答题凭空少一道。
    """
    from pathlib import Path

    from scripts.build_eval_v2 import REFUSE_SETS

    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not raw.exists():
        return  # CI 里没有语料，跳过
    downloaded = {p.name.split("_")[1] for p in raw.glob("*.pdf") if "_" in p.name}
    # 每一组都要查，不能只查 default：扩语料时撞上的可能是任意一组
    everyone = {c for group in REFUSE_SETS.values() for c, _ in group}
    clash = everyone & downloaded
    assert not clash, f"这些公司既是拒答占位又已入库：{clash}"
