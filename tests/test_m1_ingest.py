"""入库链路与接口的单元测试。

不依赖运行中的服务与外部存储；真实的端到端链路在开发时手工验证过
（上传 → 队列 → Worker → 双写 → 可检索）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.documents import sanitize_filename
from app.config.experiment import ChunkingCfg
from app.rag.chunker import ChunkUnit, est_tokens, make_uid, split_table, with_heading
from app.rag.indexer import infer_meta, truncate_utf8
from app.rag.pdf_parser import ParsedBlock

# ── 安全：路径穿越防护 ──────────────────────────────────


@pytest.mark.parametrize(
    "evil",
    [
        "../../../etc/passwd.pdf",
        "..\\..\\windows\\system32\\evil.pdf",
        "/absolute/path/x.pdf",
        "C:\\Windows\\x.pdf",
        "dir/sub/x.pdf",
    ],
)
def test_sanitize_strips_any_path(evil):
    """上传文件名完全由客户端控制，必须只保留基础名。"""
    name = sanitize_filename(evil)
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name


@pytest.mark.parametrize("bad", ["", None, ".", "..", ".hidden"])
def test_sanitize_rejects_invalid(bad):
    with pytest.raises(ValueError):
        sanitize_filename(bad)


def test_sanitize_keeps_normal_name():
    name = "600519_贵州茅台_贵州茅台2025年年度报告.pdf"
    assert sanitize_filename(name) == name


# ── 文件名推断 ──────────────────────────────────────────


def test_infer_meta_from_filename():
    code, name, year = infer_meta(Path("600519_贵州茅台_贵州茅台2025年年度报告.pdf"))
    assert (code, name, year) == ("600519", "贵州茅台", 2025)


def test_infer_meta_rejects_unknown_name():
    """推断失败必须报错而非猜默认值：错误的 doc_key 会污染检索过滤。"""
    with pytest.raises(ValueError):
        infer_meta(Path("random.pdf"))


# ── 字节截断（Milvus 的长度限制以字节计）──────────────────


def test_truncate_utf8_respects_byte_limit():
    text = "中文" * 100  # 600 字节
    out, cut = truncate_utf8(text, 100)
    assert cut is True
    assert len(out.encode("utf-8")) <= 100


def test_truncate_utf8_no_cut_when_within_limit():
    out, cut = truncate_utf8("短文本", 100)
    assert cut is False
    assert out == "短文本"


def test_truncate_utf8_does_not_break_character():
    """按字节截断不能切出半个汉字。"""
    out, _ = truncate_utf8("中文测试", 7)  # 7 不是 3 的整数倍
    out.encode("utf-8").decode("utf-8")  # 不抛异常即合法


# ── 分块 ────────────────────────────────────────────────


def test_chunk_uid_is_stable():
    """重复解析必须得到相同 ID，否则增量更新无从比对。"""
    a = make_uid("600519_2025_annual", "heading", 1, 7)
    b = make_uid("600519_2025_annual", "heading", 1, 7)
    assert a == b
    assert a != make_uid("600519_2025_annual", "fixed", 1, 7)


def test_chunk_uid_differs_by_level():
    assert make_uid("k", "heading", 0, 1) != make_uid("k", "heading", 1, 1)


def test_est_tokens_counts_cjk_per_char():
    assert est_tokens("中文测试") == 4
    assert est_tokens("abcdef") == 2


def test_with_heading_prefixes_once():
    assert with_heading("正文", "第三节 > 一、") == "第三节 > 一、\n正文"
    already = "第三节 > 一、\n正文"
    assert with_heading(already, "第三节 > 一、") == already


def test_split_table_repeats_header_in_every_piece():
    """切分后的每一片都要带量纲与表头，否则是一堆没有列名的数字。"""
    rows = "\n".join(f"| 科目{i} | {i * 1000} |" for i in range(60))
    block = ParsedBlock(
        kind="table",
        text="【单位：万元】\n营业收入表\n| 科目 | 金额 |\n| --- | --- |\n" + rows,
        page_start=1,
        page_end=1,
    )
    parts = split_table(block, max_chars=300)
    assert len(parts) > 1
    for p in parts:
        assert "【单位：万元】" in p
        assert "| 科目 | 金额 |" in p


def test_split_table_keeps_short_table_intact():
    block = ParsedBlock(kind="table", text="【单位：元】\n| a | b |\n| --- | --- |\n| 1 | 2 |",
                        page_start=1, page_end=1)
    assert split_table(block, max_chars=5000) == [block.text]


def test_chunk_unit_defaults():
    u = ChunkUnit(
        chunk_uid="x", content="c", level=1, chunk_type="text",
        heading_path="", page_start=1, page_end=1, strategy="heading",
    )
    assert u.parent_uid is None
    assert u.table_flags == []


# ── 分块配置 ────────────────────────────────────────────


def test_chunking_cfg_defaults_enable_table_protection():
    """默认配置必须保护表格：关掉会让表头与数据被切散。"""
    cfg = ChunkingCfg()
    assert cfg.table_atomic is True
    assert cfg.table_metadata is True
    assert cfg.cross_page_merge is True


# ── Milvus VARCHAR 以字节计长（招商银行入库失败的根因） ──────


def test_unit_field_fits_all_known_units():
    """「百万元」是 9 字节，原来的 max_length=8 直接溢出。

    报错是 `length of varchar field unit exceeds max length, length: 9, max length: 8`。
    语料里只有元/千元/万元时这个字段从没被撑爆过，加入银行年报才暴露。
    """
    from app.store.milvus_schema import MAX_UNIT_LEN

    for unit in ["元", "千元", "万元", "百万元", "亿元", "平方米", "股"]:
        assert len(unit.encode("utf-8")) <= MAX_UNIT_LEN


def test_truncate_utf8_is_byte_based_not_char_based():
    """按字数截断看着安全，中文满长仍会超出字节上限。"""
    text = "财" * 100  # 300 字节
    out, cut = truncate_utf8(text, 64)
    assert cut is True
    assert len(out.encode("utf-8")) <= 64
    # 按字数截断会得到 64 个字 = 192 字节，照样炸
    assert len(text[:64].encode("utf-8")) > 64
