"""流水线指纹：入库该不该重算的依据。

这些测试锁住的是一条已经踩坑两次的纪律——
「文件没变」不等于「索引是最新的」。
"""

from __future__ import annotations

from app.config.experiment import load_experiment
from app.rag.fingerprint import PIPELINE_SOURCES, pipeline_fingerprint


def test_fingerprint_is_stable():
    """同样的输入必须得到同样的指纹，否则每次入库都会全量重算。"""
    exp = load_experiment()
    assert pipeline_fingerprint(exp) == pipeline_fingerprint(exp)
    assert len(pipeline_fingerprint(exp)) == 16


def test_fingerprint_tracks_chunking_config():
    """分块配置变了，指纹必须变——块边界变了，旧向量就不该再用。"""
    exp = load_experiment()
    base = pipeline_fingerprint(exp)

    changed = exp.model_copy(deep=True)
    changed.chunking.chunk_size = exp.chunking.chunk_size + 128
    assert pipeline_fingerprint(changed) != base

    changed2 = exp.model_copy(deep=True)
    changed2.chunking.strategy = "fixed" if exp.chunking.strategy != "fixed" else "heading"
    assert pipeline_fingerprint(changed2) != base


def test_fingerprint_covers_parser_sources():
    """指纹必须覆盖解析器与分块器的源码。

    这两处是 P1-7 与「科目名跨行折断」两次事故的发生地：
    改了它们而 PDF 没动，旧逻辑的索引会被当成最新的继续用。
    """
    assert "pdf_parser.py" in PIPELINE_SOURCES
    assert "chunker.py" in PIPELINE_SOURCES
    # indexer.py 刻意不收：它只管写库与日志，收进来会让日志微调触发全量重算
    assert "indexer.py" not in PIPELINE_SOURCES


def test_cache_namespace_includes_answer_fingerprint():
    """缓存分区键必须带回答指纹，否则改了代码还会供应旧答案。

    实测撞上过：修完科目名跨行折断、重新入库，再问同一个问题，
    拿回来的仍是修复前那句「未找到」——命中的是上一版代码写的条目。
    """
    from app.rag.cache import cache_namespace
    from app.rag.fingerprint import answer_fingerprint

    exp = load_experiment()
    ns = cache_namespace(exp)
    assert exp.exp_id in ns
    assert answer_fingerprint(exp) in ns
    # Milvus 的 exp_id 字段是 VARCHAR(32)，超了整条写入会被拒
    assert len(ns.encode("utf-8")) <= 32


def test_answer_fingerprint_tracks_generation_config():
    """生成配置变了，缓存必须换分区——同一个问题的答案会不一样。"""
    from app.rag.fingerprint import answer_fingerprint

    exp = load_experiment()
    changed = exp.model_copy(deep=True)
    changed.generation.require_citation = not exp.generation.require_citation
    assert answer_fingerprint(changed) != answer_fingerprint(exp)
