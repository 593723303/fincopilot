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
