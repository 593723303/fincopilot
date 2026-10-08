"""处理流水线的指纹。

入库的幂等检查原本只比对 **PDF 文件字节的哈希**：

    if existing.content_hash == digest and existing.status == "ready":
        跳过

这条件漏掉了一整类变化——**文件没变，但处理文件的代码变了**。
解析器、分块器、分块参数、embedding 模型，任何一处改动都会让
同一份 PDF 产出不同的块，而哈希一个字节都不会变，于是入库直接跳过，
索引静默地停在旧版本上。

这不是假想的风险，是本项目已经发生过两次的事故：

- P1-7 给汇总表打 `table_summary` 标记后，17 家公司的索引里**从未出现**
  这个标记，因为没人想起要加 `--force`。缺陷藏了一天，直到建 v4
  才暴露出来
- 今天修了科目名跨行折断，同样必须手动记得 `--force`

更麻烦的是**从外部完全看不出来**：库里有块、检索有结果、回答看着正常，
只是用的是旧解析逻辑。没有任何一个指标会掉下来。

### 指纹包含什么

| 来源 | 为什么 |
|---|---|
| `pdf_parser.py` 源码 | 决定表格怎么切、单位怎么认、口径怎么继承 |
| `chunker.py` 源码 | 决定块怎么分、父子关系怎么建 |
| 分块配置 | 策略、块大小、重叠、父子块等，直接改变块边界 |
| embedding 模型与维度 | 换模型后旧向量与新查询不在同一空间 |

### 一个有意为之的取舍

按源码整文件哈希，**改注释也会触发重算**。这是刻意选的方向：
漏算的代价（索引静默过期、而且测不出来）远大于多算的代价
（22 份年报重新 embedding 约 ¥1 量级）。
想精确到「只有逻辑变化才触发」就得剥离注释与 docstring，
那套做法既脆弱又难解释，不值得。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.config.experiment import Experiment
from app.providers.registry import get_registry

# 只收会改变**块内容**的文件。indexer.py 刻意不收：
# 它负责写库与日志，改它不影响块本身，收进来会让日志微调也触发全量重算。
PIPELINE_SOURCES = ("pdf_parser.py", "chunker.py")


def pipeline_fingerprint(exp: Experiment, embed_profile: str = "default") -> str:
    """当前处理流水线的指纹，16 位十六进制。

    与 `content_hash` 配对使用：两者**都**没变才允许跳过入库。
    """
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in PIPELINE_SOURCES:
        h.update(name.encode())
        h.update(hashlib.sha256((here / name).read_bytes()).digest())

    c = exp.chunking
    h.update(
        repr(
            (
                c.strategy,
                c.chunk_size,
                c.overlap,
                c.parent_child,
                c.table_atomic,
                c.table_metadata,
                c.cross_page_merge,
            )
        ).encode()
    )

    spec = get_registry().embed_spec(embed_profile)
    h.update(f"{spec.model}:{spec.dim}".encode())
    return h.hexdigest()[:16]
