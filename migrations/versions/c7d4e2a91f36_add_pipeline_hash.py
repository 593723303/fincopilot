"""add documents.pipeline_hash

Revision ID: c7d4e2a91f36
Revises: b3f17c9d2e40
Create Date: 2026-10-08

入库的幂等检查原本只比对 PDF 字节哈希，漏掉了「文件没变、但解析代码变了」
这一整类变化——索引静默停在旧版本上，且任何指标都看不出来。
本列记录处理流水线的指纹（解析器/分块器源码 + 分块配置 + embedding 模型），
与 content_hash 配对判断：两者都没变才允许跳过。详见 app/rag/fingerprint.py。

可空：本次迁移之前入库的行为 NULL，入库时视为来历不明、必须重算。
已确认用当前流水线跑过的语料，可用
`python -m scripts.ingest --stamp-pipeline` 一次性补写，避免无谓的重新 embedding。
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7d4e2a91f36"
down_revision: Union[str, Sequence[str], None] = "b3f17c9d2e40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("pipeline_hash", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "pipeline_hash")
