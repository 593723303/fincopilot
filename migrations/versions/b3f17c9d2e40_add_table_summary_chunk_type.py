"""add table_summary chunk type

Revision ID: b3f17c9d2e40
Revises: 8c21ab77e4d1
Create Date: 2026-10-06

「近三年主要会计数据」汇总表单独成类，供检索层按标签强制召回（P1-7）。
chunk_type 的 CheckConstraint 是写死的枚举，加一类就必须迁移。
"""
from typing import Sequence, Union

from alembic import op

revision: str = "b3f17c9d2e40"
down_revision: Union[str, Sequence[str], None] = "8c21ab77e4d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD = "('text', 'table')"
NEW = "('text', 'table', 'table_summary')"


def upgrade() -> None:
    op.drop_constraint("ck_chunks_type", "chunks", type_="check")
    op.create_check_constraint("ck_chunks_type", "chunks", f"chunk_type IN {NEW}")


def downgrade() -> None:
    # 回退前先降级这些块，否则约束建不回去
    op.execute("UPDATE chunks SET chunk_type = 'table' WHERE chunk_type = 'table_summary'")
    op.drop_constraint("ck_chunks_type", "chunks", type_="check")
    op.create_check_constraint("ck_chunks_type", "chunks", f"chunk_type IN {OLD}")
