"""add scope eval category

Revision ID: 8c21ab77e4d1
Revises: 404a316a2005
Create Date: 2026-10-05

口径辨析（合并 vs 母公司）在 v2 评估集里单独成类。
CheckConstraint 是写死的枚举，加一类就必须迁移，否则导入直接被数据库拒绝。
"""
from typing import Sequence, Union

from alembic import op

revision: str = "8c21ab77e4d1"
down_revision: Union[str, Sequence[str], None] = "404a316a2005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD = "('fact', 'table', 'multihop', 'refuse')"
NEW = "('fact', 'table', 'multihop', 'scope', 'refuse')"


def upgrade() -> None:
    op.drop_constraint("ck_eval_items_category", "eval_items", type_="check")
    op.create_check_constraint("ck_eval_items_category", "eval_items", f"category IN {NEW}")


def downgrade() -> None:
    # 回退前必须先清掉 scope 类题目，否则约束建不回去
    op.execute("DELETE FROM eval_items WHERE category = 'scope'")
    op.drop_constraint("ck_eval_items_category", "eval_items", type_="check")
    op.create_check_constraint("ck_eval_items_category", "eval_items", f"category IN {OLD}")
