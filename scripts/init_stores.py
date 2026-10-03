"""初始化存储层。

    python -m scripts.init_stores           幂等创建，已存在则跳过
    python -m scripts.init_stores --drop    先删后建（会清空向量数据）

PostgreSQL 的表结构由 Alembic 管理（python -m alembic upgrade head），
本脚本只负责 Milvus collection —— 向量库没有迁移工具，schema 变更需显式重建。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pymilvus import MilvusClient

from app.config.settings import get_settings
from app.providers.registry import embedding_dim
from app.store.milvus_schema import collection_names, init_collections_sync


async def check_pg_migrated() -> tuple[bool, str]:
    """确认 Alembic 迁移已执行 —— 两边 schema 要同时就位才算完整。"""
    from sqlalchemy import text

    from app.store.pg import close_pg, init_pg, session_factory

    try:
        await init_pg()
        async with session_factory()() as session:
            rev = (await session.execute(text("SELECT version_num FROM alembic_version"))).scalar()
            tables = (
                await session.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema='public' ORDER BY table_name"
                    )
                )
            ).scalars().all()
        return True, f"迁移版本 {rev}，表：{', '.join(t for t in tables if t != 'alembic_version')}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}（是否忘了 python -m alembic upgrade head）"
    finally:
        await close_pg()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="先删除已有 collection 再重建")
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 58)
    print("  初始化存储层")
    print("=" * 58)

    # ── PostgreSQL ──
    ok, msg = asyncio.run(check_pg_migrated())
    print(f"\n── PostgreSQL ──\n  [{'  OK ' if ok else 'FAIL'}] {msg}")
    if not ok:
        return 1

    # ── Milvus ──
    print(f"\n── Milvus ──\n  连接 {settings.milvus_uri}，向量维度 {embedding_dim()}")
    if args.drop:
        print("  --drop 已指定：将删除并重建 collection，现有向量数据会丢失")
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
    try:
        result = init_collections_sync(client, drop=args.drop)
    except Exception as exc:
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        return 1

    for name, state in result.items():
        stats = client.get_collection_stats(name)
        print(f"  [  OK ] {name:16s} {state}，当前 {stats.get('row_count', 0)} 条")

    chunks_name, _ = collection_names()
    desc = client.describe_collection(chunks_name)
    fields = [f["name"] for f in desc.get("fields", [])]
    funcs = [f.get("name") for f in desc.get("functions", [])]
    print(f"\n  {chunks_name} 字段（{len(fields)}）：{', '.join(fields)}")
    print(f"  BM25 Function：{funcs or '未配置'}")

    print("\n" + "=" * 58)
    print("  存储层就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
