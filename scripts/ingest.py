"""把年报 PDF 入库。

    python -m scripts.ingest                      入库 data/raw 下全部 PDF
    python -m scripts.ingest --pages 20           只处理前 20 页（省 embedding 费用）
    python -m scripts.ingest --exp exp01_baseline 指定实验配置（决定分块策略）
    python -m scripts.ingest --force              内容未变也强制重建

同一份文档可以在不同实验配置下并存入库：块按 strategy 字段隔离，
互不干扰，这是 M4 消融实验能对比的前提（ADR-010）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pymilvus import MilvusClient

from app.config.experiment import load_experiment
from app.config.settings import get_settings
from app.rag.indexer import ingest_pdf
from app.store.pg import close_pg, init_pg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="PDF 路径，留空则处理 data/raw 下全部")
    parser.add_argument("--pages", type=int, default=None, help="只处理前 N 页")
    parser.add_argument("--exp", type=str, default=None, help="实验配置 ID")
    parser.add_argument("--force", action="store_true", help="内容未变也强制重建")
    args = parser.parse_args()

    pdfs = [Path(f) for f in args.files] if args.files else sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"{RAW_DIR} 下没有 PDF，先执行 python -m scripts.fetch_reports")
        return 1

    settings = get_settings()
    exp = load_experiment(args.exp)
    print("=" * 62)
    print(f"  入库 {len(pdfs)} 份文档")
    print(f"  实验 {exp.exp_id}  分块策略={exp.chunking.strategy}  父子块={exp.chunking.parent_child}")
    if args.pages:
        print(f"  仅处理前 {args.pages} 页")
    print("=" * 62)

    await init_pg()
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)

    total_cost = 0.0
    results = []
    try:
        for pdf in pdfs:
            print(f"\n── {pdf.name} ──")
            r = await ingest_pdf(pdf, client, exp=exp, max_pages=args.pages, force=args.force)
            results.append(r)
            if r.skipped:
                print(f"  [skip] 内容未变，已有 {r.chunk_count} 块")
            elif r.status == "ready":
                print(f"  [ OK ] {r.chunk_count} 块（子块 {r.child_count}），成本 ¥{r.embed_cost:.6f}")
                total_cost += r.embed_cost
            else:
                print(f"  [FAIL] {r.error}")
    finally:
        await close_pg()

    ok = sum(1 for r in results if r.status == "ready")
    print("\n" + "=" * 62)
    print(f"  成功 {ok}/{len(results)}，本次 embedding 成本合计 ¥{total_cost:.6f}")

    # 注意：get_collection_stats 的 row_count 是近似值，包含已标记删除
    # 但尚未 compaction 的行，会明显大于实际可查询数量。
    # 重建索引后用它判断"入库了多少"会得到误导性的结论，因此改用 query 计数。
    name = settings.yaml_get("milvus.collection_chunks", "fin_chunks")
    live = len(
        client.query(name, filter="report_year > 0", output_fields=["chunk_uid"], limit=16384)
    )
    print(f"  Milvus {name} 实际可查询 {live} 条向量")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
