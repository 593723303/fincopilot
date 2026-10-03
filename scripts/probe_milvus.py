"""Milvus 3.x 能力探查。

架构文档 §7.2 的 collection schema 是按 Milvus 2.5 设计的，而实际服务端为 3.x。
本脚本验证设计所依赖的能力在 3.x 下是否成立，重点是中文场景：

  1. dense + sparse 双向量字段共存
  2. BM25 Function 自动生成稀疏向量
  3. **中文分词**（默认分词器按空格切，中文会整句成一个 token，稀疏检索失效）
  4. HNSW 索引与标量过滤下推
  5. hybrid_search + RRF 融合

    python -m scripts.probe_milvus
"""

from __future__ import annotations

import random
import sys

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config.settings import get_settings

COLLECTION = "_probe_fin_chunks"
DIM = 1024

# 构造几条相近的中文财报片段：共享「营业收入」等关键词，
# 但公司与年份不同——正好是语义相似度区分不开、必须靠关键词和标量过滤的场景
DOCS = [
    ("600519", 2024, "table", "万元", "公司2024年度实现营业总收入1,476.94万元，同比增长15.66%。"),
    ("600519", 2023, "table", "万元", "公司2023年度实现营业总收入1,276.36万元，同比增长18.04%。"),
    ("300750", 2024, "table", "亿元", "本期营业收入为3,620.13亿元，较上年同期下降9.70%。"),
    ("600519", 2024, "text", "", "报告期内公司持续推进渠道改革，加强品牌建设与市场投入。"),
    ("300750", 2024, "text", "", "公司研发投入持续加大，动力电池系统产能利用率稳步提升。"),
]

OK, NG = "  [ OK ]", "  [FAIL]"


def rand_vec() -> list[float]:
    return [random.random() for _ in range(DIM)]


def build_schema(client: MilvusClient, analyzer_params: dict | None):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_uid", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    text_kwargs = {"max_length": 8192, "enable_analyzer": True}
    if analyzer_params:
        text_kwargs["analyzer_params"] = analyzer_params
    schema.add_field("content", DataType.VARCHAR, **text_kwargs)

    # 标量过滤字段
    schema.add_field("company_code", DataType.VARCHAR, max_length=16)
    schema.add_field("report_year", DataType.INT32)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
    schema.add_field("unit", DataType.VARCHAR, max_length=8)

    schema.add_function(
        Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def main() -> int:
    settings = get_settings()
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
    print(f"连接 {settings.milvus_uri}")

    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    # ── 1. 中文分词器 ──────────────────────────────────
    # 不配分词器时，中文整句会被当成一个 token，BM25 形同虚设。
    # 逐个尝试，确定本版本实际支持哪种写法。
    candidates = [
        ("chinese 内置", {"type": "chinese"}),
        ("jieba tokenizer", {"tokenizer": "jieba"}),
        ("默认（英文）", None),
    ]
    schema = None
    chosen = None
    for label, params in candidates:
        try:
            schema = build_schema(client, params)
            client.create_collection(collection_name=COLLECTION, schema=schema)
            chosen = (label, params)
            print(f"{OK} 分词器可用：{label}  params={params}")
            break
        except Exception as exc:
            print(f"{NG} 分词器 {label} 不可用：{str(exc)[:110]}")
            if client.has_collection(COLLECTION):
                client.drop_collection(COLLECTION)
    if chosen is None:
        print(f"{NG} 所有分词器方案均失败，无法建表")
        return 1

    # ── 2. 索引 ────────────────────────────────────────
    try:
        idx = client.prepare_index_params()
        idx.add_index(
            field_name="dense_vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        idx.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        client.create_index(COLLECTION, idx)
        client.load_collection(COLLECTION)
        print(f"{OK} HNSW(M=16, efConstruction=200) + SPARSE_INVERTED_INDEX 建立成功")
    except Exception as exc:
        print(f"{NG} 索引建立失败：{exc}")
        return 1

    # ── 3. 写入（sparse_vector 由 BM25 Function 自动生成，不手工提供）──
    rows = [
        {
            "chunk_uid": f"probe-{i}",
            "dense_vector": rand_vec(),
            "content": text,
            "company_code": code,
            "report_year": year,
            "chunk_type": ctype,
            "unit": unit,
        }
        for i, (code, year, ctype, unit, text) in enumerate(DOCS)
    ]
    try:
        client.insert(COLLECTION, rows)
        client.flush(COLLECTION)
        print(f"{OK} 写入 {len(rows)} 条，稀疏向量由 BM25 Function 自动生成")
    except Exception as exc:
        print(f"{NG} 写入失败：{exc}")
        return 1

    out_fields = ["chunk_uid", "content", "company_code", "report_year", "unit"]

    # ── 4. 稀疏检索：中文分词是否真的生效 ──────────────
    try:
        res = client.search(
            COLLECTION,
            data=["营业总收入"],
            anns_field="sparse_vector",
            limit=3,
            output_fields=out_fields,
        )
        hits = res[0]
        print(f"{OK} BM25 稀疏检索返回 {len(hits)} 条")
        for h in hits:
            print(f"         score={h['distance']:.3f}  {h['entity']['content'][:34]}")
        if not hits:
            print("         ⚠ 命中为空 —— 中文未被正确分词，BM25 实际失效")
    except Exception as exc:
        print(f"{NG} 稀疏检索失败：{exc}")

    # ── 5. 标量过滤下推 ────────────────────────────────
    try:
        res = client.search(
            COLLECTION,
            data=["营业总收入"],
            anns_field="sparse_vector",
            limit=5,
            filter='company_code == "600519" and report_year == 2024',
            output_fields=out_fields,
        )
        n = len(res[0])
        all_match = all(
            h["entity"]["company_code"] == "600519" and h["entity"]["report_year"] == 2024
            for h in res[0]
        )
        verdict = "全部满足" if all_match else "未生效"
        print(f"{OK if all_match else NG} 标量过滤下推：返回 {n} 条，过滤条件{verdict}")
    except Exception as exc:
        print(f"{NG} 标量过滤失败：{exc}")

    # ── 6. 混合检索 + RRF ──────────────────────────────
    try:
        dense_req = AnnSearchRequest(
            data=[rand_vec()], anns_field="dense_vector", param={"ef": 64}, limit=5
        )
        sparse_req = AnnSearchRequest(
            data=["营业总收入"], anns_field="sparse_vector", param={}, limit=5
        )
        res = client.hybrid_search(
            COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(60),
            limit=3,
            output_fields=out_fields,
        )
        print(f"{OK} hybrid_search + RRF(k=60) 返回 {len(res[0])} 条")
        for h in res[0]:
            print(f"         rrf={h['distance']:.4f}  {h['entity']['content'][:34]}")
    except Exception as exc:
        print(f"{NG} 混合检索失败：{exc}")

    client.drop_collection(COLLECTION)
    print("\n探查结束，测试集合已清理")
    print(f"结论：中文分词方案 = {chosen[0]}  params={chosen[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
