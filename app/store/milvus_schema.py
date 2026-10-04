"""Milvus collection 定义与建表。

对应架构文档 §7.2。已在 Milvus 3.x 实测验证（见 scripts/probe_milvus.py）：
dense + sparse 双向量、BM25 Function、中文分词、标量过滤下推、RRF 混合检索均可用。

两个刻意不使用的 3.x 新能力：
  - FunctionType.TEXTEMBEDDING（Milvus 自行调用 embedding 服务）
    → 会让 embedding 的 token 消耗脱离成本核算，且架空 Provider 抽象层
  - FunctionType.RERANK（Milvus 内置重排）
    → 重排需要做开关消融实验，控制权必须留在应用层
新能力更省事，但代价是可观测性与可替换性，不符合本项目的取舍。
"""

from __future__ import annotations

import asyncio
import logging

from pymilvus import DataType, Function, FunctionType, MilvusClient

from app.config.settings import get_settings
from app.providers.registry import embedding_dim

logger = logging.getLogger(__name__)

# 中文分词器。不配置时中文整句会被当作单个 token，BM25 形同虚设。
# 已实测：Milvus 3.x 支持内置 chinese 类型。
CHINESE_ANALYZER = {"type": "chinese"}

MAX_CONTENT_LEN = 8192
MAX_HEADING_LEN = 1024
# 量纲字段。看着只放两三个字，但 VARCHAR 的长度**以字节计**：
# 「百万元」是 9 字节，直接撑爆原来的 8——招商银行入库时就炸在这里
# （length of varchar field unit exceeds max length, length: 9, max length: 8）。
# 中文字段一律按字节留足余量，不要按字数估。
MAX_UNIT_LEN = 32


def collection_names() -> tuple[str, str]:
    s = get_settings()
    return (
        s.yaml_get("milvus.collection_chunks", "fin_chunks"),
        s.yaml_get("milvus.collection_query_cache", "query_cache"),
    )


def _build_chunks_schema(client: MilvusClient, dim: int):
    """文档块集合。

    sparse_vector 不手工写入，由 BM25 Function 从 content 自动生成，
    保证稀疏与稠密索引同生命周期——这是选择内置 BM25 而非自建索引的理由（ADR-006）。
    """
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    schema.add_field("chunk_uid", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=MAX_CONTENT_LEN,
        enable_analyzer=True,
        analyzer_params=CHINESE_ANALYZER,
    )

    # ── 标量过滤字段：必须下推到检索层，不做后置过滤 ──
    # 实测证明 BM25 对年份完全无感（2023/2024 两条得分相同），
    # 公司与报告期只能靠标量过滤卡住，指望检索器区分是不行的。
    schema.add_field("doc_key", DataType.VARCHAR, max_length=64)
    schema.add_field("company_code", DataType.VARCHAR, max_length=16)
    schema.add_field("report_year", DataType.INT32)
    schema.add_field("period", DataType.VARCHAR, max_length=16)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=16)
    schema.add_field("statement_type", DataType.VARCHAR, max_length=16)
    schema.add_field("strategy", DataType.VARCHAR, max_length=16)

    # ── 随块返回的上下文字段 ──
    # unit 必须随块带出：中文年报量纲混用（元/万元/亿元），
    # 生成阶段要据此声明单位，否则数值题会读错量级（ADR-016）。
    schema.add_field("unit", DataType.VARCHAR, max_length=MAX_UNIT_LEN)
    schema.add_field("heading_path", DataType.VARCHAR, max_length=MAX_HEADING_LEN)
    schema.add_field("page_start", DataType.INT32)
    schema.add_field("parent_uid", DataType.VARCHAR, max_length=128)

    schema.add_function(
        Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def _build_cache_schema(client: MilvusClient, dim: int):
    """语义缓存集合。

    与文档向量生命周期不同，必须独立存放，混存会污染检索（ADR-008）。

    entity_key 是实体硬约束的载体：(公司集合, 报告期集合, 指标类型) 的规范化串，
    检索时要求精确相等，向量相似度只用于匹配问法差异。
    财报问题仅差年份时向量相似度极高，纯相似度匹配会返回精确但错误的数字（ADR-013）。
    """
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("cache_key", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("answer_json", DataType.VARCHAR, max_length=16384)
    schema.add_field("entity_key", DataType.VARCHAR, max_length=256)
    schema.add_field("exp_id", DataType.VARCHAR, max_length=32)
    schema.add_field("hit_count", DataType.INT32)
    schema.add_field("created_at", DataType.INT64)
    return schema


def _chunks_index(client: MilvusClient):
    s = get_settings()
    idx = client.prepare_index_params()
    idx.add_index(
        field_name="dense_vector",
        index_type=s.yaml_get("milvus.index.type", "HNSW"),
        metric_type=s.yaml_get("milvus.index.metric", "COSINE"),
        params={
            "M": s.yaml_get("milvus.index.M", 16),
            "efConstruction": s.yaml_get("milvus.index.efConstruction", 200),
        },
    )
    idx.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    return idx


def _cache_index(client: MilvusClient):
    idx = client.prepare_index_params()
    idx.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 16})
    return idx


def _ensure(client: MilvusClient, name: str, schema, index_params, drop: bool) -> str:
    exists = client.has_collection(name)
    if exists and drop:
        client.drop_collection(name)
        exists = False
    if exists:
        client.load_collection(name)
        return "已存在"
    client.create_collection(collection_name=name, schema=schema)
    client.create_index(name, index_params)
    client.load_collection(name)
    return "已创建"


def init_collections_sync(client: MilvusClient, drop: bool = False) -> dict[str, str]:
    dim = embedding_dim()
    chunks_name, cache_name = collection_names()
    result = {
        chunks_name: _ensure(
            client, chunks_name, _build_chunks_schema(client, dim), _chunks_index(client), drop
        ),
        cache_name: _ensure(
            client, cache_name, _build_cache_schema(client, dim), _cache_index(client), drop
        ),
    }
    for name, state in result.items():
        logger.info("Milvus collection %s：%s", name, state)
    return result


async def init_collections(client: MilvusClient, drop: bool = False) -> dict[str, str]:
    """pymilvus 是同步 SDK，放线程池执行以免阻塞事件循环。"""
    return await asyncio.to_thread(init_collections_sync, client, drop)
