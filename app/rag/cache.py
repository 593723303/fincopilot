"""问答缓存：精确 + 语义两级。

两级的分工：

  精确缓存（Redis）  同一个问题原样再问一次，直接返回，零成本零延迟
  语义缓存（Milvus） 问法不同、问的是同一件事，靠向量相似度命中

**语义缓存在财报场景有一个特有的风险**：问法只差一个实体时，
向量相似度仍然很高，纯按相似度匹配会返回一个**精确但错误**的数字——
格式正确、引用齐全，从输出上看不出它答的是另一年或另一个科目。

实测各类单一差异的余弦相似度（text-embedding-v4，茅台的四种问法）：

    仅差公司   0.5824     宁德时代 vs 贵州茅台
    仅差年份   0.86–0.87  2025 vs 2024 / 2023
    仅差科目   0.9224     营业收入 vs 净利润   ← 最危险

**最危险的是科目而不是年份**，这与设计时的直觉相反：0.9224 距离
0.97 的阈值只剩 0.05 的余量，而年份差异有 0.10 以上。

所以 entity_key 必须同时约束 (公司, 年份, 科目) 三者（ADR-013），
精确相等才允许命中，向量相似度只用来吸收问法差异。
最初只约束了公司与年份，科目维度是实测相似度之后才补上的。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from app.config.experiment import Experiment
from app.providers.registry import get_embeddings
from app.store.milvus_schema import collection_names
from app.store.redis_client import redis

logger = logging.getLogger(__name__)

EXACT_PREFIX = "fc:ans:"
EXACT_TTL = 24 * 3600
MAX_ANSWER_BYTES = 16000


# 问题里出现的财务科目。只用来给缓存做实体约束，不求穷举——
# 认不出科目时 m= 为空，等于退回「公司+年份」约束，不会放宽。
METRIC_WORDS = (
    "营业总收入",
    "营业收入",
    "营业成本",
    "利润总额",
    "净利润",
    "归母净利润",
    "扣非",
    "毛利率",
    "净利率",
    "经营活动产生的现金流量净额",
    "现金流",
    "总资产",
    "资产总计",
    "净资产",
    "研发投入",
    "研发费用",
    "销售费用",
    "管理费用",
    "财务费用",
    "存货",
    "货币资金",
    "股本",
    "每股收益",
    "分红",
    "股利",
)


def metric_words(question: str) -> list[str]:
    """问题里出现了哪些科目词。"""
    return sorted({w for w in METRIC_WORDS if w in question})


def entity_key(company_codes: list[str], years: list[int], question: str = "") -> str:
    """实体三元组的规范化串：公司、年份、科目。

    排序后拼接，保证同一组实体只有一种写法。
    科目这一维是实测之后才加的——「2025年营业收入」与「2025年净利润」
    的公司与年份完全相同，而向量相似度高达 0.9224，
    只靠 0.97 的阈值挡，余量仅 0.05。
    """
    codes = ",".join(sorted(company_codes or []))
    ys = ",".join(str(y) for y in sorted(years or []))
    ms = ",".join(metric_words(question))
    return f"c={codes}|y={ys}|m={ms}"


def exact_key(question: str, exp_id: str, ekey: str) -> str:
    """精确缓存的键。

    实验配置参与计算：不同实验的检索参数不同，答案也不同，
    共用一个键会让消融实验互相污染——这与块按 strategy 隔离是同一个理由。
    """
    raw = f"{exp_id}|{ekey}|{question.strip()}"
    return EXACT_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


async def get_exact(question: str, exp: Experiment, ekey: str) -> dict[str, Any] | None:
    if not exp.cache.exact:
        return None
    try:
        raw = await redis().get(exact_key(question, exp.exp_id, ekey))
    except Exception as exc:  # 缓存不可用不应影响主流程
        logger.warning("精确缓存读取失败：%s", exc)
        return None
    return json.loads(raw) if raw else None


async def put_exact(question: str, exp: Experiment, ekey: str, payload: dict) -> None:
    if not exp.cache.exact:
        return
    try:
        await redis().setex(
            exact_key(question, exp.exp_id, ekey),
            EXACT_TTL,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("精确缓存写入失败：%s", exc)


def _cache_collection() -> str:
    return collection_names()[1]


async def get_semantic(
    client, question: str, exp: Experiment, ekey: str
) -> dict[str, Any] | None:
    """语义缓存查询。实体不同一律不命中。"""
    if not exp.cache.semantic:
        return None
    try:
        vec = (await get_embeddings().aembed_documents([question]))[0]
        hits = client.search(
            collection_name=_cache_collection(),
            data=[vec],
            limit=1,
            # 实体硬约束下推到检索层，而不是查回来再过滤：
            # 后置过滤会让 limit=1 被一个实体不符的近邻占掉，
            # 本来能命中的那条反而被挤出去。
            filter=f'entity_key == "{ekey}" and exp_id == "{exp.exp_id}"',
            output_fields=["answer_json"],
            search_params={"metric_type": "COSINE"},
        )
    except Exception as exc:
        logger.warning("语义缓存查询失败：%s", exc)
        return None

    rows = hits[0] if hits else []
    if not rows:
        return None
    top = rows[0]
    score = float(top.get("distance", 0.0))
    if score < exp.cache.similarity_threshold:
        return None
    payload = json.loads(top["entity"]["answer_json"])
    payload["cache_similarity"] = round(score, 4)
    return payload


async def put_semantic(
    client, question: str, exp: Experiment, ekey: str, payload: dict
) -> None:
    if not exp.cache.semantic:
        return
    body = json.dumps(payload, ensure_ascii=False)
    if len(body.encode("utf-8")) > MAX_ANSWER_BYTES:
        logger.debug("回答过长，跳过语义缓存写入")
        return
    try:
        vec = (await get_embeddings().aembed_documents([question]))[0]
        client.upsert(
            collection_name=_cache_collection(),
            data=[
                {
                    "cache_key": exact_key(question, exp.exp_id, ekey)[len(EXACT_PREFIX) :],
                    "vector": vec,
                    "answer_json": body,
                    "entity_key": ekey[:256],
                    "exp_id": exp.exp_id[:32],
                    # schema 里的字段一个都不能少：Milvus 对非 nullable 字段
                    # 直接拒绝整条写入，而且报错只说「missed a field」，
                    # 不会告诉你其余字段其实都对
                    "hit_count": 0,
                    "created_at": int(time.time()),
                }
            ],
        )
        # upsert 之后不 flush，新写的条目对随后的 search 不可见。
        # 表现是「刚问过的同义问题仍然没命中」，很容易被误判成阈值太高。
        client.flush(_cache_collection())
    except Exception as exc:
        logger.warning("语义缓存写入失败：%s", exc)
