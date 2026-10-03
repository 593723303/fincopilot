"""实验配置加载。

架构 ADR-010：消融实验是核心方法论，切换实验不应改代码。
一组配置即一次实验；exp_id 贯穿 Milvus 标量字段、缓存键前缀与评估记录三处，
保证实验之间互不污染。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config.settings import CONFIG_DIR, get_settings

EXPERIMENT_DIR = CONFIG_DIR / "experiments"


class ChunkingCfg(BaseModel):
    strategy: str = "heading"
    chunk_size: int = 512
    overlap: int = 64
    parent_child: bool = True
    table_atomic: bool = True
    table_metadata: bool = True
    cross_page_merge: bool = True


class RetrievalCfg(BaseModel):
    mode: str = "hybrid"
    top_k_dense: int = 20
    top_k_sparse: int = 20
    fusion: str = "rrf"
    filters_pushdown: bool = True


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "gte-rerank-v2"
    top_n: int = 5


class QueryTransformCfg(BaseModel):
    coref_resolution: bool = True
    hyde: bool = False
    decompose: bool = False


class ParentExpansionCfg(BaseModel):
    enabled: bool = True
    dedup_by_parent: bool = True
    token_budget: int = 6000
    keep_table_intact: bool = True


class GenerationCfg(BaseModel):
    profile: str = "strong"
    require_citation: bool = True
    require_unit_declaration: bool = True
    relevance_threshold: float = 0.35
    max_retry_on_low_relevance: int = 1


class CacheCfg(BaseModel):
    exact: bool = True
    semantic: bool = True
    similarity_threshold: float = 0.97
    # strict：公司/报告期/指标三元组必须精确相等才允许语义匹配。
    # 架构 ADR-013 —— 财报问题仅差年份时向量相似度极高，纯相似度匹配会
    # 返回精确但错误的数字且无法察觉。不要改成 loose。
    entity_constraint: str = "strict"
    audit_sample_rate: float = 0.05


class RouterCfg(BaseModel):
    profile: str = "light"
    confidence_threshold: float = 0.7
    # 低置信度默认走 agent：agent 是 rag 的能力超集，判错只多花成本，
    # 反向判错会直接答错（架构 §6.2 步骤 4）
    fallback_branch: str = "agent"


class AgentCfg(BaseModel):
    max_steps: int = 8
    max_tool_errors: int = 3
    budget_cny: float = 0.15


class EvalCfg(BaseModel):
    dataset: str = "v1"
    bootstrap_rounds: int = 1000
    judge_profile: str = "judge"
    numeric_tolerance: float = 0.005
    stratified_report: bool = True


class Experiment(BaseModel):
    exp_id: str
    description: str = ""
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    retrieval: RetrievalCfg = Field(default_factory=RetrievalCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    query_transform: QueryTransformCfg = Field(default_factory=QueryTransformCfg)
    parent_expansion: ParentExpansionCfg = Field(default_factory=ParentExpansionCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    cache: CacheCfg = Field(default_factory=CacheCfg)
    router: RouterCfg = Field(default_factory=RouterCfg)
    agent: AgentCfg = Field(default_factory=AgentCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)

    def cache_prefix(self) -> str:
        """缓存键前缀 —— 防止上一组实验的缓存污染这一组的指标。"""
        return f"cache:{self.exp_id}"


def available_experiments() -> list[str]:
    return sorted(p.stem for p in EXPERIMENT_DIR.glob("*.yaml"))


@lru_cache(maxsize=16)
def load_experiment(exp_id: str | None = None) -> Experiment:
    exp_id = exp_id or get_settings().exp_id
    path: Path = EXPERIMENT_DIR / f"{exp_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"实验配置不存在：{path}\n可用实验：{', '.join(available_experiments()) or '(无)'}"
        )
    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    raw.setdefault("exp_id", exp_id)
    return Experiment(**raw)
