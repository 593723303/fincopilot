"""Token 用量 → 人民币折算。

成本是本项目的核心指标之一（架构 §1.3），因此从 M0 起就记账，
而不是等到 M6 优化阶段再补——没有基线就无法证明优化有效。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.providers.registry import get_registry


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_cny: float = 0.0
    by_profile: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, profile: str, prompt_tokens: int, completion_tokens: int) -> float:
        """累加一次调用，返回本次花费。"""
        cost = estimate_cost(profile, prompt_tokens, completion_tokens)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_cny = round(self.cost_cny + cost, 6)
        self.by_profile[profile] = round(self.by_profile.get(profile, 0.0) + cost, 6)
        return cost

    def to_dict(self) -> dict:
        return {
            "in": self.prompt_tokens,
            "out": self.completion_tokens,
            "cost_cny": self.cost_cny,
            "by_profile": self.by_profile,
        }


def estimate_cost(profile: str, prompt_tokens: int, completion_tokens: int) -> float:
    """按 models.yaml 的价格表折算。价格单位为元 / 1K tokens。"""
    try:
        spec = get_registry().chat_spec(profile)
    except KeyError:
        spec = get_registry().embed_spec(profile) if profile.startswith("embed") else None
    if spec is None:
        return 0.0
    return round(
        prompt_tokens / 1000 * spec.price_in + completion_tokens / 1000 * spec.price_out,
        6,
    )


def usage_from_response(response) -> tuple[int, int]:
    """从 LangChain 响应里取 token 用量，兼容不同厂商的字段位置。"""
    meta = getattr(response, "usage_metadata", None) or {}
    if meta:
        return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    raw = getattr(response, "response_metadata", None) or {}
    token_usage = raw.get("token_usage") or raw.get("usage") or {}
    return int(token_usage.get("prompt_tokens", 0)), int(token_usage.get("completion_tokens", 0))
