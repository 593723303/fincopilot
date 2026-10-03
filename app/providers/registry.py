"""Provider Registry —— 所有模型调用的唯一入口。

架构 ADR-004：百炼 / DeepSeek / 火山 / vLLM 全部提供 OpenAI 兼容端点，
四者差异仅有 base_url、模型名、价格三项，没有任何行为差异。
因此这一层是「配置表 + 工厂函数」，而不是抽象基类加四个子类。

最初的设计是 ChatProvider 抽象基类 + 子类继承树，审查时按决策阶梯砍掉了。
少写的那三百行代码是设计成果，不是偷懒。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config.settings import CONFIG_DIR, get_settings


@dataclass(frozen=True)
class ModelSpec:
    """一个可调用模型的全部信息。价格单位：元 / 1K tokens。"""

    profile: str
    provider: str
    model: str
    base_url: str
    api_key_env: str
    price_in: float = 0.0
    price_out: float = 0.0
    timeout: int = 60
    temperature: float = 0.1
    dim: int | None = None
    batch_size: int | None = None

    def _resolve_key(self) -> str:
        """按 settings → 进程环境变量的顺序取 Key。

        pydantic-settings 的 env_file 只把值加载进 Settings 对象，
        不会注入 os.environ。只读 os.environ 会导致 .env 里的 Key 被忽略，
        表现为「明明填了 Key 却提示未配置」。
        """
        settings = get_settings()
        from_settings = getattr(settings, self.api_key_env.lower(), "") or ""
        return from_settings or os.environ.get(self.api_key_env, "") or ""

    @property
    def api_key(self) -> str:
        return self._resolve_key() or "not-needed"

    @property
    def configured(self) -> bool:
        """vLLM 等本地端点不需要真实 Key，其余必须配置。"""
        if self.provider == "vllm":
            return True
        return bool(self._resolve_key())


class _Registry:
    def __init__(self, raw: dict[str, Any]) -> None:
        self._providers: dict[str, dict] = raw.get("providers", {})
        self._chat = self._build(raw.get("profiles", {}))
        self._embed = self._build(raw.get("embeddings", {}))
        self._rerank_raw: dict[str, dict] = raw.get("rerank", {})
        self.fallback: dict[str, list[str]] = raw.get("fallback", {})

    def _build(self, section: dict[str, dict]) -> dict[str, ModelSpec]:
        specs: dict[str, ModelSpec] = {}
        for name, cfg in section.items():
            provider = cfg["provider"]
            if provider not in self._providers:
                raise ValueError(f"models.yaml: profile '{name}' 引用了未定义的 provider '{provider}'")
            base = self._providers[provider]
            specs[name] = ModelSpec(
                profile=name,
                provider=provider,
                model=cfg["model"],
                base_url=base["base_url"],
                api_key_env=base["api_key_env"],
                price_in=cfg.get("price_in", 0.0),
                price_out=cfg.get("price_out", 0.0),
                timeout=cfg.get("timeout", 60),
                temperature=cfg.get("temperature", 0.1),
                dim=cfg.get("dim"),
                batch_size=cfg.get("batch_size"),
            )
        return specs

    def chat_spec(self, profile: str) -> ModelSpec:
        if profile not in self._chat:
            raise KeyError(f"未知 chat profile '{profile}'，可用：{list(self._chat)}")
        return self._chat[profile]

    def embed_spec(self, profile: str = "default") -> ModelSpec:
        if profile not in self._embed:
            raise KeyError(f"未知 embedding profile '{profile}'，可用：{list(self._embed)}")
        return self._embed[profile]

    def rerank_model(self, profile: str = "default") -> str:
        return self._rerank_raw.get(profile, {}).get("model", "")

    def all_chat_profiles(self) -> list[str]:
        return list(self._chat)


@lru_cache(maxsize=1)
def get_registry() -> _Registry:
    path = CONFIG_DIR / "models.yaml"
    with path.open(encoding="utf-8") as fh:
        return _Registry(yaml.safe_load(fh) or {})


# ── 工厂函数：模块对外的全部接口 ───────────────────────────────


def get_chat(profile: str = "strong", **overrides: Any) -> BaseChatModel:
    """返回一个 LangChain ChatModel。

    max_retries 固定为 0 —— 重试与降级由 resilience 层统一处理，
    不让 SDK 自行重试，否则 trace 里看不到真实的失败次数。
    """
    spec = get_registry().chat_spec(profile)
    params: dict[str, Any] = {
        "model": spec.model,
        "base_url": spec.base_url,
        "api_key": spec.api_key,
        "timeout": spec.timeout,
        "temperature": spec.temperature,
        "max_retries": 0,
    }
    params.update(overrides)
    return ChatOpenAI(**params)


def get_embeddings(profile: str = "default") -> OpenAIEmbeddings:
    spec = get_registry().embed_spec(profile)
    return OpenAIEmbeddings(
        model=spec.model,
        base_url=spec.base_url,
        api_key=spec.api_key,
        chunk_size=spec.batch_size or 64,
    )


def embedding_dim(profile: str = "default") -> int:
    return get_registry().embed_spec(profile).dim or 1024


def fallback_chain(profile: str) -> list[str]:
    """主 profile 不可用时的降级顺序（架构 §10.1）。"""
    return get_registry().fallback.get(profile, [])


def configured_profiles() -> dict[str, bool]:
    """各 profile 是否已配置 Key —— 供 /readyz 与冒烟脚本展示。"""
    reg = get_registry()
    result = {p: reg.chat_spec(p).configured for p in reg.all_chat_profiles()}
    result["embed:default"] = reg.embed_spec().configured
    return result


