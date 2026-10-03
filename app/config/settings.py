"""运行时配置。

敏感项（Key、连接串）走环境变量；非敏感项走 configs/settings.yaml。
两者在这里合并为单一入口，其余模块只认 get_settings()。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """环境变量配置。字段名即环境变量名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── 运行时 ──
    app_env: str = "local"
    log_level: str = "INFO"
    exp_id: str = "exp01_baseline"

    # ── 模型平台 ──
    dashscope_api_key: str = ""
    deepseek_api_key: str = ""
    ark_api_key: str = ""
    vllm_api_key: str = "not-needed"

    # ── 存储 ──
    postgres_dsn: str = "postgresql+asyncpg://fincopilot:fincopilot@127.0.0.1:5432/fincopilot"
    redis_url: str = "redis://127.0.0.1:6379/0"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""

    # ── 可观测 ──
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── 由 YAML 载入，不来自环境变量 ──
    yaml_config: dict[str, Any] = Field(default_factory=dict)

    @property
    def langfuse_enabled(self) -> bool:
        """未配置 Key 时自动关闭上报，保证本地开发不被可观测组件阻塞。"""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def yaml_get(self, path: str, default: Any = None) -> Any:
        """按点分路径读取 YAML 配置，如 yaml_get("milvus.index.M")。"""
        node: Any = self.yaml_config
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.yaml_config = _load_yaml("settings.yaml")
    return settings
