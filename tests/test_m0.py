"""M0 骨架测试 —— 不依赖任何外部服务，CI 中可直接跑。"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.experiment import available_experiments, load_experiment
from app.observability.cost import TokenUsage, estimate_cost
from app.providers.registry import embedding_dim, get_registry
from app.schemas.chat import ChatRequest


@pytest.fixture
async def client():
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── 配置体系 ──────────────────────────────────────────


def test_baseline_experiment_loads():
    exp = load_experiment("exp01_baseline")
    assert exp.exp_id == "exp01_baseline"
    assert exp.chunking.strategy == "fixed"
    assert exp.retrieval.mode == "dense"
    assert exp.rerank.enabled is False


def test_experiment_discovery():
    assert "exp01_baseline" in available_experiments()


def test_missing_experiment_raises_with_hint():
    with pytest.raises(FileNotFoundError) as exc:
        load_experiment("exp_does_not_exist")
    assert "可用实验" in str(exc.value)


def test_cache_entity_constraint_is_strict():
    """ADR-013：缓存实体约束一旦放松就会返回精确但错误的数字。"""
    assert load_experiment("exp01_baseline").cache.entity_constraint == "strict"


def test_router_falls_back_to_agent():
    """架构 §6.2 步骤 4：低置信度必须走能力超集，不能反过来。"""
    assert load_experiment("exp01_baseline").router.fallback_branch == "agent"


def test_cache_prefix_isolates_experiments():
    a = load_experiment("exp01_baseline").cache_prefix()
    assert a.startswith("cache:exp01_baseline")


# ── Provider Registry ─────────────────────────────────


def test_registry_builds_all_profiles():
    reg = get_registry()
    for profile in ("strong", "light", "judge"):
        assert profile in reg.all_chat_profiles()


def test_judge_uses_different_vendor_than_strong():
    """架构 §12.2：同模型既生成又评判会产生自我偏好偏差。"""
    reg = get_registry()
    assert reg.chat_spec("judge").provider != reg.chat_spec("strong").provider


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        get_registry().chat_spec("nope")


def test_api_key_resolves_from_settings_not_only_env(monkeypatch):
    """回归测试：.env 中的 Key 必须能被 Provider 读到。

    pydantic-settings 的 env_file 只把值加载进 Settings 对象，不会注入
    os.environ。若 registry 只读 os.environ，就会出现「明明填了 Key
    却提示未配置」——M0 实际踩到过这个问题。
    """
    from app.config.settings import get_settings

    spec = get_registry().chat_spec("strong")
    monkeypatch.delenv(spec.api_key_env, raising=False)
    monkeypatch.setattr(get_settings(), spec.api_key_env.lower(), "sk-test-from-settings")

    assert spec.configured is True
    assert spec.api_key == "sk-test-from-settings"


def test_api_key_absent_reports_unconfigured(monkeypatch):
    from app.config.settings import get_settings

    spec = get_registry().chat_spec("strong")
    monkeypatch.delenv(spec.api_key_env, raising=False)
    monkeypatch.setattr(get_settings(), spec.api_key_env.lower(), "")

    assert spec.configured is False


def test_embedding_dim_matches_milvus_schema():
    assert embedding_dim() == 1024


# ── 成本核算 ──────────────────────────────────────────


def test_cost_estimation():
    cost = estimate_cost("strong", prompt_tokens=1000, completion_tokens=1000)
    spec = get_registry().chat_spec("strong")
    assert cost == pytest.approx(spec.price_in + spec.price_out)


def test_usage_accumulates_by_profile():
    usage = TokenUsage()
    usage.add("strong", 1000, 500)
    usage.add("light", 2000, 100)
    assert usage.total_tokens == 3600
    assert set(usage.by_profile) == {"strong", "light"}
    assert usage.cost_cny > 0


# ── 接口契约 ──────────────────────────────────────────


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_readyz_reports_stores_without_crashing(client):
    """存储未起时应返回 503 并说明原因，而不是抛异常。"""
    resp = await client.get("/readyz")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert set(body["stores"]) == {"postgres", "redis", "milvus"}
    assert "exp_id" in body["experiment"]


def test_chat_request_rejects_overlong_question():
    with pytest.raises(ValueError):
        ChatRequest(question="x" * 501)
