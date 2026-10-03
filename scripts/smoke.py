"""M0 冒烟检查。

    python -m scripts.smoke          只检查配置与存储（不花钱）
    python -m scripts.smoke --llm    额外调用一次模型（会产生少量费用）

验收标准（架构 §16 M0）：可启动、可观测。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

OK = "  [ OK ]"
NG = "  [FAIL]"
SKIP = "  [skip]"


async def check_config() -> bool:
    from app.config.experiment import available_experiments, load_experiment
    from app.config.settings import get_settings

    print("\n── 配置 ──")
    settings = get_settings()
    print(f"{OK} 环境={settings.app_env}  实验={settings.exp_id}")
    print(f"{OK} 可用实验：{', '.join(available_experiments())}")

    exp = load_experiment()
    print(f"{OK} 分块={exp.chunking.strategy}  检索={exp.retrieval.mode}  重排={exp.rerank.enabled}")
    print(f"{OK} 缓存实体约束={exp.cache.entity_constraint}  路由兜底={exp.router.fallback_branch}")
    return True


async def check_providers() -> bool:
    from app.providers.registry import configured_profiles, embedding_dim, get_registry

    print("\n── 模型 Provider ──")
    reg = get_registry()
    for profile in reg.all_chat_profiles():
        spec = reg.chat_spec(profile)
        mark = OK if spec.configured else SKIP
        state = "已配置" if spec.configured else f"未配置 {spec.api_key_env}"
        print(f"{mark} {profile:8s} {spec.provider}/{spec.model}  {state}")
    print(f"{OK} embedding 维度={embedding_dim()}")
    return any(configured_profiles().values())


async def check_stores() -> bool:
    from app.store.milvus import close_milvus, init_milvus, ping_milvus
    from app.store.pg import close_pg, init_pg, ping_pg
    from app.store.redis_client import close_redis, init_redis, ping_redis

    print("\n── 存储 ──")
    results = []
    for name, init, ping, close in (
        ("PostgreSQL", init_pg, ping_pg, close_pg),
        ("Redis", init_redis, ping_redis, close_redis),
        ("Milvus", init_milvus, ping_milvus, close_milvus),
    ):
        try:
            await init()
            ok, msg = await ping()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        finally:
            try:
                await close()
            except Exception:
                pass
        print(f"{OK if ok else NG} {name:11s} {msg}")
        results.append(ok)
    return all(results)


async def check_llm() -> bool:
    from app.graph.build import get_graph
    from app.graph.state import new_state

    print("\n── 模型调用（LangGraph 最小图）──")
    state = new_state(question="用一句话说明你是什么服务。", conv_id="smoke")
    try:
        result = await get_graph().ainvoke(state)
    except Exception as exc:
        print(f"{NG} 调用失败：{type(exc).__name__}: {exc}")
        return False
    print(f"{OK} 回答：{result.get('answer', '')[:80]}")
    print(f"{OK} 用量：{result.get('usage', {})}")
    return True


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="额外调用一次模型（会产生费用）")
    args = parser.parse_args()

    print("=" * 58)
    print("  FinCopilot M0 冒烟检查")
    print("=" * 58)

    await check_config()
    has_key = await check_providers()
    stores_ok = await check_stores()

    llm_ok = True
    if args.llm:
        if has_key:
            llm_ok = await check_llm()
        else:
            print(f"\n{SKIP} 未配置任何模型 Key，跳过模型调用")

    print("\n" + "=" * 58)
    if stores_ok and llm_ok:
        print("  M0 验收通过")
        return 0
    print("  存在未通过项，见上方 [FAIL]")
    print("  存储未连通？先执行：docker compose up -d")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
