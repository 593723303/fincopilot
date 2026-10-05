"""评估运行器。

    python -m eval.run --load eval/datasets/seed.jsonl --dataset seed
    python -m eval.run --dataset seed                      用当前实验配置跑
    python -m eval.run --dataset seed --exp exp01_baseline 指定实验
    python -m eval.run --dataset seed --limit 5            小样本试跑

直接调用图而非 HTTP 接口：评估要能在 CI 中运行，不应依赖服务已启动。

只计算主指标（确定性判定，成本近似为零）。RAGAS 等模型评判指标
单独跑，因为它们的调用量可达主流程的数倍，不该混在每次回归里
（架构 §14.6 评估成本控制）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from app.config.experiment import Experiment, load_experiment
from app.graph.build import get_graph
from app.graph.state import new_state
from app.store.milvus import close_milvus, init_milvus
from app.store.models import EvalDataset, EvalItem, EvalResult, EvalRun
from app.store.pg import close_pg, init_pg, session_factory
from eval.metrics import (
    ItemVerdict,
    MetricStat,
    RunSummary,
    doc_recall,
    fact_match,
    is_refusal,
    numeric_match,
    page_recall,
    stat,
    unit_declared,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "eval" / "reports"

CATEGORY_LABEL = {
    "fact": "事实题",
    "table": "表格数值题",
    "multihop": "多跳推理",
    # 口径辨析：同一科目在合并与母公司两个口径下的数值不同。
    # 单独成类而不并入 table，是因为它考的是另一件事——
    # 不是能不能找到那个数，而是找到之后分不分得清它出自哪张表。
    "scope": "口径辨析",
    "refuse": "应拒答",
}


def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return out.stdout.strip() or None
    except Exception:
        return None


# ── 评估集导入 ──────────────────────────────────────────


async def load_dataset(path: Path, name: str) -> int:
    """把 JSONL 导入 PG。同名数据集整体替换，避免新旧题目混在一起。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    async with session_factory()() as session:
        existing = (
            await session.execute(select(EvalDataset).where(EvalDataset.name == name))
        ).scalar_one_or_none()
        if existing:
            await session.delete(existing)
            await session.flush()

        ds = EvalDataset(name=name, description=f"来自 {path.name}", item_count=len(rows))
        session.add(ds)
        await session.flush()

        for r in rows:
            session.add(
                EvalItem(
                    dataset_id=ds.id,
                    question=r["question"],
                    category=r["category"],
                    difficulty=r.get("difficulty"),
                    ground_truth=r.get("ground_truth"),
                    numeric_value=(
                        Decimal(str(r["numeric_value"])) if r.get("numeric_value") is not None else None
                    ),
                    unit=r.get("unit"),
                    metric_name=r.get("metric_name"),
                    expected_doc_keys=r.get("expected_doc_keys"),
                    expected_pages=r.get("expected_pages"),
                    source=r.get("source", "manual"),
                    verified=bool(r.get("verified", False)),
                    note=r.get("note"),
                )
            )
        await session.commit()
    return len(rows)


# ── 单题判定 ────────────────────────────────────────────


def judge(
    item: EvalItem,
    answer: str,
    refused: bool,
    uids: list[str],
    pages: list[int],
    tol: float,
) -> ItemVerdict:
    """按类别判定单题。全部为确定性规则，不调用模型。"""
    v = ItemVerdict()
    v.recall_at_k = doc_recall(uids, item.expected_doc_keys)
    v.page_recall = page_recall(pages, item.expected_pages)
    refusal = is_refusal(answer, refused)

    if item.category == "refuse":
        # 应拒答题：拒答即正确，给出答案即误答
        v.is_correct = refusal
        v.detail = "正确拒答" if refusal else "未拒答，构成误答"
        return v

    if refusal:
        v.is_correct = False
        v.detail = "不应拒答却拒答了"
        return v

    if item.numeric_value is not None:
        ok, declared, detail = numeric_match(answer, float(item.numeric_value), item.unit, tol)
        v.numeric_ok = ok
        v.unit_ok = unit_declared(answer, item.unit)
        # 量纲未声明则判错：答对数字却不说单位，读者无法分辨量级
        v.is_correct = ok and v.unit_ok
        v.detail = detail + ("" if v.unit_ok else "；未声明量纲")
        return v

    if item.ground_truth:
        ok, detail = fact_match(answer, item.ground_truth)
        v.is_correct = ok
        v.detail = detail
        return v

    v.is_correct = None
    v.detail = "无标准答案，跳过判定"
    return v


# ── 运行 ────────────────────────────────────────────────


# 瞬时网络错误重试几次。这些「调用失败」不是被测系统的缺陷，
# 却照样计入正确率——实测一次全量评估里会有 0–1 道题栽在
# `OpenAIConnectionError: Connection error.` 上，正好是运行间波动的量级，
# 不重试就分不清「模型答错了」和「网断了一下」。
# 只重试连接/超时类，模型返回的业务错误一律不重试，避免把真失败刷掉。
TRANSIENT_ERRORS = ("connection", "timeout", "timed out", "temporarily")
MAX_RETRIES = 2


def is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in TRANSIENT_ERRORS)


async def run_one(graph, item: EvalItem, exp: Experiment, tol: float) -> dict:
    started = time.perf_counter()
    retried = 0
    try:
        for attempt in range(MAX_RETRIES + 1):
            try:
                out = await graph.ainvoke(
                    new_state(
                        question=item.question, conv_id=f"eval-{item.id}", exp_id=exp.exp_id
                    )
                )
                break
            except Exception as exc:
                if attempt >= MAX_RETRIES or not is_transient(exc):
                    raise
                retried = attempt + 1
                await asyncio.sleep(2**attempt)
    except Exception as exc:
        return {
            "item": item,
            "answer": None,
            "error": f"{type(exc).__name__}: {exc}（已重试 {retried} 次）",
            "verdict": ItemVerdict(is_correct=False, detail="调用失败"),
            "pages": [],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "cost": 0.0,
            "uids": [],
            "refused": False,
        }

    answer = out.get("answer") or ""
    retrieved = out.get("retrieved") or []
    uids = [c.get("chunk_uid", "") for c in retrieved]
    # 页码取块覆盖的整个范围：跨页表格合并后一个块可能横跨两页
    pages: list[int] = []
    for c in retrieved:
        start, end = int(c.get("page_start") or 0), int(c.get("page_end") or 0)
        pages.extend(range(start, max(start, end) + 1) if start else [])
    verdict = judge(item, answer, bool(out.get("refused")), uids, pages, tol)
    return {
        "item": item,
        "answer": answer,
        "error": None,
        "verdict": verdict,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "cost": float((out.get("usage") or {}).get("cost_cny", 0.0)),
        "uids": uids,
        "pages": sorted(set(pages)),
        "refused": bool(out.get("refused")),
    }


def summarize(results: list[dict], rounds: int) -> RunSummary:
    summary = RunSummary()

    def collect(rs: list[dict]) -> dict[str, MetricStat]:
        out: dict[str, MetricStat] = {}
        correct = [1.0 if r["verdict"].is_correct else 0.0 for r in rs if r["verdict"].is_correct is not None]
        if correct:
            out["正确率"] = stat("正确率", correct, rounds)
        units = [1.0 if r["verdict"].unit_ok else 0.0 for r in rs if r["verdict"].unit_ok is not None]
        if units:
            out["量纲声明率"] = stat("量纲声明率", units, rounds)
        pr = [r["verdict"].page_recall for r in rs if r["verdict"].page_recall is not None]
        if pr:
            out["页码召回"] = stat("页码召回", pr, rounds)
        recalls = [r["verdict"].recall_at_k for r in rs if r["verdict"].recall_at_k is not None]
        if recalls:
            out["文档召回"] = stat("文档召回", recalls, rounds)
        return out

    summary.overall = collect(results)

    # 误答率单独报告：应拒答却作答是最危险的失败模式（架构 §1.3）
    refuse_items = [r for r in results if r["item"].category == "refuse"]
    if refuse_items:
        wrong = [0.0 if r["verdict"].is_correct else 1.0 for r in refuse_items]
        summary.overall["误答率"] = stat("误答率", wrong, rounds)

    for cat in CATEGORY_LABEL:
        subset = [r for r in results if r["item"].category == cat]
        if subset:
            summary.by_category[cat] = collect(subset)
    return summary


def render_report(
    run_id: int, exp: Experiment, dataset: str, results: list[dict], summary: RunSummary
) -> str:
    lines = [
        f"# 评估报告 · run {run_id}",
        "",
        f"- 实验：`{exp.exp_id}` — {exp.description}",
        f"- 评估集：`{dataset}`，{len(results)} 题",
        f"- 配置：分块={exp.chunking.strategy} 父子块={exp.chunking.parent_child} "
        f"检索={exp.retrieval.mode} 重排={exp.rerank.enabled}",
        f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- commit：`{(git_sha() or '')[:12]}`",
        "",
        "## 总体指标",
        "",
        "指标以 bootstrap 95% 置信区间报告。两组配置的区间重叠时不得声称有提升。",
        "",
        "| 指标 | 均值 | 95% CI | 样本数 |",
        "|---|---|---|---|",
    ]
    for name, m in summary.overall.items():
        lines.append(f"| {name} | {m.mean * 100:.1f}% | [{m.lo * 100:.1f}, {m.hi * 100:.1f}] | {m.n} |")

    lines += ["", "## 分类指标", "", "总体指标会被类别间差异稀释，必须分层看。", ""]
    lines += ["| 类别 | 正确率 | 95% CI | 样本数 |", "|---|---|---|---|"]
    for cat, metrics in summary.by_category.items():
        m = metrics.get("正确率")
        if m:
            lines.append(
                f"| {CATEGORY_LABEL[cat]} | {m.mean * 100:.1f}% | "
                f"[{m.lo * 100:.1f}, {m.hi * 100:.1f}] | {m.n} |"
            )

    total_cost = sum(r["cost"] for r in results)
    lat = sorted(r["latency_ms"] for r in results)
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0
    lines += [
        "",
        "## 工程指标",
        "",
        f"- 总成本：¥{total_cost:.4f}，单题均价 ¥{total_cost / max(len(results), 1):.5f}",
        f"- 延迟：中位 {lat[len(lat) // 2] if lat else 0} ms，P95 {p95} ms",
        "",
        "## 逐题明细",
        "",
        "| # | 类别 | 问题 | 判定 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        v = r["verdict"]
        mark = "✅" if v.is_correct else ("—" if v.is_correct is None else "❌")
        q = r["item"].question[:28]
        lines.append(f"| {i} | {CATEGORY_LABEL[r['item'].category]} | {q} | {mark} | {v.detail[:46]} |")

    wrong = [r for r in results if r["verdict"].is_correct is False]
    if wrong:
        lines += ["", "## 错题（用于归因）", ""]
        for r in wrong:
            lines += [
                f"**{r['item'].question}**",
                "",
                f"- 期望：{r['item'].ground_truth or '（应拒答）'}",
                f"- 实际：{(r['answer'] or '')[:200]}",
                f"- 判定：{r['verdict'].detail}",
            ]
            # 异常必须把原文带出来。不带的话，「调用失败」和「答错了」
            # 在报告里长得一模一样，排查时只能靠重跑去碰——
            # 实测两次崩溃都是这样被耽误的。
            if r.get("error"):
                lines.append(f"- 异常：`{r['error'][:300]}`")
            lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="seed", help="评估集名称")
    parser.add_argument("--load", type=str, default=None, help="先从 JSONL 导入评估集")
    parser.add_argument("--exp", type=str, default=None, help="实验配置 ID")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--concurrency", type=int, default=3, help="并发数")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="只跑指定类别（table/multihop/fact/refuse）。仅用于定向调试，结果不得与全量运行比较",
    )
    parser.add_argument("--note", type=str, default=None)
    parser.add_argument(
        "--with-cache",
        action="store_true",
        help="评估时开启缓存。默认关闭——开着会命中上一版代码写入的答案，"
        "测的就不再是系统本身",
    )
    args = parser.parse_args()

    exp = load_experiment(args.exp)
    # 评估一律关缓存。开着会出两个问题：
    #   1. 命中的是**上一版代码**写入的答案，于是评估测的是缓存不是系统，
    #      改了代码却看不出回归——exp_id 相同时这一点尤其隐蔽
    #   2. 每题多一次 embedding 写入，拖慢整轮
    # 想单独量缓存效果时用 --with-cache。
    if not args.with_cache:
        exp.cache.exact = False
        exp.cache.semantic = False
    await init_pg()
    await init_milvus()

    try:
        if args.load:
            n = await load_dataset(Path(args.load), args.dataset)
            print(f"已导入评估集 {args.dataset}：{n} 题")

        async with session_factory()() as session:
            ds = (
                await session.execute(select(EvalDataset).where(EvalDataset.name == args.dataset))
            ).scalar_one_or_none()
            if ds is None:
                print(f"评估集 {args.dataset} 不存在，先用 --load 导入")
                return 1
            stmt = (
                select(EvalItem)
                .where(EvalItem.dataset_id == ds.id, EvalItem.verified.is_(True))
                .order_by(EvalItem.id)
            )
            items = list((await session.execute(stmt)).scalars().all())
            unverified = (
                await session.execute(
                    select(func.count())
                    .select_from(EvalItem)
                    .where(EvalItem.dataset_id == ds.id, EvalItem.verified.is_(False))
                )
            ).scalar()

        if unverified:
            # 未复核的题目不参与评分：评估集本身错了，所有指标都是假的
            print(f"提示：{unverified} 题未复核，已排除")
        if args.category:
            items = [i for i in items if i.category == args.category]
            # 子集运行改变了评估集构成，指标与全量不可比，必须在记录里留痕
            print(f"仅评估 {args.category} 类（{len(items)} 题）——该结果不可与全量运行对比")
        if args.limit:
            items = items[: args.limit]
        if not items:
            print("没有可评估的题目")
            return 1

        print("=" * 64)
        print(f"  评估 {args.dataset}：{len(items)} 题   实验 {exp.exp_id}")
        print("=" * 64)

        async with session_factory()() as session:
            run = EvalRun(
                dataset_name=args.dataset,
                exp_id=exp.exp_id,
                config_snapshot=exp.model_dump(mode="json"),
                git_sha=git_sha(),
                note=(f"[subset:{args.category}] " if args.category else "") + (args.note or ""),
                item_count=len(items),
            )
            session.add(run)
            await session.commit()
            run_id = run.id

        graph = get_graph()
        sem = asyncio.Semaphore(args.concurrency)
        tol = exp.eval.numeric_tolerance

        async def guarded(it: EvalItem) -> dict:
            async with sem:
                r = await run_one(graph, it, exp, tol)
                v = r["verdict"]
                mark = "OK  " if v.is_correct else ("--  " if v.is_correct is None else "FAIL")
                print(f"  [{mark}] {it.question[:34]:36s} {v.detail[:40]}")
                return r

        results = await asyncio.gather(*(guarded(it) for it in items))
        results = sorted(results, key=lambda r: r["item"].id)

        summary = summarize(results, exp.eval.bootstrap_rounds)
        total_cost = sum(r["cost"] for r in results)

        async with session_factory()() as session:
            for r in results:
                v = r["verdict"]
                session.add(
                    EvalResult(
                        run_id=run_id,
                        item_id=r["item"].id,
                        category=r["item"].category,
                        answer=r["answer"],
                        refused=r["refused"],
                        retrieved_uids=r["uids"] or None,
                        is_correct=v.is_correct,
                        numeric_ok=v.numeric_ok,
                        unit_ok=v.unit_ok,
                        recall_at_k=v.recall_at_k,
                        page_recall=v.page_recall,
                        latency_ms=r["latency_ms"],
                        cost=Decimal(str(round(r["cost"], 6))),
                        error=r["error"],
                    )
                )
            db_run = await session.get(EvalRun, run_id)
            db_run.status = "finished"
            db_run.finished_at = datetime.now()
            db_run.total_cost = Decimal(str(round(total_cost, 6)))

            def dump(ms: dict[str, MetricStat]) -> dict:
                return {k: {"mean": m.mean, "lo": m.lo, "hi": m.hi, "n": m.n} for k, m in ms.items()}

            db_run.metrics = {
                "overall": dump(summary.overall),
                "by_category": {cat: dump(ms) for cat, ms in summary.by_category.items()},
            }
            await session.commit()

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = render_report(run_id, exp, args.dataset, results, summary)
        report_path = REPORT_DIR / f"run_{run_id:04d}_{exp.exp_id}.md"
        report_path.write_text(report, encoding="utf-8")

        print("\n" + "=" * 64)
        for name, m in summary.overall.items():
            print(f"  {name:12s} {m.format()}")
        print(f"\n  成本 ¥{total_cost:.4f}   报告 {report_path.relative_to(PROJECT_ROOT)}")
        return 0
    finally:
        await close_milvus()
        await close_pg()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
