"""由 key_metrics.json 底稿生成评估集 v2。

    python -m scripts.build_eval_v2                 生成 eval/datasets/v2.jsonl
    python -m scripts.build_eval_v2 --sample 20     只打印抽样供人工核对

v1 只有两家公司 61 题，M5 结束时已连续两次满分——指标失去判别力
（open-issues.md P0-1）。v2 的目的是把语料扩到 12 家、题量扩到 200+，
重新获得判别力，并**检验那些规则到底是通用的还是只对茅台和宁德时代成立**。

三条出题纪律：

1. **只用通过原文对账的行**。底稿是解析器抽的，而被评估的系统用的是同一个
   解析器——不独立对账的话，解析器错了标准答案和系统回答会一起错。
2. **年份必须来自明写年份的列头**，不靠位置推断（M5 踩过这个坑）。
3. **同一公司同一科目若多年数值相同，整组弃用**——答对答错分不出来。

v1 的 61 题原样并入 v2：它们是人工核验过的，丢掉可惜。
但 v2 与 v1 的指标**不可比**，历史 run 不能和 v2 的 run 放在一张表里。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRAFT = PROJECT_ROOT / "tmp" / "key_metrics.json"
DATASETS = PROJECT_ROOT / "eval" / "datasets"

SEED = 42

# 语料外的公司：用于「应拒答」题。刻意选知名度高的，
# 模型最容易凭预训练知识直接作答。
#
# 两套名单的意义：v2 那一套在开发过程中被反复看过，
# 留出集若沿用，这 6 道题就不是「没见过的题」了。
# 建 v3 时传 --refuse holdout 换成另一组公司。
REFUSE_SETS = {
    "default": [
        ("五粮液", "2025年的营业收入"),
        ("比亚迪", "2025年的研发投入"),
        ("腾讯控股", "2025年的营业收入"),
        ("小米集团", "2025年的净利润"),
        ("京东方A", "2025年的营业收入"),
        ("中国中免", "2025年的归母净利润"),
    ],
    # v4 用第三组。每建一次留出集就得换一组——沿用旧的等于
    # 把「已经看过的题」混进留出集，它就不再是留出集了。
    "holdout2": [
        ("药明康德", "2025年的营业收入"),
        ("兆易创新", "2025年的归母净利润"),
        ("中国太保", "2025年的营业收入"),
        ("三花智控", "2025年的研发投入"),
        ("韦尔股份", "2025年的净利润"),
        ("爱尔眼科", "2025年的营业收入"),
    ],
    "holdout": [
        ("海天味业", "2025年的营业收入"),
        ("顺丰控股", "2025年的归母净利润"),
        ("中信证券", "2025年的营业收入"),
        ("长城汽车", "2025年的研发投入"),
        ("立讯精密", "2025年的净利润"),
        ("牧原股份", "2025年的营业收入"),
    ],
}
OUT_OF_CORPUS = REFUSE_SETS["default"]


def load_draft() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    if not DRAFT.exists():
        print(f"底稿不存在：{DRAFT}")
        print("先执行 python -m scripts.extract_key_metrics")
        sys.exit(1)
    data = json.loads(DRAFT.read_text(encoding="utf-8"))
    return data["metrics"], data["scope_pairs"]


def company_names() -> dict[str, str]:
    """从 data/raw 的文件名取公司简称，避免再查一次库。"""
    out: dict[str, str] = {}
    for pdf in (PROJECT_ROOT / "data" / "raw").glob("*.pdf"):
        parts = pdf.name.split("_")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def usable(rows: list[dict]) -> list[dict]:
    """过滤出可用于出题的行。"""
    rows = [r for r in rows if r.get("corroborated") and r.get("unit")]
    # 同一 (科目, 数值) 跨年重复 → 年份分不出来，整组弃用
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_metric[r["metric"]].append(r)
    keep: list[dict] = []
    for group in by_metric.values():
        values = [r["value"] for r in group]
        if len(set(values)) != len(values):
            continue
        keep.extend(group)
    return keep


def table_items(code: str, name: str, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "question": f"{name}{r['year']}年{r['metric']}是多少？",
                "category": "table",
                "difficulty": "单表直读",
                "ground_truth": f"{r['raw']}{r['unit']}",
                "numeric_value": r["value"],
                "unit": r["unit"],
                "metric_name": r["metric"],
                "expected_doc_keys": [f"{code}_{max(x['year'] for x in rows)}_annual"],
                "expected_pages": [r["page"]],
                "source": "auto_extracted",
                "verified": True,
                "note": f"{r['heading']} 第{r['page']}页，列头年份 {r['year']}；原文行：{r['source_line']}",
            }
        )
    return out


def multihop_items(code: str, name: str, rows: list[dict]) -> list[dict]:
    """同比变化题：同一科目取相邻两年。"""
    by_metric: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_metric[r["metric"]][r["year"]] = r
    doc_year = max(r["year"] for r in rows)
    out = []
    for metric, per_year in by_metric.items():
        years = sorted(per_year)
        if len(years) < 2:
            continue
        cur, prev = years[-1], years[-2]
        a, b = per_year[cur]["value"], per_year[prev]["value"]
        if b == 0:
            continue
        pct = round((a - b) / abs(b) * 100, 2)
        basis = f"({per_year[cur]['raw']} - {per_year[prev]['raw']}) / {per_year[prev]['raw']}"
        out.append(
            {
                "question": f"{name}{cur}年{metric}比{prev}年变化了百分之多少？",
                "category": "multihop",
                "difficulty": "跨列计算",
                "ground_truth": f"约{pct}%",
                "numeric_value": pct,
                "unit": "%",
                "metric_name": f"{metric}同比",
                "expected_doc_keys": [f"{code}_{doc_year}_annual"],
                "expected_pages": sorted({per_year[cur]["page"], per_year[prev]["page"]}),
                "source": "auto_extracted",
                "verified": True,
                "note": f"{basis} = {pct}%",
            }
        )
    return out


def scope_items(code: str, name: str, pairs: list[dict], doc_year: int) -> list[dict]:
    """口径辨析题：同一科目问合并、再问母公司。

    这是本项目的核心考点。两张报表在 ADR-019 之前解析出的块逐字相同，
    模型只能猜；成对出题使「蒙对一个」无法得分——两道都答对才说明
    它真的看了口径。
    """
    out = []
    for pair in pairs:
        for scope in ("合并", "母公司"):
            row = pair[scope]
            out.append(
                {
                    "question": f"{name}{doc_year}年{scope}报表口径的{pair['metric']}是多少？",
                    "category": "scope",
                    "difficulty": "口径辨析",
                    "ground_truth": f"{row['raw']}{row['unit']}",
                    "numeric_value": row["value"],
                    "unit": row["unit"],
                    "metric_name": f"{scope}{pair['metric']}",
                    "expected_doc_keys": [f"{code}_{doc_year}_annual"],
                    # 页码取整张表的区间：跨页表的目标行常落在续页上，
                    # 只标首页会把本来召回正确的检索记成失败
                    "expected_pages": list(
                        range(row["page"], max(row.get("page_end") or 0, row["page"]) + 1)
                    ),
                    "source": "auto_extracted",
                    "verified": True,
                    "note": (
                        f"{row['heading']} 第{row['page']}页；"
                        f"另一口径为 {pair['合并' if scope == '母公司' else '母公司']['raw']}，"
                        "两者不可混用"
                    ),
                }
            )
    return out


def refuse_items(names: set[str], variant: str = "default") -> list[dict]:
    out = []
    for company, what in REFUSE_SETS[variant]:
        if company in names:
            continue  # 已入库就不能再当「应拒答」
        out.append(
            {
                "question": f"{company}{what}是多少？",
                "category": "refuse",
                "difficulty": "语料外公司",
                "ground_truth": "（应拒答）",
                "expected_doc_keys": [],
                "expected_pages": [],
                "source": "manual",
                "verified": True,
                "note": f"{company} 不在已入库语料中，应明确说明未收录，不得凭预训练知识作答",
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="只抽样打印 N 条供人工核对，不写文件")
    ap.add_argument("--per-company", type=int, default=12, help="每家公司最多出多少道表格题")
    ap.add_argument("--out", default="v2", help="输出到 eval/datasets/<name>.jsonl")
    ap.add_argument(
        "--merge",
        default="v1",
        help="把该评估集的题目并进来；留出集必须传 none",
    )
    ap.add_argument(
        "--refuse",
        default="default",
        choices=sorted(REFUSE_SETS),
        help="拒答题用哪一组占位公司。留出集传 holdout，避免沿用已看过的题",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只用这些股票代码出题。建留出集时用它把开发期间用过的公司排除干净",
    )
    args = ap.parse_args()

    rng = random.Random(SEED)
    draft, scope_pairs = load_draft()
    names = company_names()

    items: list[dict] = []
    stats: list[tuple[str, int, int, int]] = []
    for code, rows in sorted(draft.items()):
        if args.only and code not in args.only:
            continue
        name = names.get(code, code)
        keep = usable(rows)
        if not keep:
            stats.append((f"{code} {name}", 0, 0, 0))
            continue
        tables = table_items(code, name, keep)
        rng.shuffle(tables)
        tables = tables[: args.per_company]
        multis = multihop_items(code, name, keep)
        rng.shuffle(multis)
        multis = multis[:4]
        doc_year = max(r["year"] for r in keep)
        scopes = scope_items(code, name, scope_pairs.get(code, []), doc_year)
        items.extend(tables + multis + scopes)
        stats.append((f"{code} {name}", len(tables), len(multis), len(scopes)))

    items.extend(refuse_items(set(names.values()), args.refuse))

    # 并入已有评估集。**留出集必须传 --merge none**：
    # 混进开发期间看过的题，它就不再是留出集了。
    merged = items
    if args.merge and args.merge != "none":
        src = DATASETS / f"{args.merge}.jsonl"
        raw = src.read_text(encoding="utf-8").splitlines()
        old = [json.loads(ln) for ln in raw if ln.strip()]
        seen = {it["question"] for it in items}
        merged = items + [it for it in old if it["question"] not in seen]

    print(f"{'公司':<18}{'表格题':>8}{'多跳题':>8}{'口径题':>8}")
    for label, n_tab, n_multi, n_scope in stats:
        print(f"{label:<18}{n_tab:>8}{n_multi:>8}{n_scope:>8}")
    by_cat: dict[str, int] = defaultdict(int)
    for it in merged:
        by_cat[it["category"]] += 1
    carried = len(merged) - len(items)
    note = f"（从 {args.merge} 带入 {carried} 题）" if carried else "（未并入已有题目）"
    print(f"\n合计 {len(merged)} 题{note}：{dict(by_cat)}")

    if args.sample:
        print(f"\n── 随机抽样 {args.sample} 条，逐条核对页码与口径 ──")
        for it in rng.sample([i for i in merged if i["source"] == "auto_extracted"], args.sample):
            print(f"\nQ: {it['question']}")
            print(f"   答案 {it['ground_truth']}  页码 {it['expected_pages']}")
            print(f"   依据 {it['note'][:150]}")
        return 0

    out_path = DATASETS / f"{args.out}.jsonl"
    body = "\n".join(json.dumps(it, ensure_ascii=False) for it in merged)
    out_path.write_text(body + "\n", encoding="utf-8")
    print(f"\n写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
