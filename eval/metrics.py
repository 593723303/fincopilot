"""评估指标。

主指标（架构 §12.2 / ADR-014）：端到端答案正确率，确定性判定，
不依赖 LLM 评判——没有评判噪声、成本近似为零、对外无需解释。

辅助指标：Recall@K 与 RAGAS 系列，仅用于归因，不作验收门槛。

所有指标以 bootstrap 95% 置信区间报告而非点估计：
样本量有限时点估计本身带误差，两组区间重叠就不能声称有提升（ADR-015）。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# 量纲归一到「元」。与解析侧保持一致，改动需同步。
UNIT_SCALE = {
    "元": 1.0,
    "千元": 1e3,
    "万元": 1e4,
    "百万元": 1e6,
    "亿元": 1e8,
    "万": 1e4,
    "亿": 1e8,
}

# 数值 + 紧随其后的可选量纲。允许千分位与小数，兼顾全角括号负数。
NUM_UNIT_PAT = re.compile(
    r"(-?[\d,，]+(?:\.\d+)?)\s*(百万元|千元|万元|亿元|亿|万|元)?",
)

# 拒答的表述方式比预想的多。首版只收了「未找到」类说法，
# 结果把「我无法提供投资建议」「未在已收录的年报中出现」判成了误答，
# 误答率虚高到 50%——评估器自身的缺陷会直接伪造指标，
# 所以这张表每次发现新表述都要补，并回跑历史用例确认没有反转。
REFUSAL_MARKERS = (
    "未找到",
    "没有找到",
    "未检索到",
    "没有检索到",
    "无法给出",
    "无法提供",
    "无法回答",
    "无法计算",
    "我无法",
    "不能提供",
    "不提供投资建议",
    "资料中未",
    "不包含",
    "无相关信息",
    "未收录",
    "未在已收录",
    "属于未来",
)


@dataclass
class ItemVerdict:
    """单题判定结果。"""

    is_correct: bool | None = None
    numeric_ok: bool | None = None
    unit_ok: bool | None = None
    recall_at_k: float | None = None
    detail: str = ""


@dataclass
class MetricStat:
    """一个指标的统计量，含置信区间。"""

    name: str
    mean: float
    lo: float
    hi: float
    n: int

    def format(self) -> str:
        return f"{self.mean * 100:.1f}% [{self.lo * 100:.1f}, {self.hi * 100:.1f}] (n={self.n})"


@dataclass
class RunSummary:
    overall: dict[str, MetricStat] = field(default_factory=dict)
    by_category: dict[str, dict[str, MetricStat]] = field(default_factory=dict)


# ── 数值处理 ────────────────────────────────────────────


def parse_number(raw: str) -> float | None:
    s = raw.replace(",", "").replace("，", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def to_yuan(value: float, unit: str | None) -> float | None:
    """按量纲折算为元。量纲未知时返回 None，不做猜测。"""
    if unit is None:
        return None
    scale = UNIT_SCALE.get(unit)
    return value * scale if scale else None


def extract_amounts(text: str) -> list[tuple[float, str | None]]:
    """抽出文本中的「数值 + 量纲」对。

    模型回答里可能同时出现原始数字与换算结果
    （如「22,146,581 千元（即 221.47 亿元）」），两者都要参与匹配——
    只要有一个与标准答案对得上就算答对。
    """
    out: list[tuple[float, str | None]] = []
    for m in NUM_UNIT_PAT.finditer(text):
        value = parse_number(m.group(1))
        if value is None:
            continue
        out.append((value, m.group(2)))
    return out


def numeric_match(
    answer: str,
    expected_value: float,
    expected_unit: str | None,
    tolerance: float = 0.005,
) -> tuple[bool, bool, str]:
    """数值题判定。

    返回 (数值是否正确, 是否声明了量纲, 说明)。

    量纲必须单独判定：答对数字但没说单位，在财报场景等于没回答——
    读者无法分辨 1,476.94 是元还是万元（ADR-016）。
    """
    target = to_yuan(expected_value, expected_unit)
    if target is None:
        target = expected_value  # 标准答案未标量纲时按原值比对

    candidates = extract_amounts(answer)
    if not candidates:
        return False, False, "回答中未出现数值"

    declared_unit = any(u for _v, u in candidates)
    best: tuple[float, float, str | None] | None = None  # (相对误差, 原值, 量纲)

    for value, unit in candidates:
        for normalized in filter(None, [to_yuan(value, unit), value]):
            err = abs(normalized) if target == 0 else abs(normalized - target) / abs(target)
            if err <= tolerance:
                return True, declared_unit, f"命中 {value}{unit or ''}"
            if best is None or err < best[0]:
                best = (err, value, unit)

    if best is None:
        return False, declared_unit, f"无可比数值，期望 {target}"
    # 报告真正最接近的候选而非第一个：回答里常混有年份等无关数字，
    # 拿第一个当「最接近」会把排查引向错误方向
    _err, value, unit = best
    return False, declared_unit, f"未命中，最接近 {value}{unit or ''}，期望 {target}"


# ── 文本与拒答判定 ──────────────────────────────────────


SENTENCE_END = re.compile(r"[。！？\n]")


def is_refusal(answer: str, refused_flag: bool) -> bool:
    """判断回答是否构成拒答。

    除系统显式拒答外，还要识别模型自己说「资料中未找到」的情况——
    后者同样是正确的拒答行为，不该被算作答错。

    关键：只看**第一句**。真正的拒答第一句就表态，而正常回答常在
    末尾附带能力说明，例如
    「贵州茅台的股票代码是600519。……但无法提供实时行情」。
    全文匹配会把这类正确回答误判为拒答（实测反转过两道题）；
    按固定字符数截取同样不可靠，句子长度差异太大。
    """
    if refused_flag:
        return True
    first = SENTENCE_END.split(answer.strip(), maxsplit=1)[0]
    return any(marker in first for marker in REFUSAL_MARKERS)


def fact_match(answer: str, ground_truth: str) -> tuple[bool, str]:
    """事实题判定。

    标准答案中的关键片段需全部出现在回答里。用关键片段而非整句比对，
    因为模型的表述方式不固定，整句匹配会把正确答案判错。
    """
    keys = [k.strip() for k in re.split(r"[、,，;；/]", ground_truth) if len(k.strip()) >= 2]
    if not keys:
        return ground_truth.strip() in answer, "整句比对"
    hit = [k for k in keys if k in answer]
    ok = len(hit) == len(keys)
    return ok, f"命中 {len(hit)}/{len(keys)} 个关键片段"


def unit_declared(answer: str, expected_unit: str | None) -> bool:
    """是否按要求声明了量纲。"""
    if not expected_unit:
        return True
    if expected_unit in answer:
        return True
    # 允许等价换算后的量纲
    return any(u in answer for u in UNIT_SCALE if UNIT_SCALE[u] >= UNIT_SCALE.get(expected_unit, 1))


# ── 检索指标 ────────────────────────────────────────────


def recall_at_k(retrieved_uids: list[str], expected_doc_keys: list[str] | None) -> float | None:
    """应命中的文档是否出现在检索结果中。

    以 doc_key 为粒度而非具体块：同一份年报里哪个块命中都算对，
    块粒度的标注成本高且不稳定（重新分块后块 ID 全变）。
    """
    if not expected_doc_keys:
        return None
    if not retrieved_uids:
        return 0.0
    hit = sum(1 for key in expected_doc_keys if any(key in uid for uid in retrieved_uids))
    return hit / len(expected_doc_keys)


# ── 统计 ────────────────────────────────────────────────


def bootstrap_ci(
    values: list[float], rounds: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float, float]:
    """bootstrap 重采样求均值与置信区间。

    seed 固定以保证同一批数据重复计算得到相同区间——
    评估结果必须可复现，否则无法判断指标变化来自改动还是随机性。
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean

    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(rounds * alpha / 2)]
    hi = means[min(rounds - 1, int(rounds * (1 - alpha / 2)))]
    return mean, lo, hi


def stat(name: str, values: list[float], rounds: int = 1000) -> MetricStat:
    mean, lo, hi = bootstrap_ci(values, rounds=rounds)
    return MetricStat(name=name, mean=mean, lo=lo, hi=hi, n=len(values))


def intervals_overlap(a: MetricStat, b: MetricStat) -> bool:
    """两个指标的置信区间是否重叠。

    重叠即不得声称有提升，只能记录「无显著差异」（ADR-015）。
    """
    return not (a.hi < b.lo or b.hi < a.lo)
