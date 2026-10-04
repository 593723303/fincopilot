"""实验配置隔离测试。

这组测试防的是一个曾让三组消融实验全部作废的缺陷：
评估入口按 --exp 加载配置，但图节点内部调用无参的 load_experiment()，
读的是环境变量 EXP_ID。结果 --exp 只影响评估记录、不影响实际运行，
三组实验跑的全是同一个基线。

它不报错、不告警，表现只是「几组实验指标几乎一模一样」——
而这种表现极易被误读为「这些优化都没用」。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from app.config.experiment import available_experiments, load_experiment
from app.graph.state import experiment_of, new_state

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── 配置必须随请求流转 ──────────────────────────────────


def test_state_carries_experiment_id():
    st = new_state(question="x", conv_id="c", exp_id="exp04_wider")
    assert st["exp_id"] == "exp04_wider"


def test_experiment_of_respects_state():
    st = new_state(question="x", conv_id="c", exp_id="exp00_debug")
    assert experiment_of(st).exp_id == "exp00_debug"


def test_different_states_get_different_configs():
    """同一进程内并发处理不同实验的请求时，配置不能互相串。"""
    a = experiment_of(new_state(question="x", conv_id="c", exp_id="exp02_heading"))
    b = experiment_of(new_state(question="x", conv_id="c", exp_id="exp04_wider"))
    assert a.exp_id != b.exp_id
    assert a.rerank.top_n != b.rerank.top_n


def test_experiment_of_falls_back_to_default():
    """未指定时回落到环境变量配置，保证普通请求仍可用。"""
    st = new_state(question="x", conv_id="c")
    assert experiment_of(st).exp_id  # 不抛异常且有值


# ── 静态检查：节点内禁止使用全局配置 ────────────────────


def test_graph_nodes_never_read_global_experiment():
    """图内代码一律不得调用无参 load_experiment()。

    这是上述缺陷的根因。用静态检查而非行为测试，是因为
    行为测试只能覆盖已知路径，而新增节点时最容易顺手写成全局调用。

    必须用 AST 而非字符串匹配：注释与文档字符串里会提到这个函数名
    （本仓库就有多处说明性提及），字符串匹配会把它们全部误报。
    """
    offenders = []
    for path in (PROJECT_ROOT / "app" / "graph").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "load_experiment"
                and not node.args
                and not node.keywords
            ):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
    assert not offenders, "图节点必须用 experiment_of(state)，不得读全局配置：\n" + "\n".join(offenders)


# ── 实验配置之间的差异必须真实存在 ──────────────────────


def test_experiments_actually_differ():
    """若两个实验的关键参数完全相同，对比就没有意义。"""
    e02 = load_experiment("exp02_heading")
    e04 = load_experiment("exp04_wider")
    assert (e02.rerank.top_n, e02.parent_expansion.token_budget) != (
        e04.rerank.top_n,
        e04.parent_expansion.token_budget,
    )


def test_rerank_experiment_differs_only_in_rerank():
    """单因子实验必须真的只差一个因子，否则结论无法归因。"""
    base = load_experiment("exp04_wider")
    rr = load_experiment("exp05_rerank_wide")
    assert rr.rerank.enabled != base.rerank.enabled
    assert rr.chunking.strategy == base.chunking.strategy
    assert rr.retrieval.mode == base.retrieval.mode
    assert rr.rerank.top_n == base.rerank.top_n
    assert rr.generation.profile == base.generation.profile


def test_model_experiment_differs_only_in_profile():
    base = load_experiment("exp04_wider")
    cheap = load_experiment("exp00_debug")
    assert cheap.generation.profile != base.generation.profile
    assert cheap.rerank.top_n == base.rerank.top_n
    assert cheap.parent_expansion.token_budget == base.parent_expansion.token_budget
    assert cheap.retrieval.mode == base.retrieval.mode


def test_chunking_experiment_differs_only_in_strategy():
    base = load_experiment("exp04_wider")
    fixed = load_experiment("exp06_fixed")
    assert fixed.chunking.strategy != base.chunking.strategy
    assert fixed.rerank.top_n == base.rerank.top_n
    assert fixed.parent_expansion.token_budget == base.parent_expansion.token_budget
    assert fixed.generation.profile == base.generation.profile


@pytest.mark.parametrize("name", ["exp02_heading", "exp04_wider", "exp00_debug"])
def test_threshold_matches_scoring_mode(name):
    """阈值的量纲随判分方式而变，必须各自校准。

    RRF 分值（k=60）理论范围：仅一路命中 1/61≈0.0164，两路都第一 2/61≈0.0328。
    沿用余弦相似度的 0.35 会导致全部拒答；重排分数又是另一个量级。
    """
    exp = load_experiment(name)
    if exp.retrieval.mode == "hybrid" and not exp.rerank.enabled:
        assert exp.generation.relevance_threshold < 0.0164, (
            f"{name} 的阈值高于 RRF 单路命中下限，会把仅一路命中的结果全部拒掉"
        )


def test_all_experiments_loadable():
    """任何一个配置文件写错都会在消融时才暴露，这里提前拦住。"""
    for name in available_experiments():
        exp = load_experiment(name)
        assert exp.exp_id == name
        assert exp.description, f"{name} 缺少说明，实验记录表里无法追溯其含义"
