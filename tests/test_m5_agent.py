"""Agent 护栏的单元测试。

护栏是 Agent 分支里最该被锁住的部分：它决定「跑飞了会怎样」。
端到端评估只能说明顺利时结果对，说明不了触顶时的行为——
而触顶的设计是**强制收敛作答，不是报错**（架构 §6.3），
这个选择必须有测试钉住，否则很容易被后续改动悄悄改成抛异常。

用假模型驱动，不触网。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.config.experiment import Experiment
from app.graph.nodes import agent as agent_mod
from app.graph.state import new_state


@dataclass
class FakeResponse:
    """最小的 AIMessage 替身。"""

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    response_metadata: dict = field(default_factory=dict)
    usage_metadata: dict | None = None


class FakeLLM:
    """按脚本逐轮返回。脚本用完后一律返回纯文本，模拟模型收敛。"""

    def __init__(self, script: list[FakeResponse]) -> None:
        self.script = list(script)
        self.calls: list[list] = []

    def bind_tools(self, tools):  # noqa: ARG002 - 签名对齐真实对象
        return self

    async def ainvoke(self, messages, config=None):  # noqa: ARG002
        self.calls.append(list(messages))
        if self.script:
            return self.script.pop(0)
        return FakeResponse(content="基于已查到的信息，结论是 X。")


@pytest.fixture
def patched(monkeypatch):
    """把 Agent 节点的外部依赖全部换成假的。"""

    async def fake_corpus():
        return [("600519", "贵州茅台", 2025)]

    monkeypatch.setattr(agent_mod, "available_corpus", fake_corpus)
    monkeypatch.setattr(agent_mod, "callbacks", lambda: [])
    monkeypatch.setattr(agent_mod, "trace_metadata", lambda **_kw: {})
    monkeypatch.setattr(agent_mod, "usage_from_response", lambda _r: (0, 0))

    def install(llm: FakeLLM, tools: list | None = None):
        monkeypatch.setattr(agent_mod, "get_chat", lambda *_a, **_kw: llm)
        monkeypatch.setattr(agent_mod, "build_tools", lambda *_a, **_kw: tools or [])
        return llm

    return install


def _state(exp_id: str = "exp00_debug") -> dict:
    return new_state(question="贵州茅台四个季度营收合计", conv_id="t", exp_id=exp_id)


@pytest.mark.asyncio
async def test_plain_answer_returns_immediately(patched):
    """模型不调工具时，第一轮的文本就是答案。"""
    llm = patched(FakeLLM([FakeResponse(content="营业收入 1,688.38 亿元 [第6页]。")]))
    out = await agent_mod.agent_node(_state(), {})
    assert "1,688.38" in out["answer"]
    assert out["tool_calls"] == 0
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_and_counted(patched, monkeypatch):
    """模型可能幻觉出不存在的工具名。

    不能让它直接把流程打断，而要把「没有这个工具」回传给模型，
    同时计入错误次数——否则模型可以无限次幻觉下去。
    """
    script = [
        FakeResponse(tool_calls=[{"name": "query_financials", "args": {}, "id": "1"}]),
        FakeResponse(content="改用检索后得到结论。"),
    ]
    patched(FakeLLM(script))
    out = await agent_mod.agent_node(_state(), {})
    assert out["tool_errors"] == 1
    assert out["scratchpad"][0]["tool"] == "query_financials"
    assert "没有名为" in out["scratchpad"][0]["result"]


@pytest.mark.asyncio
async def test_step_limit_forces_an_answer_instead_of_raising(patched, monkeypatch):
    """触及步数上限时强制收敛作答，而不是报错。

    用户宁可拿到「基于已查到的部分…」，也不要一句「超出步数限制」。
    这条测试同时钉住两件事：不抛异常，且 degraded 里留痕。
    """

    class _Tool:
        name = "calculate"
        coroutine = None

        def invoke(self, _args):
            return "1 + 1 = 2"

    exp = Experiment(exp_id="t")
    exp.agent.max_steps = 2
    monkeypatch.setattr(agent_mod, "experiment_of", lambda _s: exp)

    call = {"name": "calculate", "args": {"expression": "1+1"}, "id": "x"}
    # 脚本一直要求调工具，直到护栏把它拦下来
    patched(FakeLLM([FakeResponse(tool_calls=[call]) for _ in range(10)]), tools=[_Tool()])

    out = await agent_mod.agent_node(_state(), {})
    assert out["answer"].strip()  # 有答案，没有抛异常，也不是空串
    assert out["tool_calls"] >= exp.agent.max_steps
    assert any("agent_forced_answer" in d for d in out["degraded"])


@pytest.mark.asyncio
async def test_tool_error_limit_also_forces_an_answer(patched, monkeypatch):
    """连续工具失败同样走强制收敛，不是无限重试。"""
    exp = Experiment(exp_id="t")
    exp.agent.max_tool_errors = 2
    monkeypatch.setattr(agent_mod, "experiment_of", lambda _s: exp)

    bad = {"name": "no_such_tool", "args": {}, "id": "x"}
    patched(FakeLLM([FakeResponse(tool_calls=[bad]) for _ in range(10)]))

    out = await agent_mod.agent_node(_state(), {})
    assert out["answer"].strip()
    assert out["tool_errors"] >= exp.agent.max_tool_errors
    assert any("工具连续失败" in d for d in out["degraded"])


@pytest.mark.asyncio
async def test_text_form_tool_call_is_pushed_back(patched, monkeypatch):
    """把工具调用写成正文时要求重发，并计入错误次数。

    不拦的话这段原文会被当成最终回答返给用户——
    run 0025 返回过「calculate {"expression": ...}</tool_call>」，
    run 0032 返回过「调用 calculate 工具进行计算。」
    """
    script = [
        FakeResponse(content='calculate\n{"expression": "1+1"}\n</tool_call>'),
        FakeResponse(content="两者相差 26,959,446.43 元。"),
    ]
    patched(FakeLLM(script))
    out = await agent_mod.agent_node(_state(), {})
    assert out["tool_errors"] == 1
    assert "26,959,446.43" in out["answer"]
    assert "calculate" not in out["answer"]


@pytest.mark.asyncio
async def test_refused_state_short_circuits(patched):
    """上游已拒答时直接返回，不再调模型烧钱。"""
    llm = patched(FakeLLM([FakeResponse(content="不该被调用")]))
    state = _state()
    state["refused"] = True
    out = await agent_mod.agent_node(state, {})
    assert out == {}
    assert llm.calls == []


@pytest.mark.asyncio
async def test_forced_answer_is_never_empty(patched, monkeypatch):
    """护栏触顶后模型若仍只发工具调用、不给正文，不能返回空字符串。

    这个缺口是写本测试时才发现的：强制收敛的承诺是「给个交代」，
    端到端评估看不出来——真实模型几乎总会说点什么。
    """

    class _Tool:
        name = "calculate"
        coroutine = None

        def invoke(self, _args):
            return "1 + 1 = 2"

    exp = Experiment(exp_id="t")
    exp.agent.max_steps = 1
    monkeypatch.setattr(agent_mod, "experiment_of", lambda _s: exp)

    call = {"name": "calculate", "args": {"expression": "1+1"}, "id": "x"}
    # 模型从头到尾只发工具调用，一个字的正文都不给
    patched(FakeLLM([FakeResponse(tool_calls=[call]) for _ in range(10)]), tools=[_Tool()])

    out = await agent_mod.agent_node(_state(), {})
    assert out["answer"].strip()
    assert "未能得出结论" in out["answer"]
    assert "agent_empty_answer" in out["degraded"]
