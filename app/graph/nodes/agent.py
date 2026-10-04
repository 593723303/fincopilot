"""Agent 分支：ReAct 循环 + 护栏。

用于多跳问题——跨表加总、同比计算、跨公司对比。这类问题单次检索
拿不到答案，基线中「四个季度营收加总」就因此失败。

护栏的设计原则（架构 §6.3）：**触顶时强制收敛作答，而不是报错**。
用户宁可拿到「基于已查到的部分，结论是…」，也不要一句「超出步数限制」。

循环的中间消息不写进 state.messages：那是会话历史，
掺进工具调用记录会让后续轮次的上下文迅速膨胀，且对用户无意义。
中间过程记在 scratchpad 里，供展示与排查。
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs

from app.graph.nodes.query import available_corpus
from app.graph.nodes.rag import FINAL_ANSWER_TAG, build_citations
from app.graph.state import GraphState, experiment_of
from app.graph.tools import build_tools
from app.observability.cost import TokenUsage, merge_usage, usage_from_response
from app.observability.tracing import callbacks, trace_metadata
from app.providers.registry import get_chat
from app.rag.retrievers import RetrievedChunk

logger = logging.getLogger(__name__)

# 两种「说了要调用、其实没调用」的形态：
#   1. 把工具调用当正文吐出来（带 </tool_call> 标签或裸露的参数 JSON）
#   2. 在正文里宣告「调用 calculate 工具进行计算。」然后就结束了
# 后者实测发生在格力的同比题上：两个数都查对了、公式也列了，
# 最后一句是「调用 calculate 工具进行计算。」——没有结果，答案等于没给。
# 判据用一条不变量：**给用户看的最终回答里永远不该出现内部工具名**。
MALFORMED_CALL = re.compile(
    r'</?tool_call>|<\|tool_call\|>'
    r'|^\s*\{\s*"(expression|company|query)"\s*:'
    r'|\bcalculate\b|\bretrieve_report\b',
    re.M,
)

_SYSTEM = """你是财报分析助手，可以调用工具查阅年报并做计算。

工作方式：
1. 先想清楚回答这个问题需要哪些数据，再逐项用 retrieve_report 查
2. **任何算术都必须走 calculate 工具**，包括看起来很简单的加减和除法。
   心算的中间步骤写得再详细，最后那个数字也经常是错的
3. 数据齐全后给出结论，结论里的数字要与 calculate 的返回值一致

硬性要求：
- **涉及数值必须写明单位与科目全称**，例如「营业总收入 1,476.94 万元」。
  **单位与数字都照抄检索结果的原文，不要换算成亿元/万元**——
  实测换算错过 10 倍（把 31,629,416,193.83 元写成 31.63 亿元）。
  唯一需要换算的场合是跨公司比较（各家量纲不同，有的用元有的用千元），
  这时用 calculate 工具算，并把原值和换算后的值都写出来
- 科目名称按原文写全，「归属于上市公司股东的净利润」不可简写为「净利润」
- **财务数据默认取合并报表口径**。章节名含「母公司」的表不要用，
  除非问题明确问母公司。实测经营活动现金流净额两个口径相差三倍，
  取错口径算出来的同比完全是另一个数
- **三年并列表要按列头确认年度**。年报的「主要会计数据」通常并排列出
  本年 / 上年 / 前年三列，第二列是上一年而不是你要的那一年。
  取数前先在原文里核对该列的年份标注，核对不了就换个检索词再查一次。
  表格出现「重述后/重述前」「调整后/调整前」子列头时，往年的列被拆成两列，
  数据行的列数比表头多，中间各列对应哪一年无法确定——
  **只有最左边那个数值列可以确定是本年**，问的若不是本年就说明无法确定
- **缺的数据只能靠检索补，不得用其他指标反推**。例如不要用每股收益
  乘以股本去倒算净利润、不要用净资产收益率去倒算扣非净利润——
  这类推算出来的数字看着合理但几乎都是错的。查两次仍找不到就
  直接说明「所提供资料中未找到」
- 「营业收入」与「营业总收入」是**两个不同科目**，金额不同，
  问哪个就取哪个，不要互相替代；同理「净利润」与「归属于上市公司股东的净利润」
- 问「同比/环比变化了多少」「增长了多少」时，**答案给百分比**，
  需要的话再补一句绝对额；只给绝对额等于没回答这个问题
- **问题里的期间或科目不存在时要点明**，不得用相近的数据顶替。
  一年只有四个季度，问「第五季度」就回答不存在，不要给第四季度的数
- 标注数据来源页码，页码以检索结果中 [第N页] 的标注为准，不要凭印象写
- **比较优劣、给出选择建议的问题一律不答**：「哪只股票更值得投资」
  「该不该买」「哪家更有前景」——直接说明不提供投资建议即可，
  **不要先把两家的财务数据罗列一遍再说不给建议**，罗列本身就构成了倾向性。
  客观的单项数值对比（「哪家营业收入更高」）可以答，因为它只是读数。

{corpus_hint}"""

_FORCE_ANSWER = (
    "已达到本次分析的步数上限。请基于目前已经查到的信息给出结论，"
    "并明确说明哪些部分因资料不足而未能完成。不要再调用工具。"
)


def _corpus_hint(corpus: list[tuple[str, str, int]]) -> str:
    if not corpus:
        return "当前没有可用语料。"
    lines = "\n".join(f"- {name}（{code}） {year}年年报" for code, name, year in corpus)
    return f"可查阅的年报：\n{lines}\n注意：年报通常包含最近三年数据。"


async def agent_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """ReAct 循环。

    每一轮都先检查护栏再决定是否继续，而不是跑完才发现超限——
    后者会白白花掉最后一次调用的成本。
    """
    if state.get("refused"):
        return {}

    exp = experiment_of(state)
    cfg = exp.agent
    profile = exp.generation.profile
    corpus = await available_corpus()
    retrieved: list[dict] = []
    tools = build_tools(exp, corpus, collector=retrieved)
    tool_map = {t.name: t for t in tools}

    llm = get_chat(profile, stream_usage=True).bind_tools(tools)
    messages = [
        SystemMessage(content=_SYSTEM.format(corpus_hint=_corpus_hint(corpus))),
        HumanMessage(content=state.get("rewritten") or state["question"]),
    ]

    usage = TokenUsage()
    scratchpad: list[dict] = []
    degraded = list(state.get("degraded") or [])
    steps = 0
    errors = 0
    forced = False

    while True:
        # 护栏在调用前检查：超限还调一次是白花钱
        over_steps = steps >= cfg.max_steps
        over_errors = errors >= cfg.max_tool_errors
        over_budget = usage.cost_cny >= cfg.budget_cny
        if (over_steps or over_errors or over_budget) and not forced:
            reason = "步数上限" if over_steps else ("工具连续失败" if over_errors else "成本预算")
            logger.info("Agent 触及护栏（%s），强制收敛作答", reason)
            degraded.append(f"agent_forced_answer:{reason}")
            messages.append(HumanMessage(content=_FORCE_ANSWER))
            forced = True

        merged = merge_configs(
            config,
            {
                "callbacks": callbacks(),
                # 最后一轮（不再有工具调用）才是给用户看的回答
                "tags": [FINAL_ANSWER_TAG] if forced else [],
                "metadata": trace_metadata(
                    node="agent", profile=profile, step=steps, conv_id=state.get("conv_id")
                ),
            },
        )
        response = await llm.ainvoke(messages, config=merged)
        usage.add(profile, *usage_from_response(response))
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        text = response.content if isinstance(response.content, str) else str(response.content)

        # 小模型偶尔把工具调用当正文吐出来（文本里出现 </tool_call> 和参数 JSON），
        # 结构化的 tool_calls 却是空的。不拦的话这段原文会被当成最终回答返给用户，
        # 实测 run 0025 就返回了一句「calculate {"expression": ...}</tool_call>」。
        # 按一次工具失败计数，交给护栏收敛，不会无限重试。
        if not tool_calls and not forced and MALFORMED_CALL.search(text):
            errors += 1
            logger.warning("Agent 输出了文本形式的工具调用，要求重发（第 %d 次）", errors)
            messages.append(
                HumanMessage(
                    content="上一步把工具调用写成了正文，并没有真正调用工具。"
                    "请改用工具调用功能重新发起；如果数据已经齐了，就直接给出结论。"
                )
            )
            continue

        if not tool_calls or forced:
            # 护栏触顶后模型可能仍然只发工具调用、不给正文，这时 text 是空串，
            # 用户拿到的就是一片空白。强制收敛的承诺是「给个交代」，
            # 不是「给个空字符串」——兜一句，并说明已经查了什么。
            if not text.strip():
                looked_up = "；".join(
                    f"{s['tool']}({str(s['args'])[:40]})" for s in scratchpad[-3:]
                )
                text = (
                    "本次分析未能得出结论：已达到步数或成本上限，且模型未给出最终回答。"
                    + (f"已查询：{looked_up}。" if looked_up else "")
                    + "可以把问题拆小一些再问，例如先单独查其中一项数据。"
                )
                degraded.append("agent_empty_answer")
                logger.warning("Agent 强制收敛后仍无正文，已用兜底文案")
            logger.info("Agent 完成：%d 步，成本 ¥%.5f", steps, usage.cost_cny)
            return {
                "answer": text,
                "retrieved": retrieved,
                "citations": build_citations([RetrievedChunk(**d) for d in retrieved]),
                "messages": [AIMessage(content=text)],
                "tool_calls": steps,
                "tool_errors": errors,
                "scratchpad": scratchpad,
                "degraded": degraded,
                "usage": merge_usage(state.get("usage"), usage),
            }

        for call in tool_calls:
            steps += 1
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            tool = tool_map.get(name)
            if tool is None:
                # 工具白名单：模型可能幻觉出不存在的工具名
                errors += 1
                result = f"没有名为 {name} 的工具，可用：{', '.join(tool_map)}"
            else:
                try:
                    result = await tool.ainvoke(args) if tool.coroutine else tool.invoke(args)
                    if isinstance(result, str) and result.startswith(("未检索到", "未收录", "无法计算")):
                        errors += 1
                except Exception as exc:
                    errors += 1
                    # 把失败原因回传给模型，让它换个工具或换个参数，而不是直接中断
                    result = f"工具执行失败：{type(exc).__name__}: {exc}"
                    logger.warning("工具 %s 失败：%s", name, exc)

            result_text = str(result)
            scratchpad.append({"step": steps, "tool": name, "args": args, "result": result_text[:300]})
            messages.append(ToolMessage(content=result_text, tool_call_id=call.get("id", "")))
