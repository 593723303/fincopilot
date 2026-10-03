"""Chainlit 对话前端。

    chainlit run frontend/app.py -w --port 8001

通过 SSE 调用后端，不直接调用图：前后端解耦既符合架构 §15 的部署形态，
也让事件协议（§8.2）真正被使用一次——协议只有被消费才知道设计得对不对。
"""

from __future__ import annotations

import json
import os

import chainlit as cl
import httpx

API_BASE = os.environ.get("FINCOPILOT_API", "http://127.0.0.1:8000")
STREAM_URL = f"{API_BASE}/api/v1/chat/stream"

WELCOME = """### FinCopilot · 财报问答

基于 A 股上市公司年报回答财务问题，答案附带**页码级引用**。

可以这样问：
- 贵州茅台 2025 年第一季度营业收入是多少
- 宁德时代 2025 年研发投入情况
- 茅台的销售费用同比变化

资料中没有依据的问题会明确拒答，不会编造数字。
"""


def _fmt_citation(c: dict) -> str:
    unit = f" · 单位：{c['unit']}" if c.get("unit") else ""
    head = c.get("heading") or "（无章节信息）"
    return (
        f"**[{c['idx']}]** {c.get('doc_key', '')} 第 {c.get('page', '?')} 页{unit}\n\n"
        f"*{head}*\n\n```\n{(c.get('snippet') or '')[:300]}\n```"
    )


@cl.on_chat_start
async def start() -> None:
    cl.user_session.set("conv_id", None)
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    conv_id = cl.user_session.get("conv_id")
    answer = cl.Message(content="")
    citations: list[dict] = []
    steps_msg: cl.Message | None = None
    done_info: dict = {}

    payload = {"question": message.content, "conv_id": conv_id}

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream("POST", STREAM_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    await cl.Message(content=f"请求失败（HTTP {resp.status_code}）：{body[:200]}").send()
                    return

                event = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip() or "{}")

                    if event == "meta":
                        cl.user_session.set("conv_id", data.get("conv_id"))

                    elif event == "route":
                        branch = {"rag": "检索年报", "chat": "直接回答", "agent": "多步分析"}.get(
                            data.get("branch", ""), data.get("branch", "")
                        )
                        steps_msg = cl.Message(content=f"*处理方式：{branch}*", author="系统")
                        await steps_msg.send()

                    elif event == "token":
                        await answer.stream_token(data.get("text", ""))

                    elif event == "citation":
                        citations.append(data)

                    elif event == "refused":
                        pass  # 文案会以 token 形式补推，此处不重复展示

                    elif event == "done":
                        done_info = data

                    elif event == "error":
                        await answer.stream_token(
                            f"\n\n> 处理出错（{data.get('code')}）：{data.get('message', '')}"
                        )
    except httpx.ConnectError:
        await cl.Message(
            content=f"连接不上后端服务（{API_BASE}）。请先启动：`uvicorn app.main:app`"
        ).send()
        return

    await answer.send()

    if citations:
        elements = [
            cl.Text(name=f"来源 [{c['idx']}]", content=_fmt_citation(c), display="side")
            for c in citations
        ]
        await cl.Message(content=f"引用了 {len(citations)} 处来源", elements=elements).send()

    if done_info:
        usage = done_info.get("usage") or {}
        bits = [
            f"耗时 {done_info.get('latency_ms', 0)} ms",
            f"token {usage.get('in', 0)}+{usage.get('out', 0)}",
            f"成本 ¥{usage.get('cost_cny', 0):.4f}",
        ]
        if done_info.get("first_token_ms"):
            bits.insert(1, f"首字 {done_info['first_token_ms']} ms")
        if done_info.get("degraded"):
            bits.append(f"降级 {','.join(done_info['degraded'])}")
        await cl.Message(content=f"*{' · '.join(bits)}*", author="系统").send()
