"""Agent 工具集。

当前只实现多跳推理所必需的两个（架构 §5.5 的四工具中的前两个）：

    retrieve_report  按公司与年度检索年报内容
    calculate        数值计算

query_financials 依赖尚未落库的 financial_metrics（P1-4），
plot_chart 对评估无帮助——按决策阶梯，当前用不到的不写。

安全约定：calculate 必须是受限求值。把表达式交给 eval 等于
把任意代码执行权交给模型输出，这是最典型的注入面
（架构 §5.5 安全红线）。
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re

from langchain_core.tools import tool

from app.config.experiment import Experiment
from app.rag.retrievers import QueryFilters, expand_parents, rerank, retrieve
from app.store.milvus import milvus

logger = logging.getLogger(__name__)

# 夹在数字之间的逗号才是千分位分隔符
THOUSAND_SEP = re.compile(r"(?<=\d)[,，](?=\d)")

MAX_SNIPPET = 1200
MAX_CHUNKS_PER_CALL = 6


# ── 安全计算器 ──────────────────────────────────────────

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
}
# 指数运算可以用极小的表达式耗尽内存（如 9**9**9），必须设上限
MAX_POW_EXPONENT = 64


class UnsafeExpression(ValueError):
    """表达式包含不被允许的结构。"""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise UnsafeExpression(f"不支持的常量：{node.value!r}")
        # 保留 int 原类型：强制转 float 会让 round(x, 2) 的位数参数变成 2.0，
        # 而 round 的 ndigits 只接受整数
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise UnsafeExpression("指数过大")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"不支持的一元运算：{type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise UnsafeExpression("只允许调用 abs/round/min/max/sum/sqrt")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, ast.Tuple | ast.List):
        # 仅为支持 sum([...]) / max(...) 这类用法
        return [_eval_node(e) for e in node.elts]  # type: ignore[return-value]
    raise UnsafeExpression(f"不支持的语法：{type(node).__name__}")


def safe_eval(expression: str) -> float:
    """受限表达式求值。

    只放行数值字面量、四则与幂运算、以及白名单函数。
    不解析名称、属性、下标、推导式、lambda —— 这些是沙箱逃逸的常见入口。
    """
    # 只剥离数字中间的千分位逗号。直接 replace(",", "") 会连函数参数的
    # 分隔符一起删掉——round(x, 2) 会变成 round(x 2) 而报语法错误。
    cleaned = THOUSAND_SEP.sub("", expression).strip()
    if not cleaned:
        raise UnsafeExpression("表达式为空")
    if len(cleaned) > 400:
        raise UnsafeExpression("表达式过长")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"语法错误：{exc.msg}") from exc
    result = _eval_node(tree)
    if isinstance(result, list):
        raise UnsafeExpression("结果不是数值")
    return result


# ── 工具构造 ────────────────────────────────────────────


def build_tools(
    exp: Experiment,
    corpus: list[tuple[str, str, int]],
    collector: list[dict] | None = None,
):
    """按实验配置构造工具。

    工具需要绑定实验配置（检索参数随实验变化），因此用工厂而非模块级常量——
    全局配置会让不同实验的 Agent 用上同一套检索参数，这是 M4 踩过的坑。

    collector 收集每次检索命中的块（结构与 rag 分支的 state["retrieved"] 一致）。
    不收集的话，Agent 分支的页码召回会恒为 0 —— 指标不是「变差了」而是
    **根本没数据**，这种假指标比指标难看危险得多。
    """
    name_to_code = {name: code for code, name, _y in corpus}
    known_codes = {code for code, _n, _y in corpus}

    @tool
    async def retrieve_report(company: str, year: int, query: str) -> str:
        """检索某家公司某年度年报中与 query 相关的内容。

        company 可以是公司简称或六位股票代码；year 是数据所属年度；
        query 应当是具体的科目名或问题，例如「营业收入」「研发投入」。
        返回若干条带页码与单位的原文片段。
        """
        code = company.strip()
        if code not in known_codes:
            for cname, ccode in name_to_code.items():
                if cname and (cname in company or company in cname):
                    code = ccode
                    break
        if code not in known_codes:
            return f"未收录公司「{company}」。已收录：" + "、".join(f"{n}({c})" for c, n, _ in corpus)

        # 年份要并进**检索文本**，不能只当标量过滤条件。
        # 过滤条件走的是 report_year（文档年度），一份 2025 年报会同时匹配
        # 2023/2024/2025 三个查询年份，起不到区分作用；而「近三年主要会计数据」
        # 表把年份直接写在列头里（| 2025年 | 2024年 | 2023年 |），
        # 年份进了查询文本，BM25 才能把这张表排上来。
        # 实测：查「营业收入」p6 摘要表一次都进不了前 6，
        # 查「2023年营业收入」它就是第一名。
        search_text = f"{year}年{query}" if year else query
        chunks = await retrieve(
            milvus(),
            search_text,
            QueryFilters(company_codes=[code], years=[year] if year else []),
            exp,
        )
        if not chunks:
            return f"未检索到 {company} {year}年 关于「{query}」的内容。"

        chunks, _ = await rerank(chunks, search_text, exp)
        chunks, _ = await expand_parents(chunks, exp)
        if collector is not None:
            collector.extend(c.__dict__ for c in chunks[:MAX_CHUNKS_PER_CALL])

        parts = []
        for c in chunks[:MAX_CHUNKS_PER_CALL]:
            unit = f"（单位：{c.unit}）" if c.unit else ""
            head = c.heading_path or ""
            parts.append(f"[第{c.page_start}页{unit} {head}]\n{c.content[:MAX_SNIPPET]}")
        return "\n\n".join(parts)

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式，用于加总、求比率、算同比等。

        只支持数值与 + - * / ** % 及 abs/round/min/max/sqrt。
        例如：(168838102514.79 - 170899152276.34) / 170899152276.34 * 100
        """
        try:
            value = safe_eval(expression)
        except UnsafeExpression as exc:
            return f"无法计算：{exc}"
        except ZeroDivisionError:
            return "无法计算：除数为零"
        except Exception as exc:  # 溢出等
            return f"无法计算：{type(exc).__name__}"
        # 大额保留两位小数，比率类保留四位
        text = f"{value:,.2f}" if abs(value) >= 1000 else f"{value:,.4f}".rstrip("0").rstrip(".")
        return f"{expression} = {text}"

    return [retrieve_report, calculate]
