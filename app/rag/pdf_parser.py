"""年报 PDF 解析。

设计依据来自 scripts/probe_pdf.py 对真实年报的探查，不是假设：

  - 75–85% 的页面含表格 → 表格处理是主线而非边缘分支
  - 量纲存在跨公司系统性差异（茅台全用「元」，宁德时代全用「千元」）
    → 不归一的话跨公司对比会差 1000 倍
  - 仅 52–64% 的表格上方有「单位：」声明
    → 单位抽取必须多级回溯，不能只看表格正上方
  - 「第X节」正则在某份年报上识别到 0 个章节
    → 标题路径改为同时匹配「一、」「（一）」等次级标题，实测覆盖率 100%。
      曾考虑按字号自适应判定标题，实测正则已足够，故未引入（决策阶梯）
  - 单元格内含换行（「归属于上市公司股东的 净利润」）
    → 科目名必须规范化，否则同一科目会被当成不同词

核心约定：**单位要写进块的正文，而不只是存元数据**——模型只读文本。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# ── 正则 ────────────────────────────────────────────────
# 金额量纲：可折算为元，用于跨公司对比
AMOUNT_UNITS = "元|千元|万元|亿元|百万元"
# 非金额单位：年报中大量存在（持股表用股、占比表用 %），
# 若把它们当成金额量纲，数值会被完全误读 —— 实测董监高持股表
# 正文明写「单位：股」，却被文档主导量纲「元」覆盖。
OTHER_UNITS = "股|份|件|吨|千克|平方米|人|个|%"
UNIT_PAT = re.compile(rf"单位[：:\s]*(?:人民币)?\s*({AMOUNT_UNITS}|{OTHER_UNITS})")
AMOUNT_UNIT_PAT = re.compile(rf"单位[：:\s]*(?:人民币)?\s*({AMOUNT_UNITS})")
CURRENCY_PAT = re.compile(r"币种[：:\s]*([一-龥]{2,6})")
# 判断单元格是否为数值：含千分位、小数、负号或括号负数
NUMERIC_CELL = re.compile(r"^[-－(（]?[\d,，]+(\.\d+)?[)）%]?$")
SECTION_PAT = re.compile(r"^第[一二三四五六七八九十]{1,3}节\s*(.+)$")
SUBSEC_PAT = re.compile(r"^([一二三四五六七八九十]{1,3})[、．]\s*(.+)$")
ITEM_PAT = re.compile(r"^[（(]([一二三四五六七八九十]{1,3})[）)]\s*(.+)$")
# 目录页特征：标题后跟一长串点号与页码
TOC_PAT = re.compile(r"\.{6,}\s*\d+\s*$")

# 量纲归一到「元」的倍数
UNIT_SCALE = {"元": 1, "千元": 1_000, "万元": 10_000, "百万元": 1_000_000, "亿元": 100_000_000}

# 表格上方回溯的高度（pt）。实测单位声明通常紧邻表格，
# 但中间常混有「□适用√不适用」等无关行，所以要限定范围并逐行筛。
LOOKUP_ABOVE = 140


@dataclass
class TableContext:
    """一个表格的量纲与口径上下文。"""

    unit: str | None = None
    currency: str | None = None
    caption: str | None = None
    # 单位的来源层级，用于排查数值题错误时归因
    unit_source: str | None = None  # above | header | page | inherited


@dataclass
class ParsedBlock:
    """解析产出的最小单元：一段正文或一个表格。"""

    kind: str  # text | table
    text: str
    page_start: int
    page_end: int
    heading_path: str = ""
    unit: str | None = None
    currency: str | None = None
    caption: str | None = None
    unit_source: str | None = None
    flags: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0


@dataclass
class ParsedDocument:
    path: Path
    page_count: int
    blocks: list[ParsedBlock]
    doc_unit: str | None = None  # 文档主导量纲，供缺省继承

    @property
    def table_blocks(self) -> list[ParsedBlock]:
        return [b for b in self.blocks if b.kind == "table"]

    @property
    def text_blocks(self) -> list[ParsedBlock]:
        return [b for b in self.blocks if b.kind == "text"]


# ── 文本规范化 ──────────────────────────────────────────


def normalize_cell(value: str | None) -> str:
    """单元格规范化。

    年报单元格常含换行与全角空格，「归属于上市公司股东的\n净利润」
    若不处理会与「归属于上市公司股东的净利润」被当成两个科目。
    """
    if not value:
        return ""
    s = value.replace("　", " ").replace("\xa0", " ")
    s = re.sub(r"\s*\n\s*", "", s)  # 换行直接消除，中文不需要空格连接
    return re.sub(r"[ \t]+", " ", s).strip()


def is_toc_line(line: str) -> bool:
    return bool(TOC_PAT.search(line))


# ── 标题层级 ────────────────────────────────────────────


class HeadingTracker:
    """维护当前标题路径，如「第三节 管理层讨论与分析 > 九、分季度主要财务数据」。"""

    def __init__(self) -> None:
        self.section: str | None = None
        self.subsection: str | None = None
        self.item: str | None = None

    def feed(self, line: str) -> bool:
        line = line.strip()
        if not line or is_toc_line(line):
            return False
        if SECTION_PAT.match(line):
            self.section = line[: 40]
            self.subsection = self.item = None
            return True
        if SUBSEC_PAT.match(line):
            self.subsection = line[:40]
            self.item = None
            return True
        if ITEM_PAT.match(line):
            self.item = line[:40]
            return True
        return False

    def path(self) -> str:
        return " > ".join(p for p in (self.section, self.subsection, self.item) if p)


# ── 量纲与口径 ──────────────────────────────────────────


def is_numeric_table(rows: list[list[str]]) -> bool:
    """判断表格是否承载数值。

    年报中大量表格是文字型的（备查文件目录、释义表、人员名单），
    给它们标注量纲毫无意义且会误导 —— 实测「备查文件目录」被标成
    「单位：元」。只有数值型表格才需要量纲。
    """
    total = filled = numeric = 0
    for row in rows:
        for cell in row:
            total += 1
            s = normalize_cell(cell)
            if not s:
                continue
            filled += 1
            if NUMERIC_CELL.match(s.replace(" ", "")):
                numeric += 1
    if filled < 4:
        return False
    return numeric / filled >= 0.2


def inline_unit(rows: list[list[str]]) -> tuple[str | None, str | None, int]:
    """提取表格内部的单位声明。

    持股表这类表格把「单位：股」写在表格第一行而非上方，
    实测若不提取，会被文档主导量纲「元」错误覆盖。
    返回 (单位, 币种, 需要跳过的行数)。
    """
    for idx, row in enumerate(rows[:2]):
        joined = " ".join(normalize_cell(c) for c in row)
        if not joined.strip():
            continue
        um = UNIT_PAT.search(joined)
        cm = CURRENCY_PAT.search(joined)
        if um or cm:
            # 该行只是单位声明（没有别的实义内容）时整行跳过
            residue = UNIT_PAT.sub("", CURRENCY_PAT.sub("", joined)).strip(" |　")
            skip = idx + 1 if len(residue) < 4 else 0
            return (um.group(1) if um else None, cm.group(1) if cm else None, skip)
    return None, None, 0


def extract_context(
    page: pymupdf.Page,
    bbox,
    page_text: str,
    inherited_unit: str | None,
    rows: list[list[str]],
) -> tuple[TableContext, int]:
    """多级回溯确定表格的单位与币种，返回 (上下文, 表内需跳过的行数)。

    优先级（高到低）：
      1. 表格内部声明 —— 最可信，是这张表自己写的
      2. 表格正上方   —— 实测约半数表格在此声明
      3. 整页范围
      4. 文档主导量纲 —— 仅对数值型金额表兜底

    实测只看表格上方会漏掉 29–46%，而无条件兜底又会给文字型表格
    标上错误量纲。两边都要防。
    """
    ctx = TableContext()
    numeric = is_numeric_table(rows)

    # 1) 表格内部声明，优先级最高
    in_unit, in_cur, skip_rows = inline_unit(rows)
    if in_unit:
        ctx.unit, ctx.unit_source = in_unit, "table_inline"
    if in_cur:
        ctx.currency = in_cur

    above = page.get_text(
        "text", clip=pymupdf.Rect(0, max(0, bbox[1] - LOOKUP_ABOVE), page.rect.width, bbox[1])
    ) or ""
    lines = [ln.strip() for ln in above.splitlines() if ln.strip()]

    # 2) 表格正上方
    for ln in reversed(lines):
        if not ctx.unit and (m := UNIT_PAT.search(ln)):
            ctx.unit, ctx.unit_source = m.group(1), "above"
        if not ctx.currency and (m := CURRENCY_PAT.search(ln)):
            ctx.currency = m.group(1)
        if ctx.unit and ctx.currency:
            break

    # 表题：上方最近一个非「适用/不适用」的实义行
    for ln in reversed(lines):
        if UNIT_PAT.search(ln) or CURRENCY_PAT.search(ln):
            continue
        if "适用" in ln or len(ln) < 4 or is_toc_line(ln):
            continue
        ctx.caption = ln[:60]
        break

    # 3) 整页范围 —— 仅对数值表，且只接受金额量纲
    #    （页面上别处的「单位：股」不应落到本表上）
    if not ctx.unit and numeric and (m := AMOUNT_UNIT_PAT.search(page_text)):
        ctx.unit, ctx.unit_source = m.group(1), "page"
    if not ctx.currency and (m := CURRENCY_PAT.search(page_text)):
        ctx.currency = m.group(1)

    # 4) 文档主导量纲兜底 —— 只给数值表，文字表不标单位
    if not ctx.unit and numeric and inherited_unit:
        ctx.unit, ctx.unit_source = inherited_unit, "inherited"

    return ctx, skip_rows


def dominant_unit(doc: pymupdf.Document, sample_pages: int = 60) -> str | None:
    """统计文档主导量纲，作为缺省继承值。

    实测同一份年报的量纲高度一致（茅台 166 次「元」，宁德 200 次「千元」），
    因此用众数兜底是安全的。
    """
    counter: dict[str, int] = {}
    for i in range(min(sample_pages, doc.page_count)):
        for m in UNIT_PAT.finditer(doc[i].get_text("text") or ""):
            counter[m.group(1)] = counter.get(m.group(1), 0) + 1
    return max(counter, key=counter.get) if counter else None


def scale_to_yuan(value: float, unit: str | None) -> float | None:
    """把数值按量纲折算为元，用于跨公司对比与指标入库。"""
    if unit not in UNIT_SCALE:
        return None
    return value * UNIT_SCALE[unit]


# ── 表格渲染 ────────────────────────────────────────────


def render_table(rows: list[list[str]], ctx: TableContext, heading: str) -> str:
    """把表格渲染为带量纲声明的 Markdown。

    单位写进正文首行是刻意设计：检索返回的是文本，
    模型看不到元数据。单位若只存在元数据里，数值就会被读错量级。
    """
    head_bits = []
    if ctx.unit:
        head_bits.append(f"单位：{ctx.unit}")
    if ctx.currency:
        head_bits.append(f"币种：{ctx.currency}")

    lines: list[str] = []
    if head_bits:
        lines.append("【" + "　".join(head_bits) + "】")
    if ctx.caption:
        lines.append(ctx.caption)
    elif heading:
        lines.append(heading.split(" > ")[-1])

    if not rows:
        return "\n".join(lines)

    width = max(len(r) for r in rows)
    norm = [[normalize_cell(c) for c in r] + [""] * (width - len(r)) for r in rows]

    header = norm[0]
    if not any(header):
        header = [f"列{i + 1}" for i in range(width)]
        body = norm
    else:
        body = norm[1:]

    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([" --- "] * width) + "|")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ── 主流程 ──────────────────────────────────────────────


def parse_pdf(path: str | Path, max_pages: int | None = None) -> ParsedDocument:
    path = Path(path)
    doc = pymupdf.open(path)
    limit = min(doc.page_count, max_pages) if max_pages else doc.page_count

    doc_unit = dominant_unit(doc)
    tracker = HeadingTracker()
    blocks: list[ParsedBlock] = []

    # 跨页表格检测所需的上一页状态
    prev_tail: ParsedBlock | None = None

    for i in range(limit):
        page = doc[i]
        page_text = page.get_text("text") or ""
        page_h = page.rect.height

        for line in page_text.splitlines():
            tracker.feed(line)
        heading = tracker.path()

        try:
            tables = list(page.find_tables())
        except Exception as exc:  # 个别页版面异常不应中断整篇解析
            logger.warning("第 %d 页表格检测失败：%s", i + 1, exc)
            tables = []

        table_rects = []
        for t in tables:
            try:
                rows = t.extract()
            except Exception:
                continue
            if not rows or len(rows) < 2:
                continue

            ctx, skip_rows = extract_context(page, t.bbox, page_text, doc_unit, rows)
            flags: list[str] = []
            if len(rows[0]) > 6:
                flags.append("wide_table")
            if t.bbox[3] > page_h - 80:
                flags.append("bottom_of_page")
            if t.bbox[1] < 150:
                flags.append("top_of_page")
            # 单位来源入 flags，供后续排查数值题错误时归因：
            # inherited 的可信度明显低于 table_inline / above
            if ctx.unit_source:
                flags.append(f"unit_src:{ctx.unit_source}")

            body_rows = rows[skip_rows:] if skip_rows else rows
            blk = ParsedBlock(
                kind="table",
                text=render_table(body_rows, ctx, heading),
                page_start=i + 1,
                page_end=i + 1,
                heading_path=heading,
                unit=ctx.unit,
                currency=ctx.currency,
                caption=ctx.caption,
                unit_source=ctx.unit_source,
                flags=flags,
                n_rows=len(rows),
                n_cols=len(rows[0]),
            )

            # 跨页合并：上页末尾的表与本页开头的表列数一致则视为同一张
            merged = False
            if (
                prev_tail is not None
                and "top_of_page" in flags
                and prev_tail.n_cols == blk.n_cols
                and prev_tail.page_end == i
            ):
                body = "\n".join(blk.text.splitlines()[3:])  # 去掉重复的量纲行与表头
                if body.strip():
                    prev_tail.text += "\n" + body
                prev_tail.page_end = i + 1
                prev_tail.n_rows += blk.n_rows
                if "cross_page" not in prev_tail.flags:
                    prev_tail.flags.append("cross_page")
                merged = True

            if not merged:
                blocks.append(blk)
            table_rects.append(pymupdf.Rect(t.bbox))
            prev_tail = (prev_tail if merged else blk) if "bottom_of_page" in flags else None

        # 正文：扣除表格区域后剩余的文本
        text_parts = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            brect = pymupdf.Rect(block["bbox"])
            if any(brect.intersects(r) for r in table_rects):
                continue
            seg = "".join(
                span.get("text", "")
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if seg and not is_toc_line(seg) and len(seg) > 8:
                text_parts.append(seg)

        if text_parts:
            blocks.append(
                ParsedBlock(
                    kind="text",
                    text="\n".join(text_parts),
                    page_start=i + 1,
                    page_end=i + 1,
                    heading_path=heading,
                    unit=None,
                    currency=None,
                )
            )

    doc.close()
    logger.info(
        "解析 %s：%d 页，%d 个块（表格 %d）",
        path.name,
        limit,
        len(blocks),
        sum(1 for b in blocks if b.kind == "table"),
    )
    return ParsedDocument(path=path, page_count=limit, blocks=blocks, doc_unit=doc_unit)
