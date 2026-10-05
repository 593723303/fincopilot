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
# 银行与保险的年报不写「单位：X」，而是在括号里用一句话声明，例如
#   （人民币百万元，特别注明除外）
#   （除特别注明外，货币单位均以人民币百万元列示）
# 只认「单位：」会让这类年报的表格全部没有量纲——实测招商银行
# 前 120 页 39 张表的单位无一被识别。
# 括号内不允许出现数字，用来挡掉正文里的「（人民币7,159,767百万元，占比…）」
# 这种叙述句——那是一个具体金额，不是整张表的量纲声明。
PROSE_UNIT_PAT = re.compile(
    rf"[（(][^（()）\n\d]{{0,30}}?(?:人民币)?\s*({AMOUNT_UNITS})"
)
# 判断单元格是否为数值：含千分位、小数、负号或括号负数
NUMERIC_CELL = re.compile(r"^[-－(（]?[\d,，]+(\.\d+)?[)）%]?$")
# 「第X节」是上交所主板的写法。A+H 公司多用「第X章」，
# 招商银行还用「2.1 本集团主要会计数据」这种小数编号——
# 只认「第X节」会让这类年报的 heading_path 整篇为空，
# 于是按标题筛表的逻辑（如「主要会计数据」）一张也筛不出来。
SECTION_PAT = re.compile(r"^第[一二三四五六七八九十]{1,3}[节章]\s*(.+)$")
# 2.1 / 2.1.1 式编号，后面必须跟实义文字，避免把「2.1」这样的孤立数字当标题
DECIMAL_SEC_PAT = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\s+(\S.{2,38})$")
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
    # 报表口径标题，如「合并现金流量表」「母公司现金流量表」
    scope: str | None = None


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
    statement_type: str | None = None
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
        if m := DECIMAL_SEC_PAT.match(line):
            # 2.1 视作次级标题，2.1.1 视作三级
            if m.group(1).count(".") == 1:
                self.subsection = line[:40]
                self.item = None
            else:
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


def infer_statement_type(*texts: str | None) -> str | None:
    """从标题与表题推断三大报表类型。

    作为检索时的标量过滤条件：「查利润表里的营业收入」这类问题
    靠它精确定位，不必指望向量检索区分报表种类。
    合并报表与母公司报表归为同一类型，由上下文区分。
    """
    joined = " ".join(t for t in texts if t)
    if not joined:
        return None
    if "资产负债表" in joined:
        return "balance"
    if "现金流量表" in joined:
        return "cashflow"
    if "利润表" in joined or "损益" in joined or "综合收益" in joined:
        return "income"
    if "所有者权益变动" in joined or "股东权益变动" in joined:
        return "equity"
    return None


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

    # 2.5) 表格上方的括号式声明（银行/保险年报的写法）
    if not ctx.unit and numeric:
        for ln in reversed(lines):
            if m := PROSE_UNIT_PAT.search(ln):
                ctx.unit, ctx.unit_source = m.group(1), "prose_above"
                break

    # 3) 整页范围 —— 仅对数值表，且只接受金额量纲
    #    （页面上别处的「单位：股」不应落到本表上）
    if not ctx.unit and numeric and (m := AMOUNT_UNIT_PAT.search(page_text)):
        ctx.unit, ctx.unit_source = m.group(1), "page"

    # 3.5) 整页范围的括号式声明。放在「单位：」之后，是因为显式声明更可信
    if not ctx.unit and numeric and (m := PROSE_UNIT_PAT.search(page_text)):
        ctx.unit, ctx.unit_source = m.group(1), "prose_page"
    if not ctx.currency and (m := CURRENCY_PAT.search(page_text)):
        ctx.currency = m.group(1)

    # 4) 文档主导量纲兜底 —— 只给数值表，文字表不标单位
    if not ctx.unit and numeric and inherited_unit:
        ctx.unit, ctx.unit_source = inherited_unit, "inherited"

    return ctx, skip_rows


#  一页至少要有这么多个带千分位的数字，才值得用文本策略再试一次
TABULAR_NUMBER_HINT = 12
# 报表口径最多向后继承几页。三大报表最长也就三四页，
# 超出这个跨度还在继承，继承到的一定是别的表
SCOPE_MAX_PAGES = 4
GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+")


def looks_tabular(page_text: str) -> bool:
    """这一页看起来是否含有成片的数字——用于决定要不要回退到文本策略。

    判据用「带千分位的数字」而不是所有数字：页码、年份、条款编号到处都是，
    而 1,745,679 这种写法基本只出现在金额表里。
    """
    return len(GROUPED_NUMBER.findall(page_text)) >= TABULAR_NUMBER_HINT


def dominant_unit(doc: pymupdf.Document, sample_pages: int = 60) -> str | None:
    """统计文档主导量纲，作为缺省继承值。

    实测同一份年报的量纲高度一致（茅台 166 次「元」，宁德 200 次「千元」），
    因此用众数兜底是安全的。
    """
    counter: dict[str, int] = {}
    for i in range(min(sample_pages, doc.page_count)):
        text = doc[i].get_text("text") or ""
        for m in UNIT_PAT.finditer(text):
            counter[m.group(1)] = counter.get(m.group(1), 0) + 1
        # 括号式声明同样计入，否则银行年报统计不出任何主导量纲，
        # 兜底一层形同虚设
        for m in PROSE_UNIT_PAT.finditer(text):
            counter[m.group(1)] = counter.get(m.group(1), 0) + 1
    return max(counter, key=counter.get) if counter else None


def scale_to_yuan(value: float, unit: str | None) -> float | None:
    """把数值按量纲折算为元，用于跨公司对比与指标入库。"""
    if unit not in UNIT_SCALE:
        return None
    return value * UNIT_SCALE[unit]


# ── 表格渲染 ────────────────────────────────────────────


# 合并 / 母公司是中文财报的第二个「量纲」问题，和元/千元一样必须跨页继承。
# 三大报表每张横跨两三页，只有起始页印着「合并现金流量表」，续页没有任何标题。
# 不继承的话，续页的块与另一口径的块在文本上完全一样——实测茅台 p64（合并）
# 与 p66（母公司）解析出的块逐字相同，模型只能靠猜，而两者的经营活动现金流
# 净额相差三倍（615 亿 vs 326 亿）。
STATEMENT_SCOPE = re.compile(
    r"(合并|母公司)(资产负债表|利润表|现金流量表|所有者权益变动表|综合收益表)"
)


YEAR_HEADER = re.compile(r"(20\d{2})\s*年")
# 列头里写着年份、但本身是变动率列，不是某一年的数值列
NOT_A_YEAR_COLUMN = re.compile(r"增减|增长|变动|幅度|%|％")
BREAKDOWN_MAX_ROWS = 24


def year_breakdown(header: list[str], body: list[list[str]]) -> list[str]:
    """把三年并列表按年份展开成一行一行的明细。

    这是整个解析里最后一处「别让模型去推断结构」的改造，
    与「单位写进正文」「口径写进正文」同一个思路。

    三年并列表长这样：

        | 主要会计数据 | 2025年 | 2024年 | 本期比上年同期增减(%) | 2023年 |
        | 营业收入 | 168,838,102,514.79 | 170,899,152,276.34 | -1.21 | 147,693,604,994.14 |

    要答对「2023 年营业收入」，模型得先数清楚第几列对应哪一年。
    实测它经常数错——问 2024 年给出 2025 年的数，问 2023 年给出 2024 年的数，
    而且这个错误在 flash 与 plus 上都出现，加提示词规则也压不住。

    干脆把对齐这件事在解析阶段做完，额外输出一段：

        【按年份】营业收入：2025年=168,838,102,514.79；2024年=170,899,152,276.34；2023年=147,693,604,994.14

    这样模型只需要做字符串匹配，不需要做列对齐。

    只在能确定对齐时才输出：列头里至少有两个年份，且数据行的非空单元格数
    与列头一致。对不齐就不输出——宁可没有，也不能给一段错的。
    """
    cols = {
        i: int(m.group(1))
        for i, c in enumerate(header)
        if (m := YEAR_HEADER.search(c)) and not NOT_A_YEAR_COLUMN.search(c)
    }
    if len(cols) < 2:
        return []
    # 同一年份出现在多列时整段放弃。资产负债表的列头是
    # 「2025年末 | 2025年初」，两列都归到 2025 年，展开出来会是
    # 「2025年=43,904,550；2025年=10,000,000」——读的人无从判断哪个是哪个。
    # 歧义的输出比没有更糟。
    if len(set(cols.values())) != len(cols):
        return []

    out: list[str] = []
    for row in body[:BREAKDOWN_MAX_ROWS]:
        name = next((c for c in row if c), "")
        if not name or YEAR_HEADER.search(name):
            continue
        parts = [
            f"{year}年={row[i]}"
            for i, year in sorted(cols.items(), key=lambda kv: -kv[1])
            if i < len(row) and NUMERIC_CELL.match(row[i].replace(" ", ""))
        ]
        if len(parts) >= 2:
            out.append(f"【按年份】{name}：" + "；".join(parts))
    return [""] + out if out else []


# ── 无边框表格的坐标重建 ────────────────────────────────

# 判断单元格粘连：数一数里面有几个千分位数字，两个以上就是没切开。
#
# 不要写成「两个数字之间夹着任意字符」那种正则——它会因回溯把**单个**长数字
# 拆成两个（349,079,082,852 可以拆成 349,079 与 082,852），
# 于是位数够多的正常数字全被判成粘连，紫金矿业的坐标重建因此一直被拒绝；
# 而加了「中间必须是非数字」的限制后，又会漏掉中间夹着小数的真粘连
# （303,639,957,153 14.96 293,403,242,878）。
# 贪婪地把每个千分位数字整体匹配出来再计数，两种情况都对。
# 行内的 y 容差与列内的 x 容差（pt）
ROW_TOL = 4.0
X_TOL = 3.0
# 一个 x 位置上至少要有这么多个数字，才算一个真正的数值列
MIN_COL_HITS = 3
NUMERIC_WORD = re.compile(r"^[-(（]?\d[\d,]*\.?\d*[)）%]?$")


def count_merged_cells(rows: list[list[str]], skip_first_col: bool = False) -> int:
    """数一数有多少个单元格发生了粘连。

    skip_first_col 用于坐标重建的产物：它的第 0 列是兜底列，
    收走所有不属于任何数值列的文字（包括带多个数字的正文行），
    按粘连计数天然吃亏，会把一张已经切得很好的表判成「更差」。
    """
    return sum(
        1
        for row in rows
        for cell in (row[1:] if skip_first_col else row)
        if cell and len(GROUPED_NUMBER.findall(normalize_cell(cell))) >= 2
    )


YEAR_LABEL = re.compile(r"20\d{2}\s*年")


def year_label_count(rows: list[list[str]]) -> int:
    """**单行内**最多有几个年份标签——用来判断列头有没有被打散。

    必须按行取最大值，不能把前几行的标签加总：坐标重建是按 y 切行的，
    遇到竖排或错位的列头会把「2025年」「2024年」「2023年」拆到不同行，
    总数没变、列头却已经不可用了。实测中国建筑就是这样——
    总数都是 3，而重建后的表头行只剩「2024年」一个。
    """
    return max(
        (sum(len(YEAR_LABEL.findall(normalize_cell(c))) for c in row) for row in rows[:3]),
        default=0,
    )


def count_numeric_cells(rows: list[list[str]]) -> int:
    """单独成格的数值有多少个——直接衡量「数字有没有被分开」。"""
    return sum(
        1
        for row in rows
        for cell in row
        if cell and NUMERIC_CELL.match(normalize_cell(cell).replace(" ", ""))
    )


def has_merged_cells(rows: list[list[str]]) -> bool:
    return count_merged_cells(rows) > 0


def numeric_columns(words) -> list[tuple[float, float]]:
    """数值列的 (左界, 右界)，按**右边界**聚类。

    必须按右边界而不是左边界：表格里的数字是右对齐的，
    同一列中位数不同的数字左边界相差很大
    （紫金矿业的 349,079,082,852 与 80,752,523,141 差了一位），
    右边界却严格相同。按左边界聚类会把同一列拆成两列，
    于是同一科目的本年数有时落在第 1 列、有时落在第 2 列。

    只用数值词定列：正文的坐标散乱，还会横跨好几列把列间空隙填平。
    """
    nums = [w for w in words if NUMERIC_WORD.match(w[4])]
    if not nums:
        return []
    nums.sort(key=lambda w: w[2])
    groups: list[list] = [[nums[0]]]
    for w in nums[1:]:
        if w[2] - groups[-1][-1][2] <= X_TOL:
            groups[-1].append(w)
        else:
            groups.append([w])
    return [
        (min(w[0] for w in g), max(w[2] for w in g))
        for g in groups
        if len(g) >= MIN_COL_HITS
    ]


def rebuild_table_by_words(page: pymupdf.Page, bbox) -> list[list[str]]:
    """从词坐标重建表格。切不开就返回空，由调用方保留原结果。

    只在 find_tables 把整列塞进一个单元格时才调用——
    银行与 A+H 公司的年报大量使用无边框表格，pymupdf 的划线检测
    和文本策略都切不开，但 PDF 里的词坐标本身是整齐的。
    """
    words = page.get_text("words", clip=pymupdf.Rect(bbox))
    cols = numeric_columns(words)
    if len(cols) < 2:
        return []

    def col_of(w) -> int:
        """词落在哪一列。落不进任何数值列的归到第 0 列（标签列）。"""
        center = (w[0] + w[2]) / 2
        for i, (left, right) in enumerate(cols, start=1):
            if left - X_TOL <= center <= right + X_TOL:
                return i
        return 0

    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))
    lines: list[list] = []
    for w in words:
        if lines and abs(w[1] - lines[-1][0][1]) <= ROW_TOL:
            lines[-1].append(w)
        else:
            lines.append([w])

    out: list[list[str]] = []
    for ln in lines:
        cells = [""] * (len(cols) + 1)
        for w in sorted(ln, key=lambda w: w[0]):
            i = col_of(w)
            cells[i] = (cells[i] + " " + w[4]).strip()
        if any(c.strip() for c in cells):
            out.append(cells)
    return out


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
    # 口径写在 caption 之前：检索返回的是文本，模型看不到元数据，
    # 口径若只存在元数据里，合并与母公司的数就会被混用
    if ctx.scope:
        lines.append(ctx.scope)
    if ctx.caption:
        lines.append(ctx.caption)
    elif heading and not ctx.scope:
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

    lines.extend(year_breakdown(header, body))
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
    # 报表口径跨页继承；章节一变就失效，避免把「合并」带进财务附注
    scope_title: str | None = None
    scope_heading: str | None = None
    scope_page: int | None = None

    for i in range(limit):
        page = doc[i]
        page_text = page.get_text("text") or ""
        page_h = page.rect.height

        try:
            raw_tables = list(page.find_tables())
            # 无边框表格：默认策略靠划线识别，整页找不到任何表。
            # 实测中国平安年报 370 页里只有 6 页能检出表格，
            # 而第 187 页的合并资产负债表（53 行 × 4 列）赫然在列——
            # 它只是没画框线。改用文本对齐策略就能抽出来。
            # 仅在默认策略颗粒无收、且页面确实有成片数字时才回退，
            # 因为文本策略对普通正文页会切出大量伪表格。
            if not raw_tables and looks_tabular(page_text):
                raw_tables = list(page.find_tables(strategy="text"))
                if raw_tables:
                    logger.debug("第 %d 页改用文本策略抽出 %d 张表", i + 1, len(raw_tables))
        except Exception as exc:  # 个别页版面异常不应中断整篇解析
            logger.warning("第 %d 页表格检测失败：%s", i + 1, exc)
            raw_tables = []

        # 先筛出有效表格并记录其区域，正文需要扣除这些区域
        valid: list[tuple] = []
        table_rects: list[pymupdf.Rect] = []
        for t in raw_tables:
            try:
                rows = t.extract()
                # 整列被塞进一个单元格时，改用词坐标重建。
                # 只在确实粘连时启用，正常的表一律走原路径。
                merged_before = count_merged_cells(rows)
                if merged_before:
                    rebuilt = rebuild_table_by_words(page, t.bbox)
                    # 判据直接衡量目的：**数字有没有被分进各自的单元格**。
                    # 先后试过两种更直觉的判据，都不行：
                    #   「重建后必须零粘连」——一行没切开就丢掉整张好表
                    #   「重建后粘连更少」——重建表的兜底列会收走带数字的正文行，
                    #                       计数天然吃亏，紫金矿业因此被判成更差
                    # 还要守住列头：重建是按坐标切的，遇到跨列的长标题会把
                    # 「2025年」这类标签挤进别的格子甚至丢掉。实测中国建筑的
                    # 摘要表原本有 2025/2024/2023 三个年份列，重建后只剩 2024，
                    # 数据切得再干净也没用——年份对不上就不能出题。
                    better = (
                        rebuilt
                        and count_numeric_cells(rebuilt) > count_numeric_cells(rows)
                        and count_merged_cells(rebuilt, skip_first_col=True) < merged_before
                        and year_label_count(rebuilt) >= year_label_count(rows)
                    )
                    if better:
                        logger.debug(
                            "第 %d 页表格单元格粘连，按坐标重建（%d 行 × %d 列）",
                            i + 1,
                            len(rebuilt),
                            len(rebuilt[0]),
                        )
                        rows = rebuilt
            except Exception:
                continue
            if not rows or len(rows) < 2:
                continue
            valid.append((t, rows))
            table_rects.append(pymupdf.Rect(t.bbox))

        # 按页面纵向位置把正文段与表格混合排序。
        # 必须如此：标题追踪器若先吃完整页文本再处理表格，
        # 表格拿到的会是整页最后一个标题 —— 实测出现过
        # 「heading=十、非经常性损益」而内容是「九、分季度财务数据」的错配。
        items: list[tuple[float, str, object]] = []
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
            if seg:
                items.append((brect.y0, "text", seg))
        for t, rows in valid:
            items.append((t.bbox[1], "table", (t, rows)))
        items.sort(key=lambda x: x[0])

        pending_text: list[str] = []
        pending_heading = tracker.path()

        def flush_text(page_no: int = i + 1) -> None:
            nonlocal pending_text, pending_heading
            if pending_text:
                blocks.append(
                    ParsedBlock(
                        kind="text",
                        text="\n".join(pending_text),
                        page_start=page_no,
                        page_end=page_no,
                        heading_path=pending_heading,
                    )
                )
            pending_text = []
            pending_heading = tracker.path()

        for _y, kind, payload in items:
            if kind == "text":
                seg = payload
                for line in seg.splitlines():
                    tracker.feed(line)
                    hit = STATEMENT_SCOPE.search(line)
                    if hit:
                        scope_title = hit.group(0)
                        scope_heading = tracker.path()
                        scope_page = i
                if not pending_text:
                    pending_heading = tracker.path()
                if not is_toc_line(seg) and len(seg) > 8:
                    pending_text.append(seg)
                continue

            # 遇到表格：先结清此前累积的正文，再以当前标题状态处理表格
            flush_text()
            t, rows = payload
            heading = tracker.path()

            ctx, skip_rows = extract_context(page, t.bbox, page_text, doc_unit, rows)
            # 章节一变就丢弃继承来的口径：财务附注里也会出现「合并资产负债表」
            # 这几个字，继续沿用会把附注表错标成三大报表
            # 口径的有效期有两道闸：章节一变即失效，以及最多跨 SCOPE_MAX_PAGES 页。
            # 只靠章节是不够的——中国平安的年报章节标题识别不出来，
            # 继承会一路带到财务附注，把 35 张附注表全标成「合并现金流量表」。
            # 三大报表再长也就三四页，超出这个跨度的继承必然是错的。
            expired = scope_page is None or (i - scope_page) > SCOPE_MAX_PAGES
            if scope_title and scope_heading == heading and not expired:
                ctx.scope = scope_title
                heading = f"{heading} > {scope_title}"
            else:
                scope_title = scope_heading = scope_page = None
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
                statement_type=infer_statement_type(heading, ctx.caption),
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
            prev_tail = (prev_tail if merged else blk) if "bottom_of_page" in flags else None

        # 页面末尾残留的正文
        flush_text()

    doc.close()
    logger.info(
        "解析 %s：%d 页，%d 个块（表格 %d）",
        path.name,
        limit,
        len(blocks),
        sum(1 for b in blocks if b.kind == "table"),
    )
    return ParsedDocument(path=path, page_count=limit, blocks=blocks, doc_unit=doc_unit)
