"""从各份年报的「主要会计数据」表里抽取核心指标，供扩充评估集时作底稿。

    python -m scripts.extract_key_metrics              处理 data/raw 下全部 PDF
    python -m scripts.extract_key_metrics 600036       只处理指定公司

为什么挑这张表：它是年报里**唯一把年份直接写进列头**的汇总表
（`| 主要会计数据 | 2025年 | 2024年 | 本期比上年同期增减(%) | 2023年 |`），
因此「哪个数属于哪一年」有据可查，不必靠位置推断。三大报表只有
「本期/上期」，拿来出题容易把年份标错——这正是 M5 踩过的坑。

输出是**底稿而非成品**：每条都带页码与原始行文本，必须人工核对后
才能进评估集。自动抽取的数字直接当标准答案，等于用模型校模型。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

from app.rag.pdf_parser import ParsedDocument, parse_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "tmp"

# 这张表的标题在各家年报里写法略有差异
SUMMARY_HEADING = re.compile(r"主要会计数据|主要财务数据|财务概要|主要会计数据和财务指标")

# 想抽的科目。写全称，避免「净利润」匹配到「归属于…净利润」
WANTED = [
    "营业收入",
    "营业总收入",
    "利润总额",
    "归属于上市公司股东的净利润",
    "归属于本公司股东的净利润",
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额",
    "归属于上市公司股东的净资产",
    "总资产",
    "资产总计",
]

YEAR_IN_HEADER = re.compile(r"(20\d{2})\s*年")
# 「本期比上年同期增减(%)」「2025年比2024年增减(%)」这类列头里也写着年份，
# 但它是变动率列，不是某一年的数值列。不排除掉，中国神华的增减列
# 会被当成 2025 年列，取出来的「2025 年营业收入」其实是 -13.2。
NOT_A_YEAR_COLUMN = re.compile(r"增减|增长|变动|幅度|%|％")
# 「调整后/调整前」「重述后/重述前」子列头：上一年度被拆成两列，
# 数据行会比表头多出几列
RESTATED_SUBHEADER = re.compile(r"调整[前后]|重述[前后]|重列[前后]")
NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")


#  科目名后缀里的量纲：「营业收入（千元）」要能和「营业收入」对上
NAME_SUFFIX = re.compile(r"[（(][^）)]*[）)]\s*$")


def parse_row(line: str) -> tuple[str, list[str]] | None:
    """把表格行切成 (科目名, 数值单元格)，**按非空单元格的顺序**切。

    不能按原始列号切。深交所格式的表合并单元格多，同一张表里
    表头与数据行的列位是错开的：

        表头   |  |  |  |  | 2025 年 |  |  | 2024 年 | ...   → 年份在第 4、7 列
        数据行 |  | 营业收入（千元） |  | 456,451,731 | ...   → 数值在第 3、6 列

    按列号取会整列错位，取出来的「2025 年营业收入」其实是空值。
    按非空顺序取则两种格式都对得上。
    """
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    filled = [i for i, c in enumerate(cells) if c]
    if len(filled) < 2:
        return None
    name = cells[filled[0]]
    return NAME_SUFFIX.sub("", name).strip(), [cells[i] for i in filled[1:]]


def header_years(line: str) -> tuple[int, dict[int, int]] | None:
    """解析表头，返回 (数值列数, 第几个数值列 → 年份)。

    数值列从**第一个带年份的单元格**开始数：上交所格式表头首格是
    「主要会计数据」这样的列名，深交所格式首格是空的，
    从年份起算才能让两种格式的列数与数据行对齐。
    """
    cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
    first_year = next((i for i, c in enumerate(cells) if YEAR_IN_HEADER.search(c)), None)
    if first_year is None:
        return None
    value_cells = cells[first_year:]
    years = {
        i: int(m.group(1))
        for i, c in enumerate(value_cells)
        if (m := YEAR_IN_HEADER.search(c)) and not NOT_A_YEAR_COLUMN.search(c)
    }
    return len(value_cells), years


# 年份列里放的是千分位大数，增减率列放的是不带逗号的小数
GROUPED = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def align_by_shape(
    cells: list[str], cols: dict[int, int]
) -> tuple[dict[int, int], list[str]] | None:
    """列数对不上时，按数值形态再对齐一次。

    紫金矿业的表头把「本期比上年同期增减(%)」并进了首格，于是表头只数出
    3 个年份列，而数据行有 4 个值——按列数判定会整行丢弃。

    但两类值的形态截然不同：年份列是 349,079,082,852 这样的千分位大数，
    增减率列是 14.96 这样不带逗号的小数。只要千分位数的个数恰好等于
    年份列的个数，就能按顺序一一对上，不需要知道增减列夹在第几位。

    对不上就返回 None——宁可不出题，不能对错年份。
    """
    grouped = [c for c in cells if GROUPED.match(c.replace(" ", ""))]
    if len(grouped) != len(cols) or not grouped:
        return None
    years_in_order = [cols[i] for i in sorted(cols)]
    return {i: y for i, y in enumerate(years_in_order)}, grouped


def corroborate(doc_pdf: pymupdf.Document, page_start: int, raw: str, page_end: int = 0) -> bool:
    """用**原始页面文本**核对这个数字确实印在这张表覆盖的页上。

    这一步不能省。底稿是解析器抽的，而被评估的系统用的也是同一个解析器——
    解析器要是把列读错了，标准答案和系统回答会一起错，评估照样全绿。
    `get_text()` 走的是另一条代码路径（不经过 find_tables 与表格重建），
    能独立地证伪「这个数压根不在这张表里」。

    必须查 page_start..page_end 整个区间，不能只查 page_start：
    跨页合并的表保留首页页码，而目标行往往在续页上——
    实测茅台合并现金流量表 page_start=64，「经营活动产生的现金流量净额」
    那一行印在 65 页，只查首页会把它判成「对不上」而整条丢弃，
    口径辨析题因此一道都生成不出来。

    它只能证伪，不能证实归属：同一页上别的年份、别的口径的数也在，
    所以年份与口径仍然要靠列头与表标题。
    """
    last = max(page_end or page_start, page_start)
    for page_no in range(page_start, last + 1):
        idx = page_no - 1
        if not 0 <= idx < doc_pdf.page_count:
            continue
        text = doc_pdf[idx].get_text() or ""
        if raw in text or raw.replace(",", "") in text.replace(",", ""):
            return True
    return False


def extract(doc: ParsedDocument, doc_pdf: pymupdf.Document) -> list[dict]:
    rows: list[dict] = []
    for blk in doc.blocks:
        # caption 也要认：紫金矿业的摘要表标题「近三年主要会计数据」
        # 是表上方的一行文字，没被识别成章节标题，只落在 caption 里
        where = f"{blk.heading_path or ''} {blk.caption or ''}"
        if blk.kind != "table" or not SUMMARY_HEADING.search(where):
            continue
        lines = [ln for ln in blk.text.splitlines() if ln.startswith("|")]
        if len(lines) < 3:
            continue
        parsed_header = header_years(lines[0])
        if not parsed_header:
            continue  # 列头没有年份的表不要——年份归属无从验证
        n_values, cols = parsed_header
        # 表里有没有「调整后/调整前」这类子列头，决定列数对不上时能否解释
        restated = any(RESTATED_SUBHEADER.search(ln) for ln in lines[:4])
        for line in lines[2:]:  # 跳过分隔行
            parsed = parse_row(line)
            if not parsed:
                continue
            name, cells = parsed
            if name not in WANTED:
                continue
            # 列数对不上只在一种情况下还敢取数：表里确有「调整后/调整前」
            # （重述后/重述前）子列头——长江电力、中国神华就是这样，
            # 上一年度被拆成两列，数据行因此比表头多出几列。
            # 这时中间各列对应哪一年无从判断，不猜；但**最左的数值列一定是当年**
            # （当年不存在调整前后之分），只取这一列，其余整行放弃。
            # 没有子列头却列数不符，说明是别的原因（科目名跨行折断、
            # 合并单元格错位），那就整行弃用——猜错了也看不出来。
            if len(cells) == n_values:
                take = {i: y for i, y in cols.items()}
                values = cells
            elif (aligned := align_by_shape(cells, cols)) is not None:
                take, values = aligned
            elif restated and 0 in cols:
                take, values = {0: cols[0]}, cells
            else:
                continue
            for idx, year in take.items():
                raw = values[idx]
                if not NUMBER.fullmatch(raw.replace(" ", "")):
                    continue
                rows.append(
                    {
                        "metric": name,
                        "year": year,
                        "raw": raw,
                        "value": float(raw.replace(",", "")),
                        "unit": blk.unit,
                        "page": blk.page_start,
                        "heading": blk.heading_path,
                        "source_line": line[:160],
                        "partial_row": len(cells) != n_values,
                        "corroborated": corroborate(doc_pdf, blk.page_start, raw, blk.page_end),
                    }
                )
    return rows


# 合并与母公司差异最大、也最容易被问到的几个科目
SCOPE_METRICS = [
    "经营活动产生的现金流量净额",
    "营业收入",
    "营业总收入",
    "资产总计",
    "净利润",
]
SCOPE_TITLE = re.compile(r"^(合并|母公司)(资产负债表|利润表|现金流量表)$")
# 两个口径的数值差异要足够大，否则这道题区分不出模型有没有看口径
SCOPE_MIN_GAP = 0.05


def extract_scope_pairs(doc: ParsedDocument, doc_pdf: pymupdf.Document) -> list[dict]:
    """抽取「同一科目在合并口径与母公司口径下的两个数」。

    这是本项目差异化所在：两张表在解析前逐字相同，口径标签是 ADR-019
    加上去的。拿它出题，等于直接考「模型有没有看口径」——
    而这类题从「主要会计数据」汇总表里是抽不出来的，那张表只有合并口径。
    """
    # (报表类型, 科目) → {口径: 行}
    found: dict[tuple[str, str], dict[str, dict]] = {}
    for blk in doc.blocks:
        if blk.kind != "table":
            continue
        last = (blk.heading_path or "").split(" > ")[-1]
        m = SCOPE_TITLE.match(last)
        if not m or not blk.unit:
            continue
        scope, statement = m.group(1), m.group(2)
        for line in blk.text.splitlines():
            if not line.startswith("|"):
                continue
            parsed = parse_row(line)
            if not parsed:
                continue
            name, cells = parsed
            if name not in SCOPE_METRICS:
                continue
            for raw in cells:  # 取第一个能解析成数字的单元格 = 本期
                cleaned = raw.replace(" ", "")
                if not cleaned or not NUMBER.fullmatch(cleaned):
                    continue
                found.setdefault((statement, name), {}).setdefault(
                    scope,
                    {
                        "raw": raw,
                        "value": float(cleaned.replace(",", "")),
                        "unit": blk.unit,
                        "page": blk.page_start,
                        "page_end": blk.page_end,
                        "heading": blk.heading_path,
                        "corroborated": corroborate(doc_pdf, blk.page_start, raw, blk.page_end),
                    },
                )
                break

    out: list[dict] = []
    for (statement, name), by_scope in found.items():
        if {"合并", "母公司"} - by_scope.keys():
            continue
        a, b = by_scope["合并"], by_scope["母公司"]
        if not (a["corroborated"] and b["corroborated"]):
            continue
        if a["unit"] != b["unit"]:
            continue  # 量纲不同就不是一个可比的对子，弃用
        if a["value"] == 0 or abs(a["value"] - b["value"]) / abs(a["value"]) < SCOPE_MIN_GAP:
            continue  # 两个数差不多，答错也看不出来
        out.append({"statement": statement, "metric": name, "合并": a, "母公司": b})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("codes", nargs="*", help="股票代码，留空则处理全部")
    args = parser.parse_args()

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if args.codes:
        pdfs = [p for p in pdfs if any(p.name.startswith(c) for c in args.codes)]
    if not pdfs:
        print(f"{RAW_DIR} 下没有匹配的 PDF")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    all_rows: dict[str, list[dict]] = {}
    all_pairs: dict[str, list[dict]] = {}
    for pdf in pdfs:
        code = pdf.name.split("_")[0]
        doc_pdf = pymupdf.open(pdf)
        parsed = parse_pdf(pdf)  # 解析很慢，两个抽取器共用这一次结果
        rows = extract(parsed, doc_pdf)
        all_rows[code] = rows
        all_pairs[code] = extract_scope_pairs(parsed, doc_pdf)
        units = {r["unit"] for r in rows}
        years = sorted({r["year"] for r in rows})
        bad = [r for r in rows if not r["corroborated"]]
        flag = f"  ⚠ {len(bad)} 条未在原页文本中找到" if bad else ""
        print(
            f"{code} {pdf.name[:28]:30s} {len(rows):3d} 条  年份{years}  "
            f"单位{units or '—'}  口径对 {len(all_pairs[code])}{flag}"
        )

    out = OUT_DIR / "key_metrics.json"
    out.write_text(
        json.dumps({"metrics": all_rows, "scope_pairs": all_pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n底稿写入 {out}")
    print("注意：这是底稿，每条进评估集前必须人工核对页码与口径。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
