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

from app.rag.pdf_parser import parse_pdf

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
    "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额",
    "归属于上市公司股东的净资产",
    "总资产",
    "资产总计",
]

YEAR_IN_HEADER = re.compile(r"(20\d{2})\s*年")
NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")


def parse_row(line: str) -> tuple[str, list[str]] | None:
    """把 Markdown 表格行切成 (科目名, 其余单元格)。"""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) < 2:
        return None
    return cells[0], cells[1:]


def year_columns(header_cells: list[str]) -> dict[int, int]:
    """列号 → 年份。只认列头里明写年份的列。"""
    out: dict[int, int] = {}
    for idx, cell in enumerate(header_cells):
        if m := YEAR_IN_HEADER.search(cell):
            out[idx] = int(m.group(1))
    return out


def corroborate(doc_pdf: pymupdf.Document, page_no: int, raw: str) -> bool:
    """用**原始页面文本**核对这个数字确实印在该页上。

    这一步不能省。底稿是解析器抽的，而被评估的系统用的也是同一个解析器——
    解析器要是把列读错了，标准答案和系统回答会一起错，评估照样全绿。
    `get_text()` 走的是另一条代码路径（不经过 find_tables 与表格重建），
    能独立地证伪「这个数压根不在这一页上」。

    注意它只能证伪，不能证实归属：同一页上 2024 年的数也在，
    所以年份对不对仍然要人工看列头。
    """
    idx = page_no - 1
    if not 0 <= idx < doc_pdf.page_count:
        return False
    text = doc_pdf[idx].get_text() or ""
    return raw in text or raw.replace(",", "") in text.replace(",", "")


def extract(pdf: Path) -> list[dict]:
    doc = parse_pdf(pdf)
    doc_pdf = pymupdf.open(pdf)
    rows: list[dict] = []
    for blk in doc.blocks:
        if blk.kind != "table" or not SUMMARY_HEADING.search(blk.heading_path or ""):
            continue
        lines = [ln for ln in blk.text.splitlines() if ln.startswith("|")]
        if len(lines) < 3:
            continue
        header = parse_row(lines[0])
        if not header:
            continue
        cols = year_columns(header[1])
        if not cols:
            continue  # 列头没有年份的表不要——年份归属无从验证
        for line in lines[2:]:  # 跳过分隔行
            parsed = parse_row(line)
            if not parsed:
                continue
            name, cells = parsed
            if name not in WANTED:
                continue
            for idx, year in cols.items():
                if idx >= len(cells):
                    continue
                raw = cells[idx]
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
                        "corroborated": corroborate(doc_pdf, blk.page_start, raw),
                    }
                )
    return rows


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
    for pdf in pdfs:
        code = pdf.name.split("_")[0]
        rows = extract(pdf)
        all_rows[code] = rows
        units = {r["unit"] for r in rows}
        years = sorted({r["year"] for r in rows})
        bad = [r for r in rows if not r["corroborated"]]
        flag = f"  ⚠ {len(bad)} 条未在原页文本中找到" if bad else ""
        print(
            f"{code} {pdf.name[:30]:32s} {len(rows):3d} 条  年份{years}  单位{units or '—'}{flag}"
        )

    out = OUT_DIR / "key_metrics.json"
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n底稿写入 {out}")
    print("注意：这是底稿，每条进评估集前必须人工核对页码与口径。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
